// End-to-end test of the web client in headless Chromium: the example page, with a fake
// microphone that loops a clip. Runs on the dev server and on a production build. This checks the
// plumbing (capture, resampling, detections, stop); model accuracy is covered by test/*.test.ts.
//   pnpm test:browser   (once: pnpm exec playwright install chromium)
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { build, createServer, preview } from 'vite';

const root = fileURLToPath(new URL('../example', import.meta.url));
const clip = (name) =>
  fileURLToPath(new URL(`../../python/tests/fixtures/${name}.wav`, import.meta.url));

/**
 * 3 s of tones and noise, 16 kHz mono: clearly not speech. (negative.wav is hard speech that
 * scores up to about 0.5 at some frame alignments, and the browser's alignment is random.)
 */
function nonSpeech() {
  const n = 48000;
  const wav = Buffer.alloc(44 + 2 * n);
  wav.write('RIFF', 0);
  wav.writeUInt32LE(36 + 2 * n, 4);
  wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20); // PCM
  wav.writeUInt16LE(1, 22); // mono
  wav.writeUInt32LE(16000, 24);
  wav.writeUInt32LE(32000, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36);
  wav.writeUInt32LE(2 * n, 40);
  let seed = 1;
  for (let i = 0; i < n; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const hz = [440, 880, 1320][Math.floor(i / 16000)];
    const x = 0.3 * Math.sin((2 * Math.PI * hz * i) / 16000) + 0.05 * (seed / 2 ** 32 - 0.5);
    wav.writeInt16LE(Math.round(x * 32767), 44 + 2 * i);
  }
  const path = join(tmpdir(), 'livekit-wakeword-nonspeech.wav');
  writeFileSync(path, wav);
  return path;
}

/** Detections and the highest score seen in `seconds` of `wav` looped through the microphone. */
async function listen(url, wav, seconds = 8) {
  const browser = await chromium.launch({
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${wav}`,
    ],
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(url); // listening starts by itself (after a tap, if autoplay is blocked)
    await page.waitForFunction(() => document.getElementById('status')?.textContent !== 'loading');
    await page.mouse.click(1, 1);
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'listening',
      null,
      { timeout: 30_000 },
    );
    let best = 0;
    for (const end = Date.now() + seconds * 1000; Date.now() < end; ) {
      await page.waitForTimeout(100);
      const score = await page.textContent('output[data-model="hey_livekit"]');
      best = Math.max(best, Number.parseFloat(score));
    }
    const detections = await page.$$eval('#log li', (items) => items.length);
    await page.click('#start'); // pause: the detections() loop ends
    await page.waitForFunction(() => document.getElementById('status')?.textContent === 'paused');
    assert.deepEqual(errors, []);
    return { detections, best };
  } finally {
    await browser.close();
  }
}

async function check(label, url) {
  const positive = await listen(url, clip('positive'));
  const negative = await listen(url, nonSpeech());
  console.log(label, { positive, negative });
  // the 2.2 s clip loops: about one detection per loop, at most one per 2 s debounce
  assert.ok(positive.detections >= 2 && positive.detections <= 4, 'positive detections');
  assert.equal(negative.detections, 0, 'negative detections');
  assert.ok(negative.best < 0.5);
}

const dev = await createServer({ root, server: { port: 0 }, logLevel: 'warn' });
await dev.listen();
try {
  await check('dev', dev.resolvedUrls.local[0]);
} finally {
  await dev.close();
}

const outDir = fileURLToPath(new URL('../example/dist', import.meta.url));
await build({ root, logLevel: 'warn', build: { outDir, emptyOutDir: true } });
// served under a subpath, as GitHub Pages serves it (the build uses relative URLs)
const server = await preview({
  root,
  base: '/livekit-wakeword/',
  preview: { port: 0 },
  build: { outDir },
  logLevel: 'warn',
});
try {
  await check('build', server.resolvedUrls.local[0]);
} finally {
  await server.close();
}
console.log('browser tests passed');
