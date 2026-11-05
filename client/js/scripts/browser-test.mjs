// End-to-end test of the web client in headless Chromium: the example page, with a fake
// microphone that loops a fixture clip. Runs on the dev server and on a production build.
//   pnpm test:browser   (once: pnpm exec playwright install chromium)
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { build, createServer, preview } from 'vite';

const root = fileURLToPath(new URL('../example', import.meta.url));
const clip = (name) =>
  fileURLToPath(new URL(`../../python/tests/fixtures/${name}.wav`, import.meta.url));

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
    await page.goto(url);
    await page.click('#start');
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'listening',
      null,
      { timeout: 30_000 },
    );
    let best = 0;
    for (const end = Date.now() + seconds * 1000; Date.now() < end; ) {
      await page.waitForTimeout(100);
      best = Math.max(best, Number.parseFloat((await page.textContent('#score')).slice(6)));
    }
    const detections = await page.$$eval('#log li', (items) => items.length);
    await page.click('#start'); // stop: the detections() loop ends
    await page.waitForFunction(() => document.getElementById('status')?.textContent === 'stopped');
    assert.deepEqual(errors, []);
    return { detections, best };
  } finally {
    await browser.close();
  }
}

async function check(label, url) {
  const positive = await listen(url, clip('positive'));
  const negative = await listen(url, clip('negative'));
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
const server = await preview({ root, preview: { port: 0 }, build: { outDir }, logLevel: 'warn' });
try {
  await check('build', server.resolvedUrls.local[0]);
} finally {
  await server.close();
}
console.log('browser tests passed');
