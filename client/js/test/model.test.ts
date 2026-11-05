// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { WakeWordModel as NodeModel } from '../src/node.js';
import { WakeWordModel as WebModel } from '../src/web.js';
import { HEY_LIVEKIT, noise, parity, readWav, TINY } from './helpers.js';

// the same tests on both bindings: onnxruntime-node, and onnxruntime-web (wasm) as browsers run it
describe.each([
  ['node', NodeModel],
  ['web', WebModel],
] as const)('WakeWordModel (%s)', (_, Model) => {
  it('matches the Python client step by step', async () => {
    for (const clip of ['positive', 'negative'] as const) {
      const [, audio] = readWav(`${clip}.wav`);
      const model = await Model.create([HEY_LIVEKIT]);
      const scores: number[] = [];
      for (let i = 0; i < audio.length; i += 1280) {
        scores.push((await model.predict(audio.subarray(i, i + 1280))).hey_livekit);
      }
      expect(scores.length).toBe(parity.scores[clip].length);
      for (const [i, s] of scores.entries()) expect(s).toBeCloseTo(parity.scores[clip][i], 4);
      await model.release();
    }
  });

  it('detects the positive clip and not the negative one', async () => {
    const model = await Model.create([HEY_LIVEKIT]);
    const [rate, positive] = readWav('positive.wav');
    expect(rate).toBe(16000);
    expect((await model.predictClip(positive)).hey_livekit).toBeGreaterThanOrEqual(0.5);
    expect((await model.predictClip(readWav('negative.wav')[1])).hey_livekit).toBeLessThan(0.5);
    // < one 80 ms step
    expect((await model.predictClip(positive.subarray(0, 800))).hey_livekit).toBe(0);
  });

  it('gives the same steps whatever the chunk sizes', async () => {
    const audio = noise(16000 * 3);
    const a = await Model.create([TINY]);
    const frames: number[] = [];
    for (let i = 0; i < audio.length; i += 1280) {
      frames.push((await a.predict(audio.subarray(i, i + 1280))).streaming_tiny);
    }
    const b = await Model.create([TINY]);
    let best = 0;
    for (let pos = 0, k = 0; pos < audio.length; k++) {
      const size = [700, 1860, 1280, 333, 2227][k % 5];
      best = Math.max(best, (await b.predict(audio.subarray(pos, pos + size))).streaming_tiny);
      pos += size;
    }
    expect(best).toBeCloseTo(Math.max(...frames), 6);
    expect(frames.every((s) => s >= 0 && s <= 1)).toBe(true);
  });

  it('carries state between calls until reset', async () => {
    const audio = noise(1280 * 20);
    const model = await Model.create([TINY]);
    const run = async () => {
      const out: number[] = [];
      for (let i = 0; i < audio.length; i += 1280) {
        out.push((await model.predict(audio.subarray(i, i + 1280))).streaming_tiny);
      }
      return out;
    };
    const first = await run();
    const second = await run();
    expect(second).not.toEqual(first); // the model remembers what it heard
    await model.reset();
    const again = await run();
    for (const [i, s] of again.entries()) expect(s).toBeCloseTo(first[i], 6);
  });

  it('queues overlapping calls in order', async () => {
    const audio = noise(1280 * 10);
    const serial = await Model.create([TINY]);
    const expected: number[] = [];
    for (let i = 0; i < audio.length; i += 1280) {
      expected.push((await serial.predict(audio.subarray(i, i + 1280))).streaming_tiny);
    }
    const model = await Model.create([TINY]);
    const calls: Promise<Record<string, number>>[] = [];
    for (let i = 0; i < audio.length; i += 1280)
      calls.push(model.predict(audio.subarray(i, i + 1280)));
    const got = (await Promise.all(calls)).map((s) => s.streaming_tiny);
    for (const [i, s] of got.entries()) expect(s).toBeCloseTo(expected[i], 6);
  });

  it('takes int16 and float input alike', async () => {
    const pcm = readWav('positive.wav')[1];
    const float = Float32Array.from(pcm, (v) => v / 32768);
    const model = await Model.create([HEY_LIVEKIT]);
    const a = (await model.predictClip(pcm)).hey_livekit;
    expect((await model.predictClip(float)).hey_livekit).toBeCloseTo(a, 6);
    expect(() => model.predict(new Int32Array(10) as unknown as Int16Array)).toThrow(
      /int16 or float/,
    );
  });

  it('resamples other rates', async () => {
    const model = await Model.create([HEY_LIVEKIT], { sampleRate: 48000 });
    const [, pcm] = readWav('positive.wav');
    // 16 kHz -> 48 kHz by linear interpolation; the score survives the round trip
    const up = new Float32Array(pcm.length * 3);
    for (let i = 0; i < up.length; i++) {
      const t = i / 3;
      const j = Math.floor(t);
      const f = t - j;
      up[i] = ((pcm[j] ?? 0) * (1 - f) + (pcm[j + 1] ?? 0) * f) / 32768;
    }
    expect((await model.predictClip(up)).hey_livekit).toBeGreaterThanOrEqual(0.5);
  });

  it('loads, names and unloads models', async () => {
    const model = await Model.create();
    expect(await model.predict(new Float32Array(1280))).toEqual({});
    await model.loadModel(new Uint8Array(readFileSync(TINY)), 'custom');
    await model.loadModel(TINY);
    expect(model.modelNames).toEqual(['custom', 'streaming_tiny']);
    expect(await model.unloadModel('custom')).toBe(true);
    expect(await model.unloadModel('custom')).toBe(false);
    expect(model.modelNames).toEqual(['streaming_tiny']);
    await expect(model.loadModel(new Uint8Array(4))).rejects.toThrow(/model name/);
  });

  it('explains files it cannot load', async () => {
    const dir = tmpdir();
    const bad = join(dir, 'lkww-bad.onnx');
    writeFileSync(bad, 'not a model');
    const model = await Model.create();
    await expect(model.loadModel('hey_livekit_micro.tflite')).rejects.toThrow(/micro model/);
    await expect(model.loadModel(bad)).rejects.toThrow(/could not load .*lkww-bad.onnx/);
    expect(model.modelNames).toEqual([]);
  });
});
