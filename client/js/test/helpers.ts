// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../../../', import.meta.url));

/** The shipped hey livekit model. */
export const HEY_LIVEKIT = `${root}models/hey_livekit.onnx`;
/** The tiny streaming model shared with the Python tests. */
export const TINY = `${root}client/python/tests/fixtures/streaming_tiny.onnx`;

/** A fixture shared with the Python and Swift tests: client/python/tests/fixtures/<name>. */
export function fixture(name: string): string {
  return `${root}client/python/tests/fixtures/${name}`;
}

/** Reference outputs from the Python client (scripts/make_parity.py). */
export const parity: {
  scores: Record<'positive' | 'negative', number[]>;
  resample: Record<string, number[]>;
} = JSON.parse(readFileSync(new URL('./fixtures/parity.json', import.meta.url), 'utf8'));

/** A 16-bit mono PCM WAV as [sample rate, samples]. */
export function readWav(name: string): [number, Int16Array] {
  const buf = readFileSync(fixture(name));
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let rate = 0;
  for (let i = 12; i + 8 <= buf.length; ) {
    const id = buf.toString('ascii', i, i + 4);
    const size = view.getUint32(i + 4, true);
    if (id === 'fmt ') rate = view.getUint32(i + 12, true);
    if (id === 'data') {
      const n = Math.floor(Math.min(size, buf.length - i - 8) / 2);
      const samples = new Int16Array(n);
      for (let j = 0; j < n; j++) samples[j] = view.getInt16(i + 8 + 2 * j, true);
      return [rate, samples];
    }
    i += 8 + size + (size % 2);
  }
  throw new Error(`${name}: no data chunk`);
}

/** Seeded uniform noise in [-amp, amp]. */
export function noise(n: number, amp = 0.05, seed = 1): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = (s / 2 ** 32 - 0.5) * 2 * amp;
  }
  return out;
}
