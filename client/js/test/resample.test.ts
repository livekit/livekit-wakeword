// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';
import { Resampler } from '../src/resample.js';
import { noise, parity } from './helpers.js';

/** The same signal as `chirp()` in scripts/make_parity.py. */
function chirp(rate: number, n: number): Float32Array {
  return Float32Array.from({ length: n }, (_, i) => {
    const t = i / rate;
    return 0.3 * Math.sin(2 * Math.PI * (200 + 1500 * t) * t);
  });
}

function tone(rate: number, hz: number, seconds = 2): Float32Array {
  return Float32Array.from(
    { length: rate * seconds },
    (_, i) => 0.5 * Math.sin((2 * Math.PI * hz * i) / rate),
  );
}

describe('Resampler', () => {
  it.each([44100, 48000])('matches the Python client at %i Hz', (rate) => {
    const y = new Resampler(rate).process(chirp(rate, rate / 4));
    const expected = parity.resample[String(rate)];
    expect(y.length).toBe(expected.length);
    let worst = 0;
    for (let i = 0; i < y.length; i++) worst = Math.max(worst, Math.abs(y[i] - expected[i]));
    expect(worst).toBeLessThan(1e-5);
  });

  it.each([8000, 22050, 24000, 44100, 48000, 96000])(
    'passes tones unchanged from %i Hz',
    (rate) => {
      for (const hz of [300, 1000, 3500]) {
        const y = new Resampler(rate).process(tone(rate, hz));
        expect(Math.abs(y.length - 32000)).toBeLessThanOrEqual(1);
        // least-squares fit of a sine + cosine at hz over the steady state
        let ss = 0,
          sc = 0,
          cc = 0,
          ys = 0,
          yc = 0;
        for (let i = 4000; i < 28000; i++) {
          const s = Math.sin((2 * Math.PI * hz * i) / 16000);
          const c = Math.cos((2 * Math.PI * hz * i) / 16000);
          ss += s * s;
          sc += s * c;
          cc += c * c;
          ys += y[i] * s;
          yc += y[i] * c;
        }
        const det = ss * cc - sc * sc;
        const a = (ys * cc - yc * sc) / det;
        const b = (yc * ss - ys * sc) / det;
        expect(Math.hypot(a, b)).toBeCloseTo(0.5, 3);
        let res = 0;
        for (let i = 4000; i < 28000; i++) {
          const fit =
            a * Math.sin((2 * Math.PI * hz * i) / 16000) +
            b * Math.cos((2 * Math.PI * hz * i) / 16000);
          res += (y[i] - fit) ** 2;
        }
        expect(Math.sqrt(res / 24000)).toBeLessThan(1e-4); // better than 70 dB
      }
    },
  );

  it.each([44100, 48000])('removes frequencies above 8 kHz from %i Hz', (rate) => {
    const y = new Resampler(rate).process(tone(rate, 9000));
    let peak = 0;
    for (let i = 4000; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]));
    expect(peak).toBeLessThan(0.5 * 10 ** (-80 / 20));
  });

  it('gives the same output whatever the chunk sizes', () => {
    const x = noise(44100, 1);
    const whole = new Resampler(44100).process(x);
    const r = new Resampler(44100);
    const parts: number[] = [];
    let pos = 0;
    for (let k = 0; pos < x.length; k++) {
      const n = [1, 7, 441, 1000, 3333][k % 5];
      parts.push(...r.process(x.subarray(pos, pos + n)));
      pos += n;
    }
    expect(parts.length).toBe(whole.length);
    for (let i = 0; i < whole.length; i++) expect(Math.abs(parts[i] - whole[i])).toBeLessThan(1e-6);
    r.reset();
    expect(r.process(x)).toEqual(whole);
  });

  it('rejects bad rates', () => {
    expect(() => new Resampler(0)).toThrow(RangeError);
    expect(() => new Resampler(44100.5)).toThrow(RangeError);
  });
});
