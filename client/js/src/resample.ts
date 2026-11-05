// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

/** The rate wake word models run at. */
export const MODEL_RATE = 16000;

const ATTENUATION_DB = 90;
const TRANSITION = 0.1; // transition band, as a fraction of the lower Nyquist rate

/**
 * Streaming resampler from `rate` to 16 kHz: a polyphase windowed-sinc FIR (Kaiser window, about
 * 90 dB stopband), the same filter as the Python and Swift clients, with state carried between
 * calls.
 *
 * Output sample m is at input time m * down / up; it is computed as soon as the input reaching
 * that time has arrived, from the last `taps` input samples (zeros before the stream starts).
 */
export class Resampler {
  readonly rate: number;
  readonly up: number;
  readonly down: number;
  readonly taps: number;
  private readonly bank: Float32Array[]; // bank[p][k]: weight of input i - k at upsampled phase p
  private history!: Float32Array; // the last taps - 1 inputs, oldest first
  private seen = 0; // input samples consumed
  private made = 0; // output samples produced

  constructor(rate: number) {
    if (!Number.isInteger(rate) || rate <= 0) {
      throw new RangeError(`sample rate must be a positive integer, got ${rate}`);
    }
    const g = gcd(rate, MODEL_RATE);
    this.rate = rate;
    this.up = MODEL_RATE / g;
    this.down = rate / g;
    // low-pass in the upsampled domain: cut below the lower of the two Nyquist rates
    const nyq = 0.5 / Math.max(this.up, this.down);
    const width = TRANSITION * nyq;
    const beta = 0.1102 * (ATTENUATION_DB - 8.7);
    let n = Math.ceil((ATTENUATION_DB - 7.95) / (2.285 * 2 * Math.PI * width)) | 1;
    const taps = Math.ceil(n / this.up);
    n = taps * this.up;
    const cutoff = nyq - width / 2;
    const h = new Float64Array(n);
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const t = j - (n - 1) / 2;
      const w =
        n > 1 ? besselI0(beta * Math.sqrt(1 - ((2 * j) / (n - 1) - 1) ** 2)) / besselI0(beta) : 1;
      h[j] = 2 * cutoff * sinc(2 * cutoff * t) * w;
      sum += h[j];
    }
    const gain = this.up / sum; // unity gain after zero-stuffing
    this.bank = [];
    for (let p = 0; p < this.up; p++) {
      const row = new Float32Array(taps);
      for (let k = 0; k < taps; k++) row[k] = h[k * this.up + p] * gain;
      this.bank.push(row);
    }
    this.taps = taps;
    this.reset();
  }

  reset(): void {
    this.history = new Float32Array(this.taps - 1);
    this.seen = 0;
    this.made = 0;
  }

  /** Resamples the next mono float32 audio; returns the 16 kHz samples it completes. */
  process(audio: Float32Array): Float32Array {
    const hist = this.taps - 1;
    const x = new Float32Array(hist + audio.length);
    x.set(this.history);
    x.set(audio, hist);
    const base = this.seen - hist; // absolute index of x[0]
    this.seen += audio.length;
    // outputs whose newest input index i = floor(m * down / up) has arrived
    const last = Math.floor((this.seen * this.up - 1) / this.down);
    const out = new Float32Array(Math.max(0, last + 1 - this.made));
    for (let o = 0, m = this.made; m <= last; m++, o++) {
      const i = Math.floor((m * this.down) / this.up);
      const w = this.bank[(m * this.down) % this.up];
      const at = i - base;
      let acc = 0;
      for (let k = 0; k < this.taps; k++) acc += x[at - k] * w[k];
      out[o] = acc;
    }
    this.made = last + 1;
    this.history = x.slice(x.length - hist);
    return out;
  }
}

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

function sinc(x: number): number {
  if (x === 0) return 1;
  const y = Math.PI * x;
  return Math.sin(y) / y;
}

/** Modified Bessel function of the first kind, order 0 (power series). */
function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  const q = (x * x) / 4;
  for (let k = 1; k < 500; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return sum;
}
