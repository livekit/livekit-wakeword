// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import type { InferenceSession, Tensor } from 'onnxruntime-common';
import { MODEL_RATE, Resampler } from './resample.js';

export const FRAME_SAMPLES = 1280; // 80 ms at 16 kHz
const WARMUP_FRAMES = 25; // 2 s of silence: the state every stream starts from

/** The parts of an ONNX Runtime JS package (`onnxruntime-web`, `-node`, `-react-native`) used here. */
export interface OrtRuntime {
  InferenceSession: {
    create(
      source: string | Uint8Array,
      options?: InferenceSession.SessionOptions,
    ): Promise<InferenceSession>;
  };
  Tensor: typeof Tensor;
}

/**
 * A model file: a path (Node) or URL (browser), or the bytes of an `.onnx` file (then pass a name
 * to `loadModel`).
 */
export type ModelSource = string | URL | Uint8Array | ArrayBuffer;

/** Mono PCM: int16, or float in [-1, 1]. */
export type Audio = Float32Array | Int16Array;

export interface WakeWordModelOptions {
  /** Rate of the audio passed to `predict()`; other rates than 16 kHz are resampled internally. */
  sampleRate?: number;
  /** ONNX Runtime session options for every model loaded. */
  sessionOptions?: InferenceSession.SessionOptions;
}

/** One model's state: inputs `chunk` (1, 1280) + state tensors, outputs `score` + `<state>_out`. */
class Stream {
  private state: Record<string, Tensor> = {};
  private warm: Record<string, Tensor> = {};
  private pending = new Float32Array(FRAME_SAMPLES);
  private filled = 0;

  private constructor(
    readonly session: InferenceSession,
    private readonly ort: OrtRuntime,
    private readonly stateNames: string[],
  ) {}

  static async open(ort: OrtRuntime, session: InferenceSession): Promise<Stream> {
    const state: Record<string, Tensor> = {};
    for (const input of session.inputMetadata) {
      if (input.name === 'chunk') continue;
      const shape = input.isTensor ? input.shape : [];
      if (!input.isTensor || !shape.every((d) => typeof d === 'number' && d > 0)) {
        throw new Error(`state input '${input.name}' needs a fixed shape`);
      }
      const dims = shape as number[];
      const size = dims.reduce((a, b) => a * b, 1);
      state[input.name] =
        input.type === 'int64'
          ? new ort.Tensor('int64', new BigInt64Array(size), dims)
          : new ort.Tensor('float32', new Float32Array(size), dims);
    }
    const names = session.outputNames.filter((n) => n !== 'score');
    const s = new Stream(session, ort, names);
    s.state = state;
    const silence = new Float32Array(FRAME_SAMPLES);
    for (let i = 0; i < WARMUP_FRAMES; i++) await s.step(silence);
    s.warm = s.state;
    return s;
  }

  private async step(chunk: Float32Array): Promise<number> {
    const out = await this.session.run({
      chunk: new this.ort.Tensor('float32', chunk, [1, FRAME_SAMPLES]),
      ...this.state,
    });
    const next: Record<string, Tensor> = {};
    for (const name of this.stateNames) next[name.replace(/_out$/, '')] = out[name];
    this.state = next;
    return Number((out.score.data as Float32Array)[0]);
  }

  reset(): void {
    this.state = this.warm;
    this.filled = 0;
  }

  /** The highest score among the 80 ms steps `audio` completes (0 if none). */
  async feed(audio: Float32Array): Promise<number> {
    let best = 0;
    let pos = 0;
    while (pos < audio.length) {
      const n = Math.min(FRAME_SAMPLES - this.filled, audio.length - pos);
      this.pending.set(audio.subarray(pos, pos + n), this.filled);
      this.filled += n;
      pos += n;
      if (this.filled === FRAME_SAMPLES) {
        const chunk = this.pending;
        this.pending = new Float32Array(FRAME_SAMPLES);
        this.filled = 0;
        best = Math.max(best, await this.step(chunk));
      }
    }
    return best;
  }
}

/**
 * Wake word models (`.onnx` from `livekit-wakeword train`), streamed over mono audio. This is the
 * runtime-agnostic core: `@livekit/wakeword` binds it to onnxruntime-web in browsers and
 * onnxruntime-node in Node; `@livekit/wakeword/core` takes any ONNX Runtime JS package.
 *
 * Calls run in order: `predict()`, `reset()`, `loadModel()` and `unloadModel()` are queued, so
 * overlapping calls never interleave.
 */
export class WakeWordModel {
  readonly sampleRate: number;
  private readonly resampler: Resampler | null;
  private readonly models = new Map<string, Stream>();
  private queue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly ort: OrtRuntime,
    private readonly options: WakeWordModelOptions = {},
  ) {
    this.sampleRate = options.sampleRate ?? MODEL_RATE;
    this.resampler = this.sampleRate === MODEL_RATE ? null : new Resampler(this.sampleRate);
  }

  /** Runs `fn` after every call queued before it. */
  private enqueue<T>(fn: () => Promise<T> | T): Promise<T> {
    const run = this.queue.then(fn);
    this.queue = run.catch(() => {});
    return run;
  }

  /** Adds a model, named after the file unless `name` is given (required for bytes). */
  loadModel(source: ModelSource, name?: string): Promise<void> {
    return this.enqueue(async () => {
      const modelName = name ?? nameOf(source);
      const src = source instanceof URL ? source.href : source;
      const label = typeof src === 'string' ? src : `'${modelName}'`;
      if (typeof src === 'string' && /\.tflite(\?|#|$)/i.test(src)) {
        throw new Error(
          `${src} is a micro model (int8 TFLite for ESP32 / TensorFlow Lite Micro); load the standard .onnx model here`,
        );
      }
      let session: InferenceSession;
      try {
        session = await this.ort.InferenceSession.create(
          src instanceof ArrayBuffer ? new Uint8Array(src) : src,
          this.options.sessionOptions,
        );
      } catch (e) {
        throw new Error(
          `could not load ${label} (is it an .onnx model?): ${(e as Error).message ?? e}`,
          {
            cause: e,
          },
        );
      }
      if (!session.inputNames.includes('chunk')) {
        await session.release();
        throw new Error(
          `${label} is not a streaming wake word model; v1 classifier models are no longer supported, retrain with \`livekit-wakeword train\``,
        );
      }
      const stream = await Stream.open(this.ort, session);
      await this.models.get(modelName)?.session.release();
      this.models.set(modelName, stream);
    });
  }

  /** Removes a model; resolves to whether it was loaded. */
  unloadModel(name: string): Promise<boolean> {
    return this.enqueue(async () => {
      const stream = this.models.get(name);
      if (!stream) return false;
      this.models.delete(name);
      await stream.session.release();
      return true;
    });
  }

  get modelNames(): string[] {
    return [...this.models.keys()];
  }

  /** Forgets all audio heard so far (e.g. after a detection). */
  reset(): Promise<void> {
    return this.enqueue(() => {
      this.resampler?.reset();
      for (const m of this.models.values()) m.reset();
    });
  }

  /**
   * Feeds the next audio (mono at `sampleRate`, any length). Resolves to each model's highest
   * score among the 80 ms steps this call completed (0 if none).
   */
  predict(audio: Audio): Promise<Record<string, number>> {
    const x = asFloat(audio);
    return this.enqueue(() => this.feed(this.resampler ? this.resampler.process(x) : x));
  }

  /** Each model's highest score over a whole clip, starting fresh. */
  predictClip(audio: Audio): Promise<Record<string, number>> {
    const x = asFloat(audio);
    return this.enqueue(() => {
      this.resampler?.reset();
      for (const m of this.models.values()) m.reset();
      return this.feed(this.resampler ? this.resampler.process(x) : x);
    });
  }

  /**
   * Feeds 16 kHz audio, bypassing `sampleRate` (for listeners that resample themselves).
   * @internal
   */
  predict16k(audio: Float32Array): Promise<Record<string, number>> {
    return this.enqueue(() => this.feed(audio));
  }

  private async feed(audio: Float32Array): Promise<Record<string, number>> {
    const scores: Record<string, number> = {};
    for (const [name, m] of this.models) scores[name] = await m.feed(audio);
    return scores;
  }

  /** Releases every model's ONNX Runtime session. */
  release(): Promise<void> {
    return this.enqueue(async () => {
      for (const m of this.models.values()) await m.session.release();
      this.models.clear();
    });
  }
}

/** Mono int16 or float PCM -> float32 in [-1, 1]. */
function asFloat(audio: Audio): Float32Array {
  if (audio instanceof Float32Array) return audio;
  if (audio instanceof Int16Array) {
    const out = new Float32Array(audio.length);
    for (let i = 0; i < audio.length; i++) out[i] = audio[i] / 32768;
    return out;
  }
  throw new TypeError('expected int16 or float32 PCM (Int16Array or Float32Array)');
}

/** `hey_livekit` from `models/hey_livekit.onnx` or `https://x/hey_livekit.onnx?v=2`. */
function nameOf(source: ModelSource): string {
  if (typeof source !== 'string' && !(source instanceof URL)) {
    throw new TypeError('pass a model name when loading a model from bytes');
  }
  const path = source instanceof URL ? source.pathname : source.split(/[?#]/)[0];
  const file = path.split(/[/\\]/).pop() ?? '';
  return decodeURIComponent(file).replace(/\.onnx$/i, '') || 'model';
}
