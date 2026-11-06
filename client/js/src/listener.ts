// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import { type Detection, DetectionGate } from './gate.js';
import type { WakeWordModel } from './model.js';
import { MODEL_RATE, Resampler } from './resample.js';

const FRAME_SECONDS = 0.08; // one model step per message from the audio thread
const PROCESSOR = 'livekit-wakeword-tap';
const registered = new WeakSet<BaseAudioContext>(); // a processor name registers once per context

// Runs on the audio thread: collects 80 ms of mono input and posts it to the listener.
const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = options.processorOptions.size;
    this.buf = new Float32Array(this.size);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.size) {
          this.port.postMessage(this.buf, [this.buf.buffer]);
          this.buf = new Float32Array(this.size);
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('${PROCESSOR}', Tap);
`;

export interface WakeWordListenerOptions {
  /** Score at or above which a model fires (default 0.5). */
  threshold?: number;
  /** Minimum seconds between two detections of the same wake word (default 2). */
  debounce?: number;
  /** Called with every model's score for each 80 ms step heard (for meters and debugging). */
  onScores?: (scores: Record<string, number>) => void;
}

export interface StartOptions {
  /**
   * Audio to listen to: a `MediaStream` or `MediaStreamTrack` (such as a livekit-client
   * `LocalAudioTrack.mediaStreamTrack`). By default the listener opens the microphone, and closes
   * it on `stop()`.
   */
  source?: MediaStream | MediaStreamTrack;
  /** Microphone constraints when the listener opens it. */
  constraints?: MediaTrackConstraints;
  /** An AudioContext to run in; by default the listener makes one and closes it on `stop()`. */
  audioContext?: AudioContext;
}

/**
 * Streams the microphone (or any audio track) through a `WakeWordModel` in 80 ms steps.
 *
 * After a detection the listener pauses (and the model is reset) until `waitForDetection()` is
 * called again, which each turn of the `detections()` loop does:
 *
 *     const listener = new WakeWordListener(model, { threshold: 0.5, debounce: 2 });
 *     await listener.start(); // call from a user gesture, so the AudioContext may start
 *     for await (const detection of listener.detections()) {
 *       await talkToAgent(); // no detection fires while this runs
 *     }
 *     // the loop ends on stop(), or throws if the model failed
 */
export class WakeWordListener {
  private readonly gate: DetectionGate;
  private readonly onScores?: (scores: Record<string, number>) => void;
  private context: AudioContext | null = null;
  private ownsContext = false;
  private stream: MediaStream | null = null;
  private ownsStream = false;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private tap: AudioWorkletNode | null = null;
  private resampler: Resampler | null = null;
  private listening = false; // false while a detection is being handled
  private running = false;
  private error: unknown = null;
  private queued: Detection[] = [];
  private waiters: { resolve: (d: Detection) => void; reject: (e: unknown) => void }[] = [];

  constructor(
    readonly model: WakeWordModel,
    options: WakeWordListenerOptions = {},
  ) {
    this.gate = new DetectionGate(options.threshold ?? 0.5, options.debounce ?? 2.0);
    this.onScores = options.onScores;
  }

  get isRunning(): boolean {
    return this.running;
  }

  /** Starts listening. Call it from a user gesture (a click), or the AudioContext stays suspended. */
  async start(options: StartOptions = {}): Promise<void> {
    if (this.running) return;
    try {
      this.ownsContext = !options.audioContext;
      const context = options.audioContext ?? new AudioContext();
      this.context = context;
      if (context.state === 'suspended') await context.resume();

      const { source } = options;
      this.ownsStream = !source;
      this.stream =
        source instanceof MediaStream
          ? source
          : source
            ? new MediaStream([source])
            : await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, ...options.constraints },
              });

      if (!registered.has(context)) {
        const url = URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' }));
        try {
          await context.audioWorklet.addModule(url);
        } finally {
          URL.revokeObjectURL(url);
        }
        registered.add(context);
      }
      const rate = context.sampleRate;
      this.resampler = rate === MODEL_RATE ? null : new Resampler(rate);
      this.sourceNode = context.createMediaStreamSource(this.stream);
      this.tap = new AudioWorkletNode(context, PROCESSOR, {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 1,
        channelCountMode: 'explicit', // stereo inputs are mixed down to mono
        processorOptions: { size: Math.round(rate * FRAME_SECONDS) },
      });
      this.tap.port.onmessage = (e: MessageEvent<Float32Array>) => this.onAudio(e.data);
      this.sourceNode.connect(this.tap);
    } catch (e) {
      await this.teardown();
      throw e;
    }
    this.gate.reset();
    this.queued = [];
    this.error = null;
    await this.model.reset();
    this.listening = true;
    this.running = true;
  }

  /** Stops listening; pending `waitForDetection()` calls reject and `detections()` loops end. */
  async stop(): Promise<void> {
    if (!this.running && !this.context) return;
    this.running = false;
    this.listening = false;
    await this.teardown();
    this.flush(new Error('WakeWordListener stopped'));
  }

  /** The next detection; resumes listening if a detection paused it. */
  waitForDetection(): Promise<Detection> {
    const next = this.queued.shift();
    if (next) return Promise.resolve(next);
    if (this.error) return Promise.reject(this.error);
    if (!this.running) return Promise.reject(new Error('WakeWordListener is not running'));
    this.listening = true;
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }

  /** Detections until `stop()`; each turn resumes listening (see `waitForDetection()`). */
  async *detections(): AsyncGenerator<Detection, void, undefined> {
    while (this.running || this.queued.length) {
      try {
        yield await this.waitForDetection();
      } catch {
        if (this.error) throw this.error;
        return;
      }
    }
    if (this.error) throw this.error;
  }

  private onAudio(frame: Float32Array): void {
    if (!this.running || !this.listening) return; // paused: audio heard meanwhile is dropped
    const audio = this.resampler ? this.resampler.process(frame) : frame;
    this.model.predict16k(audio).then(
      (scores) => {
        if (!this.running || !this.listening) return;
        this.onScores?.(scores);
        const detection = this.gate.check(scores, performance.now());
        if (!detection) return;
        this.listening = false;
        this.resampler?.reset();
        void this.model.reset();
        const waiter = this.waiters.shift();
        if (waiter) waiter.resolve(detection);
        else this.queued.push(detection);
      },
      (e: unknown) => {
        if (!this.running) return;
        this.error = e;
        this.running = false;
        void this.teardown();
        this.flush(e);
      },
    );
  }

  private flush(error: unknown): void {
    for (const w of this.waiters.splice(0)) w.reject(error);
  }

  private async teardown(): Promise<void> {
    if (this.tap) this.tap.port.onmessage = null;
    this.sourceNode?.disconnect();
    this.tap?.disconnect();
    this.sourceNode = null;
    this.tap = null;
    if (this.ownsStream) for (const t of this.stream?.getTracks() ?? []) t.stop();
    this.stream = null;
    const context = this.context;
    this.context = null;
    if (this.ownsContext && context && context.state !== 'closed') await context.close();
  }
}
