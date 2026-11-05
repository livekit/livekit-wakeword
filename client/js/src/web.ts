// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import * as ort from 'onnxruntime-web/wasm';
import { WakeWordModel as Core, type ModelSource, type WakeWordModelOptions } from './model.js';

export type { Detection } from './gate.js';
export { DetectionGate } from './gate.js';
export type { StartOptions, WakeWordListenerOptions } from './listener.js';
export { WakeWordListener } from './listener.js';
export type { Audio, ModelSource, WakeWordModelOptions } from './model.js';
export { MODEL_RATE, Resampler } from './resample.js';

/**
 * onnxruntime-web's settings, such as `env.wasm.wasmPaths` (where its `.wasm` files are served
 * from) and `env.wasm.numThreads`.
 */
export const env: typeof ort.env = ort.env;

/**
 * Wake word models on onnxruntime-web (WebAssembly):
 *
 *     const model = await WakeWordModel.create(['/models/hey_livekit.onnx']);
 *     const scores = await model.predict(frame); // { hey_livekit: 0.97 }
 */
export class WakeWordModel extends Core {
  constructor(options: WakeWordModelOptions = {}) {
    super(ort, {
      ...options,
      sessionOptions: { executionProviders: ['wasm'], ...options.sessionOptions },
    });
  }

  /** A model with `models` (URLs or bytes) loaded. */
  static async create(
    models: ModelSource[] = [],
    options: WakeWordModelOptions = {},
  ): Promise<WakeWordModel> {
    const model = new WakeWordModel(options);
    for (const source of models) await model.loadModel(source);
    return model;
  }
}
