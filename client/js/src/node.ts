// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

import type { InferenceSession } from 'onnxruntime-common';
import * as ort from 'onnxruntime-node';
import { WakeWordModel as Core, type ModelSource, type WakeWordModelOptions } from './model.js';

export type { Detection } from './gate.js';
export { DetectionGate } from './gate.js';
export type { Audio, ModelSource, WakeWordModelOptions } from './model.js';
export { MODEL_RATE, Resampler } from './resample.js';

/**
 * Wake word models on onnxruntime-node:
 *
 *     const model = await WakeWordModel.create(['hey_livekit.onnx'], { sampleRate: 48000 });
 *     const scores = await model.predict(frame); // { hey_livekit: 0.97 }; Int16Array or Float32Array
 */
export class WakeWordModel extends Core {
  constructor(options: WakeWordModelOptions = {}) {
    super(ort, { ...options, sessionOptions: cpuOptions(options.sessionOptions) });
  }

  /** A model with `models` (paths or bytes) loaded. */
  static async create(
    models: ModelSource[] = [],
    options: WakeWordModelOptions = {},
  ): Promise<WakeWordModel> {
    const model = new WakeWordModel(options);
    for (const source of models) await model.loadModel(source);
    return model;
  }
}

/**
 * CPU, with KleidiAI off unless the caller says otherwise: onnxruntime 1.27 on arm64 ran batch-1
 * convolutions through KleidiAI kernels that returned wrong results (as in the Python client).
 */
function cpuOptions(
  options: InferenceSession.SessionOptions = {},
): InferenceSession.SessionOptions {
  const extra = (options.extra ?? {}) as Record<string, unknown>;
  const mlas = (extra.mlas ?? {}) as Record<string, unknown>;
  return {
    executionProviders: ['cpu'],
    ...options,
    extra: { ...extra, mlas: { disable_kleidiai: '1', ...mlas } },
  };
}
