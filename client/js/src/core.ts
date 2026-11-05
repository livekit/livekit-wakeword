// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

export type { Detection } from './gate.js';
export { DetectionGate } from './gate.js';
/**
 * The runtime-agnostic core, for ONNX Runtime packages other than onnxruntime-web and -node
 * (such as onnxruntime-react-native):
 *
 *     import * as ort from 'onnxruntime-react-native';
 *     import { WakeWordModel } from '@livekit/wakeword/core';
 *     const model = new WakeWordModel(ort, { sampleRate: 16000 });
 *     await model.loadModel(bytes, 'hey_livekit');
 */
export type { Audio, ModelSource, OrtRuntime, WakeWordModelOptions } from './model.js';
export { WakeWordModel } from './model.js';
export { MODEL_RATE, Resampler } from './resample.js';
