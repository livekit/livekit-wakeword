// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation

/// Errors raised by ``WakeWordModel`` / ``WakeWordListener``.
public enum WakeWordError: Error, LocalizedError, Sendable {
    /// A bundled ONNX resource (mel/embedding model) was not found inside
    /// the ``LiveKitWakeWord`` resource bundle.
    case bundledResourceMissing(name: String)
    /// The classifier file at the given URL does not exist or is not a
    /// readable ONNX model.
    case classifierNotFound(url: URL)
    /// An ONNX model's output dictionary was missing an expected key.
    case modelOutputMissing(key: String)
    /// An ONNX model returned a tensor with an unexpected shape.
    case unexpectedOutputShape(expected: String, actual: [Int])
    /// ``predict`` was called with a PCM buffer that resampled to an audio
    /// length outside the mel model's usable range (currently 1–3 s at
    /// 16 kHz).
    case audioOutOfRange(samples: Int)
    /// The requested sample rate cannot be resampled to 16 kHz by
    /// ``AVAudioConverter``.
    case unsupportedSampleRate(rate: UInt32)
    /// ``AVAudioConverter`` reported an error while resampling.
    case resamplingFailed(underlying: Error?)
    /// The ONNX Runtime raised an error during session creation or
    /// inference.
    case runtimeFailure(underlying: Error)
    /// Acoustic echo cancellation was requested (``WakeWordListener`` created
    /// with `echoCancellation: true`) but the platform's voice-processing I/O
    /// unit could not be enabled.
    case echoCancellationUnavailable(underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .bundledResourceMissing(let name):
            return "LiveKitWakeWord: bundled resource '\(name)' is missing."
        case .classifierNotFound(let url):
            return "LiveKitWakeWord: classifier not found at \(url.path)."
        case .modelOutputMissing(let key):
            return "LiveKitWakeWord: ONNX model output missing '\(key)'."
        case .unexpectedOutputShape(let expected, let actual):
            return "LiveKitWakeWord: unexpected output shape (expected \(expected), got \(actual))."
        case .audioOutOfRange(let samples):
            return "LiveKitWakeWord: audio chunk must resample to 1–3 s at 16 kHz (got \(samples) samples)."
        case .unsupportedSampleRate(let rate):
            return "LiveKitWakeWord: sample rate \(rate) Hz cannot be resampled to 16 kHz."
        case .resamplingFailed(let underlying):
            if let underlying {
                return "LiveKitWakeWord: resampling failed (\(underlying))."
            }
            return "LiveKitWakeWord: resampling failed."
        case .runtimeFailure(let underlying):
            return "LiveKitWakeWord: ONNX Runtime error (\(underlying))."
        case .echoCancellationUnavailable(let underlying):
            return "LiveKitWakeWord: could not enable echo cancellation (\(underlying))."
        }
    }
}
