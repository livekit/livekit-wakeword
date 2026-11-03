// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation

/// Errors raised by ``WakeWordModel`` and ``WakeWordListener``.
public enum WakeWordError: Error, LocalizedError, Sendable {
    /// No model file at the given URL.
    case classifierNotFound(url: URL)
    /// The file is not a streaming wake word model (v1 classifiers are not supported).
    case notStreamingModel(url: URL)
    /// The model did not return an expected output.
    case modelOutputMissing(key: String)
    /// The sample rate cannot be resampled to 16 kHz.
    case unsupportedSampleRate(rate: UInt32)
    /// Resampling failed.
    case resamplingFailed(underlying: Error?)
    /// ONNX Runtime failed.
    case runtimeFailure(underlying: Error)

    public var errorDescription: String? {
        switch self {
        case .classifierNotFound(let url):
            return "LiveKitWakeWord: model not found at \(url.path)."
        case .notStreamingModel(let url):
            return "LiveKitWakeWord: \(url.lastPathComponent) is not a streaming wake word model "
                + "(v1 classifier models are not supported; retrain with livekit-wakeword train)."
        case .modelOutputMissing(let key):
            return "LiveKitWakeWord: model output '\(key)' missing."
        case .unsupportedSampleRate(let rate):
            return "LiveKitWakeWord: sample rate \(rate) Hz cannot be resampled to 16 kHz."
        case .resamplingFailed(let underlying):
            return "LiveKitWakeWord: resampling failed\(underlying.map { " (\($0))" } ?? "")."
        case .runtimeFailure(let underlying):
            return "LiveKitWakeWord: ONNX Runtime error (\(underlying))."
        }
    }
}
