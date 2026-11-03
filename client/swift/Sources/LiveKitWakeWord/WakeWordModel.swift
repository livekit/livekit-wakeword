// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// ONNX Runtime execution provider used by ``WakeWordModel``.
public enum ExecutionProvider: Sendable, Equatable {
    /// CoreML (Neural Engine, GPU and CPU).
    case coreML
    /// CoreML on CPU and GPU only.
    case coreMLCPUAndGPU
    /// CoreML on CPU only.
    case coreMLCPUOnly
    /// ONNX Runtime's CPU provider.
    case cpu
}

/// Streaming wake word detector: feed PCM as it arrives, get per-model scores back.
///
/// Each model is one `.onnx` file from `livekit-wakeword train`, keyed by its filename stem.
/// Calls on one instance must not overlap; ``WakeWordListener`` serialises them for you.
public final class WakeWordModel: @unchecked Sendable {
    /// The rate models run at; other input rates are resampled.
    public static let modelSampleRate: UInt32 = 16_000

    public let sampleRate: UInt32
    public let executionProvider: ExecutionProvider

    private let env: ORTEnv
    private let sessionOptions: ORTSessionOptions
    private let resampler: AudioResampler?
    private let lock = NSLock()
    private var models: [String: StreamingClassifier] = [:]

    /// Loads `models`; `sampleRate` is the rate of the PCM passed to ``predict(_:)``.
    public init(
        models modelURLs: [URL],
        sampleRate: UInt32,
        executionProvider: ExecutionProvider = .coreML
    ) throws {
        self.sampleRate = sampleRate
        self.executionProvider = executionProvider
        env = try ORTRuntime.sharedEnv()
        sessionOptions = try ORTRuntime.sessionOptions(for: executionProvider)
        resampler = sampleRate == Self.modelSampleRate ? nil : try AudioResampler(inputSampleRate: sampleRate)
        for url in modelURLs { try loadModel(url: url) }
    }

    /// Adds a model, keyed by `name` or the filename stem.
    public func loadModel(url: URL, name: String? = nil) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WakeWordError.classifierNotFound(url: url)
        }
        let key = name ?? url.deletingPathExtension().lastPathComponent
        let session = try ORTRuntime.wrap {
            try ORTSession(env: env, modelPath: url.path, sessionOptions: sessionOptions)
        }
        let model = try StreamingClassifier(name: key, url: url, session: session)
        lock.lock()
        models[key] = model
        lock.unlock()
    }

    /// Feeds the next audio (any length); returns each model's highest score over the 80 ms
    /// steps this call completed (0 if none).
    public func predict(_ pcm: UnsafeBufferPointer<Int16>) throws -> [String: Float] {
        try run(pcm, fresh: false)
    }

    public func predict(_ pcm: [Int16]) throws -> [String: Float] {
        try pcm.withUnsafeBufferPointer { try predict($0) }
    }

    /// Scores a whole clip from a fresh state: each model's highest score in it.
    public func predictClip(_ pcm: UnsafeBufferPointer<Int16>) throws -> [String: Float] {
        try run(pcm, fresh: true)
    }

    public func predictClip(_ pcm: [Int16]) throws -> [String: Float] {
        try pcm.withUnsafeBufferPointer { try predictClip($0) }
    }

    /// Forgets all audio heard so far.
    public func reset() {
        for model in snapshot().values { model.reset() }
    }

    /// Removes a model; returns `true` if it was loaded.
    @discardableResult
    public func unloadModel(name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return models.removeValue(forKey: name) != nil
    }

    /// Names of the loaded models.
    public var modelNames: [String] { Array(snapshot().keys) }

    private func snapshot() -> [String: StreamingClassifier] {
        lock.lock()
        defer { lock.unlock() }
        return models
    }

    private func run(_ pcm: UnsafeBufferPointer<Int16>, fresh: Bool) throws -> [String: Float] {
        let models = snapshot()
        if models.isEmpty { return [:] }
        let audio = try resampler?.resample(samples: pcm) ?? pcm.map { Float($0) / 32768 }
        var scores = [String: Float]()
        for (name, model) in models {
            if fresh { model.reset() }
            scores[name] = try model.feed(audio) ?? 0
        }
        return scores
    }
}
