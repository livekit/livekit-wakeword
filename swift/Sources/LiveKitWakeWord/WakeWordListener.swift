// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

@preconcurrency import AVFoundation
import Foundation

/// A single wake-word trigger, emitted by ``WakeWordListener``.
public struct Detection: Sendable {
    /// The name the classifier was loaded under.
    public let name: String
    /// Confidence score in `[0, 1]`.
    public let confidence: Float
    /// `Date` at which the tap callback that produced this detection fired.
    public let timestamp: Date

    public init(name: String, confidence: Float, timestamp: Date) {
        self.name = name
        self.confidence = confidence
        self.timestamp = timestamp
    }
}

/// Drives the microphone and invokes ``WakeWordModel`` on rolling audio
/// windows, emitting ``Detection``s through an ``AsyncStream``.
///
/// This is the Swift equivalent of the Python `WakeWordListener`: it owns
/// an `AVAudioEngine`, captures mono PCM at the hardware sample rate, and
/// sends it to the underlying model with automatic resampling.
///
/// ```swift
/// let model = try WakeWordModel(classifiers: [heyLiveKitURL])
/// let listener = WakeWordListener(model: model, threshold: 0.5)
/// try await listener.start()
/// for await det in listener.detections() {
///     print("\(det.name): \(det.confidence)")
/// }
/// ```
public actor WakeWordListener {
    public let threshold: Float
    public let debounce: TimeInterval
    public let echoCancellation: Bool

    private let model: WakeWordModel
    private var engine: AVAudioEngine?
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?

    private var ringBuffer: [Int16] = []
    private var writeIndex: Int = 0
    private var samplesWritten: Int = 0

    private let windowSeconds: Double
    /// Minimum interval between two predict() calls. 20 ms ≈ 50 Hz updates.
    private let predictInterval: TimeInterval = 0.02
    private var lastPredictAt: CFAbsoluteTime = 0
    private var inflight: Bool = false

    private var lastDetectionAt: [String: Date] = [:]
    private var continuations: [UUID: AsyncStream<Detection>.Continuation] = [:]

    /// Create a listener around ``model``.
    ///
    /// - Parameters:
    ///   - threshold: Minimum confidence for a detection to be emitted.
    ///   - debounce: Minimum interval between consecutive detections of the
    ///     same wake word. Defaults to 2 s to avoid re-triggering on the
    ///     same utterance.
    ///   - windowSeconds: Length of the rolling audio window fed to the
    ///     model. 2 s matches the Rust crate's recommendation.
    ///   - echoCancellation: When `true`, the microphone is routed through the
    ///     platform's voice-processing I/O unit so audio the device is playing
    ///     out (e.g. an assistant's own TTS) is removed from the captured
    ///     signal. Defaults to `false` (raw capture).
    public init(
        model: WakeWordModel,
        threshold: Float = 0.5,
        debounce: TimeInterval = 2.0,
        windowSeconds: Double = 2.0,
        echoCancellation: Bool = false
    ) {
        self.model = model
        self.threshold = threshold
        self.debounce = debounce
        self.windowSeconds = windowSeconds
        self.echoCancellation = echoCancellation
    }

    /// Start capturing audio and running inference. Must be called after
    /// microphone permission has been granted.
    public func start() throws {
        if engine != nil { return }

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        let mode: AVAudioSession.Mode = echoCancellation ? .voiceChat : .measurement
        try session.setCategory(.playAndRecord, mode: mode, options: [.defaultToSpeaker])
        try session.setActive(true, options: [])
        #endif

        let engine = AVAudioEngine()
        let input = engine.inputNode

        if echoCancellation {
            do {
                try input.setVoiceProcessingEnabled(true)
            } catch {
                throw WakeWordError.echoCancellationUnavailable(underlying: error)
            }
        }

        let hwFormat = input.inputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0 else {
            throw WakeWordError.unsupportedSampleRate(rate: 0)
        }

        // Target the model's declared sample rate so the model receives
        // exactly what it expects regardless of what the mic hardware gives
        // us. AVAudioConverter does the resampling from hwFormat when they
        // differ; if hwRate == model.sampleRate it's a plain F32→Int16
        // conversion with no resample.
        let modelRate = Double(model.sampleRate)
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: modelRate,
            channels: 1,
            interleaved: true
        ) else {
            throw WakeWordError.unsupportedSampleRate(rate: model.sampleRate)
        }
        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else {
            throw WakeWordError.unsupportedSampleRate(rate: UInt32(hwFormat.sampleRate))
        }

        let ringSize = max(Int(modelRate * windowSeconds), 1)
        ringBuffer = [Int16](repeating: 0, count: ringSize)
        writeIndex = 0
        samplesWritten = 0
        lastPredictAt = 0
        inflight = false
        lastDetectionAt = [:]

        self.engine = engine
        self.converter = converter
        self.targetFormat = targetFormat

        // The tap runs on a real-time audio thread; we hop back into the
        // actor before touching any shared state.
        input.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let snapshot = Self.convert(
                buffer: buffer,
                converter: converter,
                targetFormat: targetFormat
            ) else { return }
            Task { [snapshot] in
                await self.ingest(samples: snapshot)
            }
        }

        engine.prepare()
        try engine.start()
    }

    /// Stop capturing audio. Idempotent.
    public func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        converter = nil
        targetFormat = nil

        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        #endif

        ringBuffer = []
        writeIndex = 0
        samplesWritten = 0
        inflight = false

        // Close every outstanding stream so callers' for-await loops exit.
        for (_, c) in continuations { c.finish() }
        continuations.removeAll()
    }

    /// A fresh async stream of detections. Multiple consumers each get their
    /// own stream; events are broadcast to all.
    public nonisolated func detections() -> AsyncStream<Detection> {
        AsyncStream<Detection> { continuation in
            let id = UUID()
            Task { await self.addContinuation(id: id, continuation: continuation) }
            continuation.onTermination = { [weak self] _ in
                Task { await self?.removeContinuation(id: id) }
            }
        }
    }

    // MARK: - Actor-isolated helpers

    private func addContinuation(id: UUID, continuation: AsyncStream<Detection>.Continuation) {
        continuations[id] = continuation
    }

    private func removeContinuation(id: UUID) {
        continuations.removeValue(forKey: id)
    }

    private func ingest(samples: [Int16]) async {
        let size = ringBuffer.count
        guard size > 0, !samples.isEmpty else { return }

        var idx = writeIndex
        for s in samples {
            ringBuffer[idx] = s
            idx += 1
            if idx >= size { idx = 0 }
        }
        writeIndex = idx
        samplesWritten = min(samplesWritten + samples.count, size)
        guard samplesWritten >= size else { return }

        let now = CFAbsoluteTimeGetCurrent()
        guard !inflight, (now - lastPredictAt) >= predictInterval else { return }
        lastPredictAt = now
        inflight = true

        // Linearize ring into chronological order before running predict.
        var snapshot = [Int16](repeating: 0, count: size)
        let tail = size - writeIndex
        snapshot.withUnsafeMutableBufferPointer { dst in
            ringBuffer.withUnsafeBufferPointer { src in
                guard let s = src.baseAddress, let d = dst.baseAddress else { return }
                d.update(from: s + writeIndex, count: tail)
                if writeIndex > 0 {
                    (d + tail).update(from: s, count: writeIndex)
                }
            }
        }

        // Resample to the model's expected rate using the stateless model.
        // ``WakeWordModel`` was built with ``sampleRate == hardwareSampleRate``
        // by the caller; if not, it will perform the resample internally.
        do {
            let scores = try snapshot.withUnsafeBufferPointer { try self.model.predict($0) }
            emit(scores: scores, timestamp: Date())
        } catch {
            // Silently swallow per-chunk errors — a listener should keep
            // running through transient issues. Surface via a logger if
            // this becomes a pattern.
        }
        inflight = false
    }

    private func emit(scores: [String: Float], timestamp: Date) {
        for (name, confidence) in scores where confidence >= threshold {
            if let last = lastDetectionAt[name],
               timestamp.timeIntervalSince(last) < debounce {
                continue
            }
            lastDetectionAt[name] = timestamp
            let detection = Detection(name: name, confidence: confidence, timestamp: timestamp)
            for (_, c) in continuations { c.yield(detection) }
        }
    }

    // MARK: - Audio conversion helper (non-isolated, runs on audio thread)

    private static func convert(
        buffer inputBuffer: AVAudioPCMBuffer,
        converter: AVAudioConverter,
        targetFormat: AVAudioFormat
    ) -> [Int16]? {
        // Size the output for the resampled rate. When targetFormat's
        // sample rate matches the input, the ratio is 1. When resampling
        // down (e.g. 48k→16k) we need 1/3 of the input frames; up we need
        // more. +8 frames of slack accounts for AVAudioConverter's
        // internal alignment.
        let ratio = targetFormat.sampleRate / inputBuffer.format.sampleRate
        let outCapacity = AVAudioFrameCount(ceil(Double(inputBuffer.frameLength) * ratio)) + 8
        guard let outBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: outCapacity
        ) else { return nil }
        var consumed = false
        var error: NSError?
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }
        guard status != .error, error == nil,
              let channel = outBuffer.int16ChannelData else { return nil }

        let count = Int(outBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: channel[0], count: count))
    }
}
