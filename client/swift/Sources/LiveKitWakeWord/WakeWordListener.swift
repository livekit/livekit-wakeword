// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

@preconcurrency import AVFoundation
import Foundation

/// A wake word detection emitted by ``WakeWordListener``.
public struct Detection: Sendable {
    /// The model name.
    public let name: String
    /// Score in `[0, 1]`.
    public let confidence: Float
    public let timestamp: Date

    public init(name: String, confidence: Float, timestamp: Date) {
        self.name = name
        self.confidence = confidence
        self.timestamp = timestamp
    }
}

/// Streams the microphone through a ``WakeWordModel`` and emits ``Detection``s.
///
/// ```swift
/// let model = try WakeWordModel(models: [heyLiveKitURL], sampleRate: 16_000)
/// let listener = WakeWordListener(model: model)
/// try await listener.start()
/// for await d in listener.detections() { print(d.name, d.confidence) }
/// ```
public actor WakeWordListener {
    public let threshold: Float
    /// Minimum time between two detections of the same wake word.
    public let debounce: TimeInterval

    private let model: WakeWordModel
    private var engine: AVAudioEngine?
    private var pending: [Int16] = []
    private var inflight = false
    private var lastDetectionAt: [String: Date] = [:]
    private var continuations: [UUID: AsyncStream<Detection>.Continuation] = [:]

    public init(model: WakeWordModel, threshold: Float = 0.5, debounce: TimeInterval = 2.0) {
        self.model = model
        self.threshold = threshold
        self.debounce = debounce
    }

    /// Starts capturing; call after microphone permission is granted.
    public func start() throws {
        if engine != nil { return }
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true, options: [])
        #endif

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let hwFormat = input.inputFormat(forBus: 0)
        guard hwFormat.sampleRate > 0 else { throw WakeWordError.unsupportedSampleRate(rate: 0) }
        guard
            let target = AVAudioFormat(
                commonFormat: .pcmFormatInt16, sampleRate: Double(model.sampleRate),
                channels: 1, interleaved: true),
            let converter = AVAudioConverter(from: hwFormat, to: target)
        else {
            throw WakeWordError.unsupportedSampleRate(rate: UInt32(hwFormat.sampleRate))
        }
        pending = []
        inflight = false
        lastDetectionAt = [:]
        self.engine = engine

        input.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self, let samples = Self.convert(buffer, converter, target) else { return }
            Task { await self.ingest(samples) }
        }
        engine.prepare()
        try engine.start()
    }

    /// Stops capturing and finishes every detection stream.
    public func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        #endif
        pending = []
        inflight = false
        for c in continuations.values { c.finish() }
        continuations.removeAll()
    }

    /// A new stream of detections; every stream gets every detection.
    public nonisolated func detections() -> AsyncStream<Detection> {
        AsyncStream { continuation in
            let id = UUID()
            Task { await self.add(id, continuation) }
            continuation.onTermination = { [weak self] _ in
                Task { await self?.remove(id) }
            }
        }
    }

    private func add(_ id: UUID, _ c: AsyncStream<Detection>.Continuation) { continuations[id] = c }

    private func remove(_ id: UUID) { continuations.removeValue(forKey: id) }

    private func ingest(_ samples: [Int16]) {
        pending.append(contentsOf: samples)
        guard !inflight, !pending.isEmpty else { return }
        inflight = true
        defer { inflight = false }
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        guard let scores = try? model.predict(batch) else { return }
        let now = Date()
        var fired = false
        for (name, confidence) in scores where confidence >= threshold {
            if let last = lastDetectionAt[name], now.timeIntervalSince(last) < debounce { continue }
            lastDetectionAt[name] = now
            let detection = Detection(name: name, confidence: confidence, timestamp: now)
            for c in continuations.values { c.yield(detection) }
            fired = true
        }
        if fired { model.reset() }
    }

    private static func convert(
        _ buffer: AVAudioPCMBuffer, _ converter: AVAudioConverter, _ target: AVAudioFormat
    ) -> [Int16]? {
        let ratio = target.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio)) + 8
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return nil }
        var consumed = false
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, outStatus in
            outStatus.pointee = consumed ? .noDataNow : .haveData
            defer { consumed = true }
            return consumed ? nil : buffer
        }
        guard status != .error, error == nil, let channel = out.int16ChannelData else { return nil }
        return Array(UnsafeBufferPointer(start: channel[0], count: Int(out.frameLength)))
    }
}
