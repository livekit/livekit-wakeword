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

/// Streams the microphone through a ``WakeWordModel`` and reports ``Detection``s.
///
/// After a detection the listener pauses (audio is dropped and the model is reset) until the next
/// ``waitForDetection()``, so it cannot fire again while you handle one (e.g. while an agent
/// talks). This is the same as the Python `WakeWordListener`.
///
/// ```swift
/// let model = try WakeWordModel(models: [heyLiveKitURL], sampleRate: 16_000)
/// let listener = WakeWordListener(model: model)
/// try await listener.start()
/// for await d in listener.detections() {  // each turn of the loop resumes listening
///     print(d.name, d.confidence)
///     await talkToAgent()                  // no detections in the meantime
/// }
/// ```
///
/// If the model fails, the listener stops and ``error`` says why.
public actor WakeWordListener {
    public let threshold: Float
    /// Minimum time between two detections of the same wake word.
    public let debounce: TimeInterval

    /// Why the listener stopped on its own (the model failed), if it did; cleared by ``start()``.
    public private(set) var error: Error?

    private let model: WakeWordModel
    private var engine: AVAudioEngine?
    // microphone buffers, in capture order, to the task that runs the model
    private var audio: AsyncStream<[Int16]>.Continuation?
    private var consumer: Task<Void, Never>?
    private var lastDetectionAt: [String: Date] = [:]
    private var paused = false  // after a detection, until the next waitForDetection()
    private var pending: Detection?  // a detection nobody was waiting for yet
    private var waiters: [UUID: AsyncStream<Detection>.Continuation] = [:]

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
        lastDetectionAt = [:]
        error = nil
        paused = false
        pending = nil
        self.engine = engine

        // One stream rather than a Task per buffer: separate Tasks may reach the actor out of
        // order, and the model must see the audio in order.
        let (stream, continuation) = AsyncStream<[Int16]>.makeStream()
        audio = continuation
        consumer = Task { [weak self] in
            for await samples in stream {
                guard let self else { return }
                await self.ingest(samples)
            }
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { buffer, _ in
            guard let samples = Self.convert(buffer, converter, target) else { return }
            continuation.yield(samples)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            stop()
            throw error
        }
    }

    /// Stops capturing; pending ``waitForDetection()`` calls throw and detection loops end.
    public func stop() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        audio?.finish()
        audio = nil
        consumer?.cancel()
        consumer = nil
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        #endif
        testing = false
        paused = false
        pending = nil
        for c in waiters.values { c.finish() }
        waiters.removeAll()
    }

    /// Resumes listening and returns the next detection. Throws ``error`` if the model failed,
    /// ``WakeWordError/listenerStopped`` after ``stop()`` (or before ``start()``), and
    /// `CancellationError` if the calling task is cancelled.
    public func waitForDetection() async throws -> Detection {
        guard running else { throw error ?? WakeWordError.listenerStopped }
        paused = false
        if let detection = pending {  // heard before anyone asked
            pending = nil
            return detection
        }
        let (stream, continuation) = AsyncStream<Detection>.makeStream()
        let id = UUID()
        waiters[id] = continuation
        defer { waiters.removeValue(forKey: id) }
        for await detection in stream { return detection }
        try Task.checkCancellation()
        throw error ?? WakeWordError.listenerStopped
    }

    /// Detections, one per turn of a `for await` loop: each turn resumes listening (as
    /// ``waitForDetection()`` does) and the loop ends when the listener stops.
    public nonisolated func detections() -> Detections { Detections(listener: self) }

    /// Calls ``waitForDetection()`` for each element.
    public struct Detections: AsyncSequence, Sendable {
        public typealias Element = Detection
        let listener: WakeWordListener

        public struct AsyncIterator: AsyncIteratorProtocol {
            let listener: WakeWordListener
            public mutating func next() async -> Detection? {
                try? await listener.waitForDetection()
            }
        }

        public func makeAsyncIterator() -> AsyncIterator { AsyncIterator(listener: listener) }
    }

    // Capturing (or, in tests, fed by `ingest` directly).
    var running: Bool { engine != nil || testing }
    private var testing = false

    var hasWaiter: Bool { !waiters.isEmpty }
    var hasPending: Bool { pending != nil }

    /// Tests feed audio through `ingest` without a microphone.
    func startForTesting() {
        testing = true
        paused = false
        pending = nil
    }

    func ingest(_ samples: [Int16]) {
        guard running, !paused else { return }  // stopped, or paused after a detection
        let scores: [String: Float]
        do {
            scores = try model.predict(samples)
        } catch {
            // like the Python listener's crashed audio loop: stop, and say why in `error`
            self.error = error
            stop()
            return
        }
        let now = Date()
        for (name, confidence) in scores.sorted(by: { $0.value > $1.value })
        where confidence >= threshold {
            if let last = lastDetectionAt[name], now.timeIntervalSince(last) < debounce { continue }
            lastDetectionAt[name] = now
            let detection = Detection(name: name, confidence: confidence, timestamp: now)
            paused = true  // until the next waitForDetection()
            model.reset()
            if waiters.isEmpty {
                pending = detection
            } else {
                for c in waiters.values { c.yield(detection) }
            }
            return  // one detection per pause, as in Python
        }
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
