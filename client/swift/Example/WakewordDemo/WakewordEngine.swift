// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

@preconcurrency import AVFoundation
import Foundation
import LiveKitWakeWord
import SwiftUI

enum ComputeBackend: String, CaseIterable, Identifiable, Sendable {
    case coreML
    case coreMLCPUAndGPU
    case coreMLCPUOnly
    case cpu

    var id: String { rawValue }

    var label: String {
        switch self {
        case .coreML: return "CoreML (ANE+GPU+CPU)"
        case .coreMLCPUAndGPU: return "CoreML (GPU+CPU)"
        case .coreMLCPUOnly: return "CoreML (CPU)"
        case .cpu: return "ORT CPU"
        }
    }

    var executionProvider: ExecutionProvider {
        switch self {
        case .coreML: return .coreML
        case .coreMLCPUAndGPU: return .coreMLCPUAndGPU
        case .coreMLCPUOnly: return .coreMLCPUOnly
        case .cpu: return .cpu
        }
    }
}

/// Streams the microphone through `WakeWordModel` and publishes the score for the UI.
final class WakewordEngine: ObservableObject, @unchecked Sendable {

    // MARK: - Published UI state (mutated on MainActor)

    @Published private(set) var isRunning = false
    @Published private(set) var score: Float = 0
    @Published private(set) var isTriggered = false
    @Published private(set) var lastError: String?
    @Published private(set) var volume: Float = 0
    @Published private(set) var tick: UInt64 = 0
    @Published var backend: ComputeBackend = .coreML {
        didSet {
            guard oldValue != backend else { return }
            Task { @MainActor in self.rebuildModel() }
        }
    }

    // MARK: - Tuning

    private let triggerThreshold: Float = 0.5
    private let triggerHoldDuration: TimeInterval = 1.5

    // MARK: - Wake-word model

    private let classifierURLs: [URL]
    private var model: WakeWordModel?

    // MARK: - Audio capture

    private var engine: AVAudioEngine?
    private let workQueue = DispatchQueue(label: "io.livekit.wakeword.predict", qos: .userInteractive)

    private let pendingLock = NSLock()
    private var pending: [Int16] = []
    private var predictInFlight = false

    private var triggerResetTask: Task<Void, Never>?

    // MARK: - Init

    init() throws {
        guard let url = Bundle.main.url(forResource: "hey_livekit", withExtension: "onnx") else {
            throw NSError(
                domain: "WakewordEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "hey_livekit.onnx not found in app bundle"]
            )
        }
        self.classifierURLs = [url]
    }

    // MARK: - Public API

    @MainActor
    func toggle() {
        if isRunning {
            stop()
        } else {
            Task { await self.startAfterAuth() }
        }
    }

    // MARK: - Permission + start

    @MainActor
    private func startAfterAuth() async {
        let granted = await requestMicrophonePermission()
        guard granted else {
            lastError = "Microphone permission denied. Enable it in System Settings → Privacy & Security → Microphone."
            return
        }
        do {
            try start()
            lastError = nil
        } catch {
            lastError = "Start failed: \(error.localizedDescription)"
            stopInternal()
        }
    }

    private func requestMicrophonePermission() async -> Bool {
        if #available(iOS 17.0, macOS 14.0, *) {
            if AVAudioApplication.shared.recordPermission == .granted { return true }
            return await AVAudioApplication.requestRecordPermission()
        } else {
            #if os(iOS)
            let session = AVAudioSession.sharedInstance()
            if session.recordPermission == .granted { return true }
            return await withCheckedContinuation { cont in
                session.requestRecordPermission { cont.resume(returning: $0) }
            }
            #else
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized: return true
            case .notDetermined:
                return await withCheckedContinuation { cont in
                    AVCaptureDevice.requestAccess(for: .audio) { cont.resume(returning: $0) }
                }
            default: return false
            }
            #endif
        }
    }

    // MARK: - Start / Stop

    @MainActor
    private func start() throws {
        try configureAudioSession()

        let engine = AVAudioEngine()
        self.engine = engine

        let input = engine.inputNode
        let hwFormat = input.inputFormat(forBus: 0)

        guard hwFormat.sampleRate > 0 else {
            throw NSError(
                domain: "WakewordEngine",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Input has no valid sample rate (is a microphone connected?)"]
            )
        }

        if model == nil {
            model = try WakeWordModel(
                models: classifierURLs,
                sampleRate: WakeWordModel.modelSampleRate,
                executionProvider: backend.executionProvider
            )
        }
        let modelRate = Double(WakeWordModel.modelSampleRate)
        model?.reset()
        pendingLock.lock()
        pending.removeAll(keepingCapacity: true)
        predictInFlight = false
        pendingLock.unlock()

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: modelRate,
            channels: 1,
            interleaved: true
        ) else {
            throw NSError(
                domain: "WakewordEngine",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Could not create target Int16 format"]
            )
        }
        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else {
            throw NSError(
                domain: "WakewordEngine",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Could not create AVAudioConverter"]
            )
        }

        input.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            self?.handleInput(buffer: buffer, converter: converter, targetFormat: targetFormat)
        }

        engine.prepare()
        try engine.start()
        isRunning = true
    }

    @MainActor
    private func stop() {
        stopInternal()
    }

    @MainActor
    private func stopInternal() {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil

        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        #endif

        pendingLock.lock()
        pending.removeAll(keepingCapacity: true)
        predictInFlight = false
        pendingLock.unlock()

        isRunning = false
        score = 0
        volume = 0
        tick = 0
        isTriggered = false
        triggerResetTask?.cancel()
        triggerResetTask = nil
    }

    @MainActor
    private func rebuildModel() {
        let wasRunning = isRunning
        if wasRunning {
            stopInternal()
        }
        model = nil
        if wasRunning {
            Task { await self.startAfterAuth() }
        }
    }

    @MainActor
    private func configureAudioSession() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setActive(true, options: [])
        #endif
    }

    // MARK: - Audio tap (real-time thread)

    private func handleInput(
        buffer inputBuffer: AVAudioPCMBuffer,
        converter: AVAudioConverter,
        targetFormat: AVAudioFormat
    ) {
        guard let outBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: inputBuffer.frameCapacity
        ) else { return }

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
              let channelData = outBuffer.int16ChannelData else {
            return
        }

        let frameCount = Int(outBuffer.frameLength)
        guard frameCount > 0 else { return }

        let level = computeLevel(samples: channelData[0], count: frameCount)
        Task { @MainActor [weak self] in
            self?.publish(volume: level)
        }

        if let batch = append(samples: channelData[0], count: frameCount) {
            workQueue.async { [weak self] in
                self?.runPredict(batch: batch)
            }
        }
    }

    private func append(samples: UnsafePointer<Int16>, count: Int) -> [Int16]? {
        pendingLock.lock()
        defer { pendingLock.unlock() }
        pending.append(contentsOf: UnsafeBufferPointer(start: samples, count: count))
        guard !predictInFlight else { return nil }
        predictInFlight = true
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        return batch
    }

    private func computeLevel(samples: UnsafePointer<Int16>, count: Int) -> Float {
        guard count > 0 else { return 0 }
        var sumSquares: Double = 0
        for i in 0..<count {
            let v = Double(samples[i])
            sumSquares += v * v
        }
        let rms = sqrt(sumSquares / Double(count))
        let normalized = rms / Double(Int16.max)
        let floorDb: Double = -60
        let db: Double
        if normalized <= 1e-7 {
            db = floorDb
        } else {
            db = max(floorDb, 20.0 * log10(normalized))
        }
        let level = (db - floorDb) / -floorDb
        return Float(max(0, min(1, level)))
    }

    private func runPredict(batch: [Int16]) {
        defer {
            pendingLock.lock()
            predictInFlight = false
            pendingLock.unlock()
        }
        guard let model else { return }
        do {
            let scores = try model.predict(batch)
            let maxScore = scores.values.max() ?? 0
            Task { @MainActor [weak self] in
                self?.publish(score: maxScore)
            }
        } catch {
            Task { @MainActor [weak self] in
                self?.lastError = "predict failed: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - UI publish (MainActor)

    @MainActor
    private func publish(volume newVolume: Float) {
        let alpha: Float = 0.35
        volume = (alpha * newVolume) + ((1 - alpha) * volume)
        tick &+= 1
    }

    @MainActor
    private func publish(score newScore: Float) {
        score = newScore
        if newScore >= triggerThreshold {
            isTriggered = true
            triggerResetTask?.cancel()
            let hold = triggerHoldDuration
            triggerResetTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(hold * 1_000_000_000))
                if !Task.isCancelled {
                    await MainActor.run { self?.isTriggered = false }
                }
            }
        }
    }
}
