// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import XCTest

@testable import LiveKitWakeWord

/// Latency of one 80 ms step per execution provider:
/// `LIVEKIT_WAKEWORD_BENCH=1 swift test --filter BenchmarkTests`.
final class BenchmarkTests: XCTestCase {
    func testPredictPerformanceCoreML() throws { try bench(.coreML) }

    func testPredictPerformanceCoreMLCPUAndGPU() throws { try bench(.coreMLCPUAndGPU) }

    func testPredictPerformanceCPU() throws { try bench(.cpu) }

    private func bench(_ provider: ExecutionProvider) throws {
        guard ProcessInfo.processInfo.environment["LIVEKIT_WAKEWORD_BENCH"] != nil else {
            throw XCTSkip("set LIVEKIT_WAKEWORD_BENCH=1 to run benchmarks")
        }
        let model = try WakeWordModel(
            models: [try fixture("hey_livekit.onnx")], sampleRate: 16_000, executionProvider: provider)
        let audio = [Int16](repeating: 0, count: 1280)
        for _ in 0..<10 { _ = try model.predict(audio) }
        let ms = try (0..<200).map { _ -> Double in
            let start = CFAbsoluteTimeGetCurrent()
            _ = try model.predict(audio)
            return (CFAbsoluteTimeGetCurrent() - start) * 1000
        }.sorted()
        print(String(format: "[bench] %@: p50=%.3f ms p95=%.3f ms p99=%.3f ms", "\(provider)", ms[100], ms[190], ms[198]))
    }
}
