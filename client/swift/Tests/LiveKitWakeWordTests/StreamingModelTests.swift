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

/// The Swift runtime reproduces the Python one step for step (fixtures from
/// `client/python/tests/train/make_streaming_tiny.py`: per-80 ms scores of a deterministic chirp).
final class StreamingModelTests: XCTestCase {
    private struct Expected: Decodable {
        let scores: [Float]
    }

    func testMatchesPythonRuntime() throws {
        let model = try WakeWordModel(
            models: [try fixture("streaming_tiny.onnx")], sampleRate: 16_000, executionProvider: .cpu)
        let expected = try JSONDecoder().decode(
            Expected.self, from: Data(contentsOf: try fixture("streaming_tiny_expected.json"))
        ).scores
        let pcm: [Int16] = (0..<(16_000 * 3)).map { i in
            let t = Double(i) / 16_000
            return Int16(Int(sin(2 * Double.pi * (150 + 400 * t) * t) * 9000))
        }

        let scores = try stride(from: 0, through: pcm.count - 1280, by: 1280).map {
            try model.predict(Array(pcm[$0..<($0 + 1280)]))["streaming_tiny"] ?? -1
        }
        XCTAssertEqual(scores.count, expected.count)
        for (a, b) in zip(scores, expected) { XCTAssertEqual(a, b, accuracy: 1e-4) }

        // uneven pieces after a reset reproduce the same stream
        model.reset()
        var best: Float = 0
        var pos = 0
        for size in Array(repeating: [700, 1860, 1280, 333, 2227], count: 20).joined() where pos < pcm.count {
            let end = min(pcm.count, pos + size)
            best = max(best, try model.predict(Array(pcm[pos..<end]))["streaming_tiny"] ?? 0)
            pos = end
        }
        XCTAssertEqual(best, expected.max()!, accuracy: 1e-4)
    }
}
