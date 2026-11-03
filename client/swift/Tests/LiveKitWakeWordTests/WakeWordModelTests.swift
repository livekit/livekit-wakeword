// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import XCTest

@testable import LiveKitWakeWord

/// The shipped hey_livekit model on the clips shared with the Python tests.
final class WakeWordModelTests: XCTestCase {
    private static let threshold: Float = 0.5

    private func model(sampleRate: UInt32 = 16_000) throws -> WakeWordModel {
        try WakeWordModel(models: [try fixture("hey_livekit.onnx")], sampleRate: sampleRate, executionProvider: .cpu)
    }

    private func score(_ wav: String) throws -> Float {
        let (rate, samples) = try readWav(wav)
        return try model(sampleRate: rate).predictClip(samples)["hey_livekit"] ?? -1
    }

    func testScoresAreProbabilities() throws {
        let model = try model()
        let sine = (0..<32_000).map { Int16(sin(2 * Double.pi * 440 * Double($0) / 16_000) * 32767) }
        let score = try XCTUnwrap(model.predictClip(sine)["hey_livekit"])
        XCTAssertTrue((0...1).contains(score))
        XCTAssertEqual(try model.predictClip(Array(sine[..<800]))["hey_livekit"], 0)  // < one 80 ms step
    }

    func testPositiveWavAboveThreshold() throws {
        XCTAssertGreaterThanOrEqual(try score("positive.wav"), Self.threshold)
    }

    func testNegativeWavBelowThreshold() throws {
        XCTAssertLessThan(try score("negative.wav"), Self.threshold)
    }

    func testLoadAndUnloadModel() throws {
        let model = try WakeWordModel(models: [], sampleRate: 16_000, executionProvider: .cpu)
        try model.loadModel(url: try fixture("hey_livekit.onnx"), name: "custom")
        XCTAssertEqual(model.modelNames, ["custom"])
        XCTAssertNotNil(try model.predictClip([Int16](repeating: 0, count: 32_000))["custom"])
        XCTAssertTrue(model.unloadModel(name: "custom"))
        XCTAssertEqual(model.modelNames, [])
    }
}
