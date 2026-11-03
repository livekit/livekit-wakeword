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

/// The listener pauses after a detection until the next `waitForDetection()`, as in Python.
/// Audio goes in through `ingest` (no microphone); threshold 0 makes every step a detection.
final class ListenerTests: XCTestCase {
    private let step = [Int16](repeating: 0, count: 1280)

    private func listener() throws -> WakeWordListener {
        let model = try WakeWordModel(
            models: [try fixture("streaming_tiny.onnx")], sampleRate: 16_000, executionProvider: .cpu)
        return WakeWordListener(model: model, threshold: 0, debounce: 0)
    }

    func testPausesAfterADetectionUntilAskedAgain() async throws {
        let listener = try listener()
        await listener.startForTesting()
        await listener.ingest(step)  // heard before anyone asked: kept for the first wait
        let first = try await listener.waitForDetection()
        XCTAssertEqual(first.name, "streaming_tiny")

        let waiter = Task { try await listener.waitForDetection() }
        while await !listener.hasWaiter { await Task.yield() }
        await listener.ingest(step)
        let second = try await waiter.value
        XCTAssertGreaterThanOrEqual(second.timestamp, first.timestamp)

        // paused now: audio is dropped until the next wait, so nothing is pending
        await listener.ingest(step)
        await listener.ingest(step)
        let heardWhilePaused = await listener.hasPending
        XCTAssertFalse(heardWhilePaused)
        let third = Task { try await listener.waitForDetection() }
        while await !listener.hasWaiter { await Task.yield() }
        await listener.ingest(step)
        let detection = try await third.value
        XCTAssertGreaterThan(detection.timestamp, second.timestamp)
    }

    func testStopEndsWaitsAndLoops() async throws {
        let listener = try listener()
        do {
            _ = try await listener.waitForDetection()
            XCTFail("waiting before start() should throw")
        } catch WakeWordError.listenerStopped {}

        await listener.startForTesting()
        let loop = Task { () -> Int in
            var n = 0
            for await _ in listener.detections() { n += 1 }
            return n
        }
        while await !listener.hasWaiter { await Task.yield() }
        await listener.ingest(step)
        while await !listener.hasWaiter { await Task.yield() }  // the loop asked for the next one
        await listener.stop()
        let count = await loop.value
        XCTAssertEqual(count, 1)
    }

    func testCancellingAWaitThrows() async throws {
        let listener = try listener()
        await listener.startForTesting()
        let waiter = Task { try await listener.waitForDetection() }
        while await !listener.hasWaiter { await Task.yield() }
        waiter.cancel()
        do {
            _ = try await waiter.value
            XCTFail("a cancelled wait should throw")
        } catch is CancellationError {}
    }
}
