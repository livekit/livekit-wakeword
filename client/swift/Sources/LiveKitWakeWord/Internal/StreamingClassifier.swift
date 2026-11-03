// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// One streaming model: inputs `chunk` (1, 1280) + state, outputs `score` + `<state>_out`.
/// The state starts at zero and is warmed up on 2 s of silence, as in the Python runtime.
final class StreamingClassifier {
    static let chunkSamples = 1280
    static let warmupSteps = 25

    let name: String
    private let session: ORTSession
    private let stateOutputs: [String]
    private var state: [String: ORTValue] = [:]
    private var warm: [String: ORTValue] = [:]
    private var pending: [Float] = []

    init(name: String, url: URL, session: ORTSession) throws {
        self.name = name
        self.session = session
        let inputs = try ONNXInput.read(url: url)
        guard inputs.contains(where: { $0.name == "chunk" }) else {
            throw WakeWordError.notStreamingModel(url: url)
        }
        stateOutputs = try ORTRuntime.wrap { try session.outputNames().filter { $0 != "score" } }
        for input in inputs where input.name != "chunk" {
            guard input.shape.allSatisfy({ $0 > 0 }) else { throw WakeWordError.notStreamingModel(url: url) }
            let isInt64 = input.elementType == ONNXInput.int64
            state[input.name] = try Self.tensor(
                Data(count: input.shape.reduce(1, *) * (isInt64 ? 8 : 4)),
                type: isInt64 ? .int64 : .float, shape: input.shape)
        }
        let silence = [Float](repeating: 0, count: Self.chunkSamples)
        for _ in 0..<Self.warmupSteps { _ = try step(silence) }
        warm = state
    }

    func reset() {
        state = warm
        pending = []
    }

    /// Feeds 16 kHz audio; returns the highest score of the 80 ms steps completed, if any.
    func feed(_ audio: [Float]) throws -> Float? {
        pending.append(contentsOf: audio)
        var best: Float?
        while pending.count >= Self.chunkSamples {
            let score = try step(Array(pending.prefix(Self.chunkSamples)))
            pending.removeFirst(Self.chunkSamples)
            best = max(best ?? 0, score)
        }
        return best
    }

    private func step(_ chunk: [Float]) throws -> Float {
        var inputs = state
        inputs["chunk"] = try Self.tensor(
            chunk.withUnsafeBufferPointer { Data(buffer: $0) }, type: .float, shape: [1, chunk.count])
        let outputs = try ORTRuntime.wrap {
            try session.run(withInputs: inputs, outputNames: Set(["score"] + stateOutputs), runOptions: nil)
        }
        guard let score = outputs["score"] else { throw WakeWordError.modelOutputMissing(key: "score") }
        for out in stateOutputs {
            guard let v = outputs[out] else { throw WakeWordError.modelOutputMissing(key: out) }
            state[String(out.dropLast("_out".count))] = v
        }
        return try ORTRuntime.wrap { try score.tensorData().bytes.assumingMemoryBound(to: Float.self).pointee }
    }

    private static func tensor(_ data: Data, type: ORTTensorElementDataType, shape: [Int]) throws -> ORTValue {
        try ORTRuntime.wrap {
            try ORTValue(
                tensorData: NSMutableData(data: data), elementType: type,
                shape: shape.map { NSNumber(value: $0) })
        }
    }
}
