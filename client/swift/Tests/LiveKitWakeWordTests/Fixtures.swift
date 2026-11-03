// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import XCTest

/// A test fixture, shared with the Python tests: `models/hey_livekit.onnx`, else
/// `client/python/tests/fixtures/<name>`.
func fixture(_ name: String) throws -> URL {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../../..").standardized
    let dir = name == "hey_livekit.onnx" ? "models" : "client/python/tests/fixtures"
    let url = root.appendingPathComponent(dir).appendingPathComponent(name)
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: url.path])
    }
    return url
}

/// A 16-bit mono PCM WAV fixture as (sample rate, samples).
func readWav(_ name: String) throws -> (UInt32, [Int16]) {
    let bytes = [UInt8](try Data(contentsOf: try fixture(name)))
    func u32(_ i: Int) -> Int { Int(bytes[i]) | Int(bytes[i + 1]) << 8 | Int(bytes[i + 2]) << 16 | Int(bytes[i + 3]) << 24 }
    var rate = 0
    var i = 12
    while i + 8 <= bytes.count {
        let id = String(decoding: bytes[i..<(i + 4)], as: UTF8.self)
        let size = u32(i + 4)
        if id == "fmt " { rate = u32(i + 12) }
        if id == "data" {
            let samples = stride(from: i + 8, to: min(i + 8 + size, bytes.count) - 1, by: 2).map {
                Int16(bitPattern: UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8)
            }
            return (UInt32(rate), samples)
        }
        i += 8 + size + size % 2
    }
    throw CocoaError(.fileReadCorruptFile)
}
