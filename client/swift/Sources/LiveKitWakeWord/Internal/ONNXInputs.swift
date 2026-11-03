// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation

/// A graph input of an `.onnx` file, read from the protobuf (ORT's ObjC API cannot report input
/// shapes): ModelProto.graph(7).input(11) { name(1), type(2).tensor_type(1) { elem_type(1),
/// shape(2).dim(1).dim_value(1) } }.
struct ONNXInput {
    static let int64 = 7

    let name: String
    let elementType: Int
    let shape: [Int]  // -1 for symbolic dims

    static func read(url: URL) throws -> [ONNXInput] {
        let bytes = [UInt8](try Data(contentsOf: url, options: .mappedIfSafe))
        return try bytes.withUnsafeBufferPointer { buf in
            var inputs: [ONNXInput] = []
            var model = Reader(buf: buf, pos: 0, end: buf.count)
            while let (field, graph) = try model.next() {
                guard field == 7, var graph else { continue }
                while let (f, info) = try graph.next() {
                    if f == 11, let info { inputs.append(try parse(info)) }
                }
            }
            return inputs
        }
    }

    private static func parse(_ info: Reader) throws -> ONNXInput {
        var info = info
        var name = ""
        var elementType = 0
        var shape: [Int] = []
        while let (field, value) = try info.next() {
            if field == 1, let value { name = value.string() }
            guard field == 2, var type = value else { continue }
            while let (f, t) = try type.next() {
                guard f == 1, var tensor = t else { continue }
                while let (f, v) = try tensor.next() {
                    if f == 1 { elementType = tensor.varint }
                    guard f == 2, var dims = v else { continue }
                    while let (f, d) = try dims.next() {
                        guard f == 1, var dim = d else { continue }
                        var size = -1
                        while let (f, _) = try dim.next() {
                            if f == 1 { size = dim.varint }
                        }
                        shape.append(size)
                    }
                }
            }
        }
        return ONNXInput(name: name, elementType: elementType, shape: shape)
    }

    /// Protobuf wire-format reader over `buf[pos..<end]`.
    private struct Reader {
        let buf: UnsafeBufferPointer<UInt8>
        var pos: Int
        let end: Int
        /// Value of the last varint field returned by ``next()``.
        private(set) var varint = 0

        /// Next field number, with a sub-reader for length-delimited fields.
        mutating func next() throws -> (Int, Reader?)? {
            guard pos < end else { return nil }
            let key = try readVarint()
            switch key & 7 {
            case 0: varint = Int(truncatingIfNeeded: try readVarint())
            case 1: pos += 8
            case 5: pos += 4
            case 2:
                let n = Int(try readVarint())
                guard n >= 0, pos + n <= end else { throw ONNXInput.malformed }
                defer { pos += n }
                return (Int(key >> 3), Reader(buf: buf, pos: pos, end: pos + n))
            default: throw ONNXInput.malformed
            }
            return (Int(key >> 3), nil)
        }

        func string() -> String {
            String(decoding: UnsafeBufferPointer(rebasing: buf[pos..<end]), as: UTF8.self)
        }

        private mutating func readVarint() throws -> UInt64 {
            var result: UInt64 = 0
            var shift: UInt64 = 0
            while pos < end, shift < 64 {
                let b = buf[pos]
                pos += 1
                result |= UInt64(b & 0x7F) << shift
                if b & 0x80 == 0 { return result }
                shift += 7
            }
            throw ONNXInput.malformed
        }
    }

    private static let malformed = WakeWordError.runtimeFailure(
        underlying: NSError(
            domain: "LiveKitWakeWord", code: -2,
            userInfo: [NSLocalizedDescriptionKey: "malformed ONNX file"]))
}
