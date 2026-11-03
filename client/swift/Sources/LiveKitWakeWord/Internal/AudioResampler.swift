// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import AVFoundation
import Foundation

/// Mono Int16 PCM at any rate -> 16 kHz Float32.
final class AudioResampler {
    private let inputFormat: AVAudioFormat
    private let outputFormat: AVAudioFormat
    private let converter: AVAudioConverter

    init(inputSampleRate: UInt32) throws {
        guard
            let inFmt = AVAudioFormat(
                commonFormat: .pcmFormatInt16, sampleRate: Double(inputSampleRate),
                channels: 1, interleaved: true),
            let outFmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(WakeWordModel.modelSampleRate),
                channels: 1, interleaved: false),
            let conv = AVAudioConverter(from: inFmt, to: outFmt)
        else {
            throw WakeWordError.unsupportedSampleRate(rate: inputSampleRate)
        }
        inputFormat = inFmt
        outputFormat = outFmt
        converter = conv
    }

    func resample(samples: UnsafeBufferPointer<Int16>) throws -> [Float] {
        let count = samples.count
        guard count > 0, let base = samples.baseAddress else { return [] }
        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        guard
            let input = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(count)),
            let output = AVAudioPCMBuffer(
                pcmFormat: outputFormat,
                frameCapacity: AVAudioFrameCount(ceil(Double(count) * ratio)) + 8)
        else {
            throw WakeWordError.resamplingFailed(underlying: nil)
        }
        input.frameLength = AVAudioFrameCount(count)
        input.int16ChannelData?[0].update(from: base, count: count)

        var consumed = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, outStatus in
            outStatus.pointee = consumed ? .noDataNow : .haveData
            defer { consumed = true }
            return consumed ? nil : input
        }
        if status == .error || error != nil {
            throw WakeWordError.resamplingFailed(underlying: error)
        }
        guard let ptr = output.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: ptr, count: Int(output.frameLength)))
    }
}
