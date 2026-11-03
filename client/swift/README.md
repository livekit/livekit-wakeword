# Swift

Streaming wake word detection for iOS 16+ and macOS 14+, on ONNX Runtime (CoreML execution
provider by default). Models come from [training](../python/TRAINING.md) or the
[release](../../README.md#models).

```swift
.package(url: "https://github.com/livekit/livekit-wakeword", from: "2.0.0"),
```

```swift
import LiveKitWakeWord

let url = Bundle.main.url(forResource: "hey_livekit", withExtension: "onnx")!
let model = try WakeWordModel(models: [url], sampleRate: 16_000)  // any rate; resampled internally

let scores = try model.predict(audioChunk)  // Int16 PCM as it arrives, any chunk size
if (scores["hey_livekit"] ?? 0) > 0.5 { print("Wake word detected!") }

let listener = WakeWordListener(model: model, threshold: 0.5, debounce: 2.0)
try await listener.start()
for await detection in listener.detections() {
    print("Detected \(detection.name) (\(detection.confidence))")
    await talkToAgent()  // the listener is paused until the loop asks for the next detection
}
// the loop ends on stop(), or if the model fails: then `await listener.error` says why
```

After a detection the listener pauses (as in Python) until the next `waitForDetection()`, which
each turn of the `detections()` loop calls, so it cannot fire again while you handle one.

`executionProvider:` picks the ONNX Runtime provider. The listener needs `NSMicrophoneUsageDescription`
in Info.plist (and `com.apple.security.device.audio-input` for sandboxed macOS apps).

[`Example/`](Example) is a SwiftUI demo for iOS and macOS. The package manifest is
[`Package.swift`](../../Package.swift) at the repo root, so Xcode can add it by URL; `swift test`
runs the tests from the root.
