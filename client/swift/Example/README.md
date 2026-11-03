# Wake Word Demo (iOS + macOS)

SwiftUI app that streams the microphone through the [`LiveKitWakeWord`](..) package.
Tap **Unmute mic** and say "Hey LiveKit": when the score crosses 0.5 the status turns green. A
picker switches the ONNX Runtime execution provider (CoreML ANE+GPU+CPU, GPU+CPU, CPU, or ORT
CPU) to compare latency.

Two schemes build from the same sources: **`WakewordDemo`** (iOS 16+) and **`WakewordDemoMac`**
(macOS 14+).

## Run

Open `WakewordDemo.xcodeproj` in Xcode 15+; the local Swift package and ONNX Runtime resolve
automatically.

- **macOS:** scheme `WakewordDemoMac`, destination *My Mac*, `Cmd+R`, grant microphone access.
- **iOS:** scheme `WakewordDemo`, a simulator or device (set a signing team), `Cmd+R`.

The project is generated from [`project.yml`](./project.yml) with
[XcodeGen](https://github.com/yonaskolb/XcodeGen) (`xcodegen generate`).

## Files

| Path | Purpose |
| --- | --- |
| `WakewordDemo/WakewordEngine.swift` | Microphone tap, feeds new audio to `WakeWordModel.predict()` |
| `WakewordDemo/ContentView.swift` | Score graph, level meter, mic and provider controls |
| `../../../models/hey_livekit.onnx` | The wake word model (shared with the tests and the other examples) |

## Other wake words

Train one with `livekit-wakeword train --wakeword "<phrase>"` (see
[`train/`](../../../train)), add the `.onnx` to the app's resources and to `classifierURLs` in
`WakewordEngine.init()`. `predict()` returns a score per model, keyed by
filename stem.
