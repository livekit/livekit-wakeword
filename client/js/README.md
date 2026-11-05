# JavaScript / TypeScript

Streaming wake word detection in the browser and in Node, on ONNX Runtime. Models come from
[training](../python/TRAINING.md) or the [release](../../README.md#models). Detection runs on the
device: in a browser, no audio leaves the page.

The package is not on npm yet. Build it from source (Node 20+ and
[pnpm](https://pnpm.io/installation)), then install the tarball into your app with the runtime you
use:

```bash
git clone https://github.com/livekit/livekit-wakeword
cd livekit-wakeword/client/js
pnpm install && pnpm pack                        # -> livekit-wakeword-<version>.tgz

cd /path/to/your/app
npm install /path/to/livekit-wakeword-<version>.tgz onnxruntime-web    # browser
npm install /path/to/livekit-wakeword-<version>.tgz onnxruntime-node   # Node
```

`@livekit/wakeword` picks the right binding itself: onnxruntime-web (WebAssembly) in browsers and
bundlers, onnxruntime-node in Node. Install the runtime you use; both are optional peer
dependencies.

## Browser

```ts
import { WakeWordListener, WakeWordModel } from '@livekit/wakeword';

const model = await WakeWordModel.create(['/models/hey_livekit.onnx']);
const listener = new WakeWordListener(model, { threshold: 0.5, debounce: 2 });

button.onclick = async () => {
  await listener.start(); // opens the microphone; call it from a user gesture
  for await (const detection of listener.detections()) {
    console.log(`Detected ${detection.name} (${detection.confidence})`);
    await talkToAgent(); // the listener is paused until the loop asks for the next detection
  }
  // the loop ends on listener.stop(), or throws if the model fails
};
```

After a detection the listener pauses (as in Python and Swift) until the next
`waitForDetection()`, which each turn of the `detections()` loop calls, so it cannot fire again
while you handle one. `onScores` is called with every 80 ms step's scores, for a level meter.

`start()` listens to the microphone by default. Pass `{ source }` to listen to any
`MediaStream` or `MediaStreamTrack` instead, such as a livekit-client track, so you can publish
the microphone only after the wake word:

```ts
import { createLocalAudioTrack } from 'livekit-client';

const track = await createLocalAudioTrack();
await listener.start({ source: track.mediaStreamTrack });
await listener.waitForDetection();
await room.localParticipant.publishTrack(track); // the agent hears you from here on
```

The listener runs in its own `AudioContext` (or pass `{ audioContext }`), captures audio with an
AudioWorklet, resamples it to 16 kHz and runs the model on the main thread: one 80 ms step takes
about 1 ms in WebAssembly.

**Serving the model and runtime.** The standard model is about 7 MB and onnxruntime-web's
WebAssembly about 15 MB (4 MB gzipped); browsers cache both after the first visit. Bundlers such
as Vite and webpack emit the `.wasm` file next to your bundle. With Vite, exclude the runtime
from dependency pre-bundling so it finds that file:

```ts
// vite.config.ts
export default defineConfig({ optimizeDeps: { exclude: ['onnxruntime-web'] } });
```

To serve the `.wasm` from somewhere else (a CDN), set `env.wasm.wasmPaths` before loading a
model: `import { env } from '@livekit/wakeword'`. Multi-threaded WebAssembly needs a
cross-origin isolated page (COOP/COEP headers); the model runs fine on one thread.

## Node

```ts
import { WakeWordModel } from '@livekit/wakeword';

const model = await WakeWordModel.create(['hey_livekit.onnx'], { sampleRate: 48000 });
const scores = await model.predict(frame); // { hey_livekit: 0.97 }; Int16Array or Float32Array
if (scores.hey_livekit >= 0.5) console.log('Wake word detected!');
```

`predict()` takes mono audio of any length at `sampleRate` (16 kHz by default; other rates are
resampled internally) and resolves to each model's highest score among the 80 ms steps the call
completed. Calls are queued, so you do not need to await one before the next. `predictClip()`
scores a whole clip from a fresh state, and `reset()` forgets the audio heard so far.

With [`@livekit/rtc-node`](https://www.npmjs.com/package/@livekit/rtc-node), for example in a
LiveKit agent, feed it a participant's track:

```ts
import { AudioStream } from '@livekit/rtc-node';
import { DetectionGate } from '@livekit/wakeword';

const gate = new DetectionGate(0.5, 2); // threshold, debounce in seconds
for await (const frame of new AudioStream(track, 16000, 1)) {
  const detection = gate.check(await model.predict(frame.data), performance.now());
  if (detection) console.log(`${detection.name} (${detection.confidence})`);
}
```

## Other runtimes

`@livekit/wakeword/core` is the same model on any ONNX Runtime JavaScript package, such as
onnxruntime-react-native:

```ts
import * as ort from 'onnxruntime-react-native';
import { WakeWordModel } from '@livekit/wakeword/core';

const model = new WakeWordModel(ort);
await model.loadModel(bytes, 'hey_livekit');
```

## Example

[`example/`](example) is a browser demo with a live score meter. From this folder:

```bash
pnpm install
pnpm example        # http://localhost:5173, then say "hey livekit"
```

## Development

```bash
pnpm install
pnpm lint && pnpm typecheck
pnpm test           # Node: both bindings, checked step by step against the Python client
pnpm exec playwright install chromium
pnpm test:browser   # the example in headless Chromium, with a fake microphone
```

`scripts/make_parity.py` regenerates the Python reference scores in `test/fixtures/parity.json`
(`uv run client/js/scripts/make_parity.py` from the repo root).
