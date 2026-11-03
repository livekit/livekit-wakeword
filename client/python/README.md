# Python

Streaming wake word detection with ONNX Runtime (Python 3.11+). Models come from
[training](TRAINING.md) or the [release](../../README.md#models).

```bash
uv add livekit-wakeword                          # detection only: numpy + onnxruntime
# microphone listener: add the [listener] extra (needs portaudio: brew install portaudio / apt install portaudio19-dev)
```

```python
from livekit.wakeword import WakeWordModel

model = WakeWordModel(models=["hey_livekit.onnx"])

# stream audio as it arrives: 16 kHz mono, int16 or float32, any chunk size
scores = model.predict(audio_chunk)
if scores["hey_livekit"] > 0.5:  # 0.5 is calibrated to about 0.5 false alarms per hour
    print("Wake word detected!")
    model.reset()  # start fresh after a detection

model.predict_clip(clip)  # highest score in a whole clip

model.load_model("another.onnx", model_name="custom")  # add or replace models at runtime
model.unload_model("custom")
```

`predict()` returns one score per loaded model, keyed by file name stem. Each 80 ms of audio is one
model step (about 1 ms on one CPU thread); the state carries over between calls.

```python
import asyncio
from livekit.wakeword import WakeWordListener, WakeWordModel


async def main():
    model = WakeWordModel(models=["hey_livekit.onnx"])
    async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
        while True:
            detection = await listener.wait_for_detection()
            print(f"Detected {detection.name} ({detection.confidence:.2f})")


asyncio.run(main())
```

From a clone of this repo: `uv run python client/python/examples/inference.py` (scores a clip) and
`uv run --with pyaudio python client/python/examples/listener.py` (microphone).
