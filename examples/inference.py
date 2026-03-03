"""Simple wake word detection example."""

from pathlib import Path

import numpy as np

from livekit.wakeword import WakeWordModel

# Load model
model = WakeWordModel(models=[Path(__file__).parent / "resources" / "hey_livekit.onnx"])

# Simulate 3 seconds of random audio (16kHz)
audio = np.random.randint(-32768, 32767, size=48000, dtype=np.int16)

# The model is stateless — pass a ~2-second chunk and get scores back.
# Slide a 2-second window over the audio with a 320ms stride.
CHUNK = 32000  # 2 seconds at 16 kHz
STRIDE = 1280 * 4  # 320ms

for start in range(0, len(audio) - CHUNK + 1, STRIDE):
    chunk = audio[start : start + CHUNK]
    scores = model.predict(chunk)

    for name, score in scores.items():
        print(f"[{start / 16000:.2f}s] {name}: {score:.4f}")
