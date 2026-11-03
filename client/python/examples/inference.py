"""Simple wake word detection example."""

import wave
from pathlib import Path

import numpy as np

from livekit.wakeword import WakeWordModel

HERE = Path(__file__).parent
model = WakeWordModel(models=[HERE.parents[2] / "models" / "hey_livekit.onnx"])

with wave.open(str(HERE.parent / "tests" / "fixtures" / "positive.wav")) as w:
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)  # 16 kHz mono

# The model streams: feed consecutive 80 ms frames, it keeps state between calls.
for start in range(0, len(audio) - 1279, 1280):
    for name, score in model.predict(audio[start : start + 1280]).items():
        print(f"[{start / 16000:.2f}s] {name}: {score:.4f}")
