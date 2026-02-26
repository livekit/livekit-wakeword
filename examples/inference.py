"""Simple wake word detection example."""

from pathlib import Path

import numpy as np

from livewakeword import Model

# Load model
model = Model(wakeword_models=[Path(__file__).parent / "resources" / "hey_livekit.onnx"])

# Simulate 3 seconds of random audio (16kHz) - need ~2s to fill embedding buffer
audio = np.random.randint(-32768, 32767, size=48000, dtype=np.int16)

# Process in 80ms chunks (1280 samples)
for i in range(0, len(audio), 1280):
    frame = audio[i : i + 1280]
    scores = model.predict(frame)

    for name, score in scores.items():
        print(f"{name}: {score:.4f}")