"""Reference outputs from the Python client, for the JS parity tests (test/fixtures/parity.json).

Run from the repo root: uv run client/js/scripts/make_parity.py
"""

import json
import wave
from pathlib import Path

import numpy as np

from livekit.wakeword import WakeWordModel
from livekit.wakeword._resample import Resampler

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "client/python/tests/fixtures"


def chirp(rate: int, n: int) -> np.ndarray:
    """The same signal as `chirp()` in test/resample.test.ts."""
    t = np.arange(n, dtype=np.float64) / rate
    return (0.3 * np.sin(2 * np.pi * (200 + 1500 * t) * t)).astype(np.float32)


def step_scores(name: str) -> list[float]:
    with wave.open(str(FIXTURES / f"{name}.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    model = WakeWordModel([ROOT / "models/hey_livekit.onnx"])
    return [model.predict(audio[i : i + 1280])["hey_livekit"] for i in range(0, len(audio), 1280)]


out = {
    "scores": {name: step_scores(name) for name in ("positive", "negative")},
    "resample": {
        str(rate): Resampler(rate)(chirp(rate, rate // 4)).tolist() for rate in (44100, 48000)
    },
}
path = ROOT / "client/js/test/fixtures/parity.json"
path.write_text(json.dumps(out) + "\n")
print(f"wrote {path}")
