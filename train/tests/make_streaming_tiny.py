"""Regenerate `streaming_tiny.onnx` (tiny random-weight streaming model) and
`streaming_tiny_expected.json` (the Python runtime's per-80 ms scores of the 3 s chirp the Swift
test generates).

    uv run python train/tests/make_streaming_tiny.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from livekit.wakeword import WakeWordModel
from livekit.wakeword_train.export import export_streaming
from livekit.wakeword_train.nn import Detector, Encoder, Head

OUT = Path(__file__).resolve().parents[2] / "client/python/tests/fixtures/streaming_tiny.onnx"


def chirp() -> np.ndarray:
    t = np.arange(16_000 * 3) / 16_000.0
    return np.trunc(np.sin(2.0 * np.pi * (150.0 + 400.0 * t) * t) * 9000.0).astype(np.int16)


def main() -> None:
    torch.manual_seed(0)
    enc = Encoder(dim=32, d=64, layers=2, heads=2, ff=2, k=7, left=16, sub_c=16)
    det = Detector(enc, Head(32, 32, 1, n_heads=2)).eval()
    export_streaming(det, OUT, metadata={"wake_word": "streaming tiny"})
    pcm = chirp()
    model = WakeWordModel(models=[OUT])
    scores = [
        model.predict(pcm[i : i + 1280])["streaming_tiny"] for i in range(0, len(pcm) - 1279, 1280)
    ]
    OUT.with_name("streaming_tiny_expected.json").write_text(
        json.dumps({"sample_rate": 16000, "scores": scores}) + "\n"
    )
    print(f"{OUT} ({OUT.stat().st_size / 1e3:.0f} kB), {len(scores)} expected scores")


if __name__ == "__main__":
    main()
