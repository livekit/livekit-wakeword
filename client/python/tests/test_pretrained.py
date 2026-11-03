"""The shipped hey_livekit model on the reference clips shared with the Swift tests."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from onnxruntime import SessionOptions

from livekit.wakeword import WakeWordModel
from livekit.wakeword.model import DISABLE_KLEIDIAI, session_options

MODEL = Path(__file__).resolve().parents[3] / "models" / "hey_livekit.onnx"
FIXTURES = Path(__file__).parent / "fixtures"
THRESHOLD = 0.5


def clip(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name)) as w:
        assert w.getframerate() == 16_000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


@pytest.fixture(scope="module")
def model() -> WakeWordModel:
    return WakeWordModel(models=[MODEL])


def test_positive_clip_detected(model: WakeWordModel) -> None:
    assert model.predict_clip(clip("positive.wav"))["hey_livekit"] >= THRESHOLD


def test_negative_clip_rejected(model: WakeWordModel) -> None:
    assert model.predict_clip(clip("negative.wav"))["hey_livekit"] < THRESHOLD


def test_caller_options_are_kept() -> None:
    options = SessionOptions()
    options.intra_op_num_threads = 1
    model = WakeWordModel(models=[MODEL], sess_options=options)
    assert options.intra_op_num_threads == 1
    assert model.predict_clip(clip("positive.wav"))["hey_livekit"] >= THRESHOLD


def test_caller_can_opt_back_in() -> None:
    options = SessionOptions()
    options.add_session_config_entry(DISABLE_KLEIDIAI, "0")
    assert session_options(options).get_session_config_entry(DISABLE_KLEIDIAI) == "0"
