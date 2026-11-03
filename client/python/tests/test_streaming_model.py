"""Streaming models (single ONNX file with explicit state) through WakeWordModel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword import WakeWordModel

FIXTURE = Path(__file__).parent / "fixtures" / "streaming_tiny.onnx"


@pytest.fixture
def audio() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)


def test_model_names() -> None:
    model = WakeWordModel(models=[FIXTURE])
    assert model.model_names == ["streaming_tiny"]


def test_chunking_does_not_change_scores(audio: np.ndarray) -> None:
    """Feeding 80 ms frames or uneven pieces must give identical per-step scores."""
    a = WakeWordModel(models=[FIXTURE])
    frame_scores = [
        a.predict(audio[i : i + 1280])["streaming_tiny"] for i in range(0, len(audio), 1280)
    ]

    b = WakeWordModel(models=[FIXTURE])
    uneven: list[float] = []
    pos = 0
    for size in [700, 1860, 1280, 333, 2227] * 20:
        if pos >= len(audio):
            break
        before = b._models["streaming_tiny"]._pending.size
        steps = (before + len(audio[pos : pos + size])) // 1280
        s = b.predict(audio[pos : pos + size])["streaming_tiny"]
        if steps:
            uneven.append(s)
        pos += size
    # each uneven call returns the max over the steps it completed; the overall max must agree
    assert max(uneven) == pytest.approx(max(frame_scores), abs=1e-6)
    assert all(0.0 <= s <= 1.0 for s in frame_scores)


def test_state_is_carried_and_reset(audio: np.ndarray) -> None:
    model = WakeWordModel(models=[FIXTURE])
    first = [
        model.predict(audio[i : i + 1280])["streaming_tiny"] for i in range(0, 1280 * 20, 1280)
    ]
    second = [
        model.predict(audio[i : i + 1280])["streaming_tiny"] for i in range(0, 1280 * 20, 1280)
    ]
    assert first != second  # the model remembers what it heard
    model.reset()
    again = [
        model.predict(audio[i : i + 1280])["streaming_tiny"] for i in range(0, 1280 * 20, 1280)
    ]
    np.testing.assert_allclose(first, again, atol=1e-6)


def test_int16_input(audio: np.ndarray) -> None:
    model = WakeWordModel(models=[FIXTURE])
    pcm = (audio * 32767).astype(np.int16)
    s = model.predict(pcm)["streaming_tiny"]
    assert 0.0 <= s <= 1.0


def test_input_shapes_and_types(audio: np.ndarray, tmp_path: Path) -> None:
    model = WakeWordModel(models=[FIXTURE])
    mono = model.predict_clip(audio)["streaming_tiny"]
    assert model.predict_clip(audio[:, None])["streaming_tiny"] == pytest.approx(mono)  # (n, 1)
    assert model.predict_clip(audio.astype(np.float64))["streaming_tiny"] == pytest.approx(mono)
    with pytest.raises(ValueError, match="mono"):
        model.predict(np.stack([audio, audio], axis=1))  # stereo
    with pytest.raises(TypeError, match="int16 or float"):
        model.predict((audio * 2**31).astype(np.int32))
    (tmp_path / "m.tflite").write_bytes(b"TFL3")
    with pytest.raises(ValueError, match="micro model"):
        WakeWordModel(models=[tmp_path / "m.tflite"])
    (tmp_path / "bad.onnx").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="not a valid ONNX model"):
        WakeWordModel(models=[tmp_path / "bad.onnx"])


def test_predict_matches_fresh_stream(audio: np.ndarray) -> None:
    model = WakeWordModel(models=[FIXTURE])
    live = WakeWordModel(models=[FIXTURE])
    expected = max(
        live.predict(audio[i : i + 1280])["streaming_tiny"] for i in range(0, len(audio), 1280)
    )
    assert model.predict_clip(audio)["streaming_tiny"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.asyncio
async def test_listener_streams_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """The listener feeds 80 ms frames to predict() for streaming models (no 2 s re-buffering)."""
    import sys
    import types

    from livekit.wakeword import WakeWordListener

    calls: list[int] = []

    class Model(WakeWordModel):
        def predict(self, audio: np.ndarray) -> dict[str, float]:
            calls.append(len(audio))
            return {"streaming_tiny": 0.0}

    class Stream:
        def read(self, n: int, exception_on_overflow: bool = False) -> bytes:
            return np.zeros(n, np.int16).tobytes()

        def stop_stream(self) -> None: ...

        def close(self) -> None: ...

    class PyAudio:
        def open(self, **_: object) -> Stream:
            return Stream()

        def terminate(self) -> None: ...

    fake = types.SimpleNamespace(paInt16=8, PyAudio=PyAudio)
    monkeypatch.setitem(sys.modules, "pyaudio", fake)
    import asyncio

    async with WakeWordListener(Model(models=[FIXTURE])):
        await asyncio.sleep(0.3)
    assert calls and set(calls) == {1280}
