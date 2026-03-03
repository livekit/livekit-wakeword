"""Tests for WakeWordListener async behavior."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from livekit.wakeword.inference.listener import (
    Detection,
    WakeWordListener,
    FRAME_SAMPLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeModel:
    """Mock WakeWordModel that returns pre-configured scores."""

    def __init__(self, scores_sequence: list[dict[str, float]] | None = None):
        self._scores_sequence = scores_sequence or []
        self._call_count = 0
        self.reset_count = 0

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        if self._call_count < len(self._scores_sequence):
            scores = self._scores_sequence[self._call_count]
        else:
            scores = {"test": 0.0}
        self._call_count += 1
        return scores

    def reset(self) -> None:
        self.reset_count += 1


class FakeStream:
    """Mock PyAudio stream that returns silence."""

    def __init__(self, *, error_after: int | None = None):
        self._error_after = error_after
        self._read_count = 0

    def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
        self._read_count += 1
        if self._error_after is not None and self._read_count > self._error_after:
            raise IOError("Simulated stream read error")
        return np.zeros(num_frames, dtype=np.int16).tobytes()

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakePyAudio:
    """Mock PyAudio that returns a FakeStream."""

    def __init__(self, stream: FakeStream):
        self._stream = stream

    def open(self, **kwargs: object) -> FakeStream:
        return self._stream

    def terminate(self) -> None:
        pass


def _patch_pyaudio(stream: FakeStream):
    """Return a patch context that replaces pyaudio with a fake."""
    fake_pa = FakePyAudio(stream)
    mock_module = MagicMock()
    mock_module.PyAudio.return_value = fake_pa
    mock_module.paInt16 = 8  # pyaudio constant
    return patch.dict("sys.modules", {"pyaudio": mock_module})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_runs_in_executor():
    """predict() should execute on a non-main thread (in the executor)."""
    predict_threads: list[threading.Thread] = []

    class ThreadTrackingModel(FakeModel):
        def predict(self, frame: np.ndarray) -> dict[str, float]:
            predict_threads.append(threading.current_thread())
            return super().predict(frame)

    model = ThreadTrackingModel(scores_sequence=[{"test": 0.0}] * 50)
    stream = FakeStream()

    with _patch_pyaudio(stream):
        async with WakeWordListener(model, threshold=0.5) as listener:
            # Let a few iterations run
            await asyncio.sleep(0.3)

    assert len(predict_threads) > 0
    main_thread = threading.main_thread()
    for t in predict_threads:
        assert t is not main_thread, "predict() ran on the main thread"


@pytest.mark.asyncio
async def test_error_propagation():
    """If the audio loop crashes, wait_for_detection raises RuntimeError."""

    class ErrorModel:
        def predict(self, frame: np.ndarray) -> dict[str, float]:
            raise ValueError("ONNX inference failed")

        def reset(self) -> None:
            pass

    model = ErrorModel()
    stream = FakeStream()

    with _patch_pyaudio(stream):
        async with WakeWordListener(model, threshold=0.5) as listener:
            with pytest.raises(RuntimeError, match="Audio loop crashed"):
                await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)


@pytest.mark.asyncio
async def test_stream_error_propagation():
    """If the audio stream crashes, wait_for_detection raises RuntimeError."""
    model = FakeModel(scores_sequence=[{"test": 0.0}] * 100)
    stream = FakeStream(error_after=3)

    with _patch_pyaudio(stream):
        async with WakeWordListener(model, threshold=0.5) as listener:
            with pytest.raises(RuntimeError, match="Audio loop crashed"):
                await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)


@pytest.mark.asyncio
async def test_buffer_reset_after_detection():
    """model.reset() is called after each detection."""
    scores = [{"test": 0.0}] * 5 + [{"test": 0.9}]
    model = FakeModel(scores_sequence=scores)
    stream = FakeStream()

    with _patch_pyaudio(stream):
        async with WakeWordListener(model, threshold=0.5, debounce=0.0) as listener:
            detection = await asyncio.wait_for(
                listener.wait_for_detection(), timeout=5.0
            )
            assert detection.name == "test"
            assert detection.confidence == pytest.approx(0.9)
            assert model.reset_count == 1


@pytest.mark.asyncio
async def test_loop_pauses_after_detection():
    """After detection, the loop pauses — no stale detections pile up."""
    # Provide continuous high scores
    scores = [{"test": 0.9}] * 200
    model = FakeModel(scores_sequence=scores)
    stream = FakeStream()

    with _patch_pyaudio(stream):
        async with WakeWordListener(model, threshold=0.5, debounce=0.0) as listener:
            det1 = await asyncio.wait_for(
                listener.wait_for_detection(), timeout=5.0
            )

            # Give time for the loop to potentially queue more (it shouldn't)
            await asyncio.sleep(0.3)
            assert listener._detection_queue.empty(), (
                "Loop should be paused — no extra detections in queue"
            )

            # Second call resumes listening and gets next detection
            det2 = await asyncio.wait_for(
                listener.wait_for_detection(), timeout=5.0
            )
            assert det2.timestamp > det1.timestamp


@pytest.mark.asyncio
async def test_shutdown_while_paused():
    """Exiting context manager doesn't hang when loop is paused after detection."""
    scores = [{"test": 0.0}] * 3 + [{"test": 0.9}]
    model = FakeModel(scores_sequence=scores)
    stream = FakeStream()

    with _patch_pyaudio(stream):
        async with WakeWordListener(model, threshold=0.5) as listener:
            await asyncio.wait_for(listener.wait_for_detection(), timeout=5.0)
            # Loop is now paused. Exiting context manager should not hang.
    # If we reach here, shutdown succeeded.


@pytest.mark.asyncio
async def test_shutdown_during_active_listening():
    """Exiting context manager completes cleanly during active listening."""
    model = FakeModel(scores_sequence=[{"test": 0.0}] * 1000)
    stream = FakeStream()

    with _patch_pyaudio(stream):
        async with WakeWordListener(model, threshold=0.5) as listener:
            await asyncio.sleep(0.2)  # let loop run a few iterations
    # If we reach here without hanging, shutdown is clean.
