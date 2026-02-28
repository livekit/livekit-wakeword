"""Async wake word listener with audio capture."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from livewakeword.inference.model import Model

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80ms


@dataclass
class Detection:
    """Wake word detection result."""

    name: str
    confidence: float
    timestamp: float


class Listener:
    """Async wake word listener that handles audio capture.

    Example:
        from livewakeword import Model, Listener

        model = Model(wakeword_models=["hey_livekit.onnx"])

        async with Listener(model, threshold=0.5, debounce=2.0) as listener:
            while True:
                detection = await listener.wait_for_detection()
                print(f"Detected {detection.name}! (confidence={detection.confidence:.2f})")
    """

    def __init__(
        self,
        model: Model,
        threshold: float = 0.5,
        debounce: float = 2.0,
    ):
        """Initialize listener.

        Args:
            model: Wake word Model instance with loaded classifiers.
            threshold: Detection threshold (0-1).
            debounce: Minimum seconds between detections.
        """
        self._model = model
        self._threshold = threshold
        self._debounce = debounce

        self._stream = None
        self._pa = None
        self._running = False
        self._last_detection_time = 0.0
        self._detection_queue: asyncio.Queue[Detection] = asyncio.Queue()

    async def __aenter__(self) -> "Listener":
        """Start audio capture."""
        import pyaudio

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )
        self._running = True
        self._task = asyncio.create_task(self._audio_loop())
        return self

    async def __aexit__(self, *_) -> None:
        """Stop audio capture."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    async def _audio_loop(self) -> None:
        """Background task that captures audio and runs detection."""
        loop = asyncio.get_event_loop()

        while self._running:
            # Read audio in executor to not block
            data = await loop.run_in_executor(
                None,
                lambda: self._stream.read(FRAME_SAMPLES, exception_on_overflow=False),
            )
            frame = np.frombuffer(data, dtype=np.int16)

            # Run inference
            scores = self._model.predict(frame)

            # Check for detections
            now = time.monotonic()
            for name, score in scores.items():
                if score >= self._threshold:
                    if now - self._last_detection_time >= self._debounce:
                        self._last_detection_time = now
                        await self._detection_queue.put(
                            Detection(name=name, confidence=score, timestamp=now)
                        )

    async def wait_for_detection(self) -> Detection:
        """Wait for and return the next wake word detection."""
        return await self._detection_queue.get()
