"""Async microphone listener."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import WakeWordModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms
PYAUDIO_HINT = (
    "pip install 'livekit-wakeword[listener]' (or `uv run --with pyaudio ...`); PyAudio needs "
    "PortAudio: brew install portaudio / apt install portaudio19-dev"
)


@dataclass
class Detection:
    name: str
    confidence: float
    timestamp: float


class WakeWordListener:
    """Streams the microphone through ``WakeWordModel.predict()`` in 80 ms frames.

    After a detection the loop pauses (and the model is reset) until ``wait_for_detection()`` is
    called again::

        async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
            while True:
                detection = await listener.wait_for_detection()
    """

    def __init__(self, model: WakeWordModel, threshold: float = 0.5, debounce: float = 2.0):
        self._model = model
        self._threshold = threshold
        self._debounce = debounce
        self._stream: Any = None
        self._pa: Any = None
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_detection_time = 0.0
        self._detection_queue: asyncio.Queue[Detection] = asyncio.Queue()
        self._error: BaseException | None = None
        # one worker thread for mic reads and predict(), off the event loop
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._listening = asyncio.Event()  # cleared while a detection is being handled
        self._done_event = asyncio.Event()  # set when the audio loop exits

    async def __aenter__(self) -> WakeWordListener:
        try:
            import pyaudio
        except ImportError as e:
            raise ImportError(f"WakeWordListener needs PyAudio: {PYAUDIO_HINT}") from e

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )
        self._running = True
        self._listening.set()
        self._done_event.clear()
        self._error = None
        self._detection_queue = asyncio.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._task = asyncio.create_task(self._audio_loop())
        return self

    async def __aexit__(self, *_: object) -> None:
        self._running = False
        self._listening.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        # close the stream only once no thread is reading from it
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    async def _audio_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                await self._listening.wait()
                if not self._running:
                    break
                data = await loop.run_in_executor(
                    self._executor,
                    lambda: self._stream.read(FRAME_SAMPLES, exception_on_overflow=False),
                )
                if not self._running:
                    break
                frame = np.frombuffer(data, dtype=np.int16)
                scores = await loop.run_in_executor(self._executor, self._model.predict, frame)
                if not self._running:
                    break
                now = time.monotonic()
                for name, score in scores.items():
                    if (
                        score >= self._threshold
                        and now - self._last_detection_time >= self._debounce
                    ):
                        self._last_detection_time = now
                        self._listening.clear()
                        self._model.reset()
                        await self._detection_queue.put(
                            Detection(name=name, confidence=score, timestamp=now)
                        )
                        break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Audio loop crashed: %s", exc, exc_info=True)
            self._error = exc
        finally:
            self._done_event.set()

    async def wait_for_detection(self) -> Detection:
        """The next detection (resumes listening). Raises RuntimeError if the audio loop died."""
        self._listening.set()
        queue_waiter = asyncio.ensure_future(self._detection_queue.get())
        done_waiter = asyncio.ensure_future(self._done_event.wait())
        done, pending = await asyncio.wait(
            {queue_waiter, done_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if queue_waiter in done:
            return queue_waiter.result()
        if not self._detection_queue.empty():
            return self._detection_queue.get_nowait()
        if self._error is not None:
            raise RuntimeError(f"Audio loop crashed: {self._error}") from self._error
        raise RuntimeError("Audio loop ended unexpectedly")
