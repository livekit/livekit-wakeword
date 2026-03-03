"""Async wake word listener with audio capture."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .model import WakeWordModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80ms per frame
CHUNK_SECONDS = 2.0
# Number of frames that fill a ~2-second chunk (25 × 80ms = 2000ms)
CHUNK_FRAMES = int(CHUNK_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)


@dataclass
class Detection:
    """Wake word detection result."""

    name: str
    confidence: float
    timestamp: float


class WakeWordListener:
    """Async wake word listener that handles audio capture.

    The listener owns the audio buffer and passes fixed ~2-second chunks
    to the stateless ``WakeWordModel.predict()``.  After a detection the
    loop pauses automatically and resumes when the consumer calls
    ``wait_for_detection()`` again.

    Example:
        from livekit.wakeword import WakeWordModel, WakeWordListener

        model = WakeWordModel(models=["hey_livekit.onnx"])

        async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
            while True:
                detection = await listener.wait_for_detection()
                print(f"Detected {detection.name}! (confidence={detection.confidence:.2f})")
    """

    def __init__(
        self,
        model: WakeWordModel,
        threshold: float = 0.5,
        debounce: float = 2.0,
    ):
        """Initialize listener.

        Args:
            model: WakeWordModel instance with loaded classifiers.
            threshold: Detection threshold (0-1).
            debounce: Minimum seconds between detections.
        """
        self._model = model
        self._threshold = threshold
        self._debounce = debounce

        self._stream = None
        self._pa = None
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_detection_time = 0.0
        self._detection_queue: asyncio.Queue[Detection] = asyncio.Queue()

        # Error propagation: stored exception from _audio_loop crash
        self._error: BaseException | None = None

        # Single-thread executor keeps predict() off the event loop
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Pause/resume control: cleared after detection, set when consumer
        # calls wait_for_detection() again.
        self._listening = asyncio.Event()

        # Signals when _audio_loop exits (success or crash) so
        # wait_for_detection() can raise instead of hanging forever.
        self._done_event = asyncio.Event()

        # Sliding window of recent audio frames (listener owns the buffer)
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=CHUNK_FRAMES)

    async def __aenter__(self) -> WakeWordListener:
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
        self._listening.set()
        self._done_event.clear()
        self._error = None
        self._frame_buffer.clear()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._task = asyncio.create_task(self._audio_loop())
        return self

    async def __aexit__(self, *_: object) -> None:
        """Stop audio capture with safe shutdown sequence."""
        # 1. Signal the loop to stop
        self._running = False
        # Unblock if paused so the loop can see _running=False
        self._listening.set()

        # 2. Let the loop exit naturally (worst case: one 80ms read finishes)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        # 3. Wait for any in-flight executor work to complete
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # 4. Now safe to close stream — no thread is reading from it
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    async def _audio_loop(self) -> None:
        """Background task that captures audio and runs detection."""
        loop = asyncio.get_event_loop()

        try:
            while self._running:
                # Block here when paused (after detection, until consumer resumes)
                await self._listening.wait()
                if not self._running:
                    break

                # Read audio in executor to not block event loop
                data = await loop.run_in_executor(
                    self._executor,
                    lambda: self._stream.read(  # type: ignore[union-attr]
                        FRAME_SAMPLES, exception_on_overflow=False
                    ),
                )
                if not self._running:
                    break

                frame = np.frombuffer(data, dtype=np.int16)
                self._frame_buffer.append(frame)

                # Wait until the buffer has enough audio for the model
                if len(self._frame_buffer) < CHUNK_FRAMES:
                    continue

                # Build the audio chunk and run inference in executor
                chunk = np.concatenate(list(self._frame_buffer))
                scores = await loop.run_in_executor(
                    self._executor,
                    self._model.predict,
                    chunk,
                )
                if not self._running:
                    break

                # Check for detections (lightweight, fine on event loop)
                now = time.monotonic()
                for name, score in scores.items():
                    if score >= self._threshold:
                        if now - self._last_detection_time >= self._debounce:
                            self._last_detection_time = now

                            # Pause the loop and clear the buffer so no stale
                            # audio is processed while the consumer handles
                            # the detection.
                            self._listening.clear()
                            self._frame_buffer.clear()

                            await self._detection_queue.put(
                                Detection(
                                    name=name, confidence=score, timestamp=now
                                )
                            )
                            break  # one detection per iteration
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Audio loop crashed: %s", exc, exc_info=True)
            self._error = exc
        finally:
            self._done_event.set()

    async def wait_for_detection(self) -> Detection:
        """Wait for and return the next wake word detection.

        Resumes the audio loop if it was paused after a previous detection.

        Raises:
            RuntimeError: If the background audio loop has crashed.
        """
        # Resume listening (no-op if already active)
        self._listening.set()

        # Race: either we get a detection, or the audio loop dies
        queue_waiter = asyncio.ensure_future(self._detection_queue.get())
        done_waiter = asyncio.ensure_future(self._done_event.wait())

        done, pending = await asyncio.wait(
            {queue_waiter, done_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if queue_waiter in done:
            return queue_waiter.result()

        # Loop ended — check if there's still a queued detection
        if not self._detection_queue.empty():
            return self._detection_queue.get_nowait()

        if self._error is not None:
            raise RuntimeError(
                f"Audio loop crashed: {self._error}"
            ) from self._error

        raise RuntimeError("Audio loop ended unexpectedly")
