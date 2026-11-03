"""Streaming wake word inference: one stateful ONNX file per wake word, 80 ms per step."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from onnxruntime import InferenceSession, SessionOptions

from ._resample import MODEL_RATE, Resampler

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 1280  # 80 ms at 16 kHz
WARMUP_FRAMES = 25  # 2 s of silence: the state every stream starts from
# onnxruntime 1.27.0 on arm64 runs batch-1 convolutions through KleidiAI kernels that return wrong
# results; older versions accept the entry and ignore it
DISABLE_KLEIDIAI = "mlas.disable_kleidiai"


def session_options(options: SessionOptions | None = None) -> SessionOptions:
    """`options` (or defaults) with KleidiAI off, unless the caller set that entry."""
    options = options if options is not None else SessionOptions()
    try:
        options.get_session_config_entry(DISABLE_KLEIDIAI)
    except RuntimeError:
        options.add_session_config_entry(DISABLE_KLEIDIAI, "1")
    return options


class _Stream:
    """One model's state: inputs `chunk` (1, 1280) + state tensors, outputs `score` + `<state>_out`.
    The state starts at zero and is warmed up on silence, as in training."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._names = [o.name.removesuffix("_out") for o in session.get_outputs()[1:]]
        self._state = {
            i.name: np.zeros(i.shape, np.int64 if "int64" in i.type else np.float32)
            for i in session.get_inputs()
            if i.name != "chunk"
        }
        for _ in range(WARMUP_FRAMES):
            self._step(np.zeros((1, FRAME_SAMPLES), np.float32))
        self._warm = self._state
        self._pending = np.zeros(0, np.float32)

    def _step(self, chunk: np.ndarray) -> float:
        out = self._session.run(None, {"chunk": chunk, **self._state})
        self._state = dict(zip(self._names, out[1:]))
        return float(np.asarray(out[0]).reshape(-1)[0])

    def reset(self) -> None:
        self._state = self._warm
        self._pending = np.zeros(0, np.float32)

    def feed(self, audio: np.ndarray) -> float:
        """The highest score among the 80 ms steps `audio` completes (0.0 if none)."""
        self._pending = np.concatenate([self._pending, audio])
        best = 0.0
        while len(self._pending) >= FRAME_SAMPLES:
            best = max(best, self._step(self._pending[None, :FRAME_SAMPLES]))
            self._pending = self._pending[FRAME_SAMPLES:]
        return best


class WakeWordModel:
    """Wake word models (`.onnx` from `livekit-wakeword train`), streamed over mono audio::

    model = WakeWordModel(models=["hey_livekit.onnx"])
    scores = model.predict(frame)  # {"hey_livekit": 0.97}; frames of any length

    `sample_rate` is the rate of the audio passed to `predict()`; models run at 16 kHz, and audio
    at any other rate is resampled internally.
    """

    def __init__(
        self,
        models: list[str | Path] | None = None,
        sess_options: SessionOptions | None = None,
        sample_rate: int = MODEL_RATE,
    ):
        self.sample_rate = int(sample_rate)
        self._resampler = Resampler(self.sample_rate) if self.sample_rate != MODEL_RATE else None
        self._sess_options = sess_options
        self._models: dict[str, _Stream] = {}
        for path in models or []:
            self.load_model(path)

    def load_model(
        self,
        model_path: str | Path,
        model_name: str | None = None,
        sess_options: SessionOptions | None = None,
    ) -> None:
        """Add a model, named after the file unless `model_name` is given."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Wake word model not found: {model_path}")
        if model_path.suffix == ".tflite":
            raise ValueError(
                f"{model_path} is a micro model (int8 TFLite for ESP32 / TensorFlow Lite Micro); "
                "load the standard .onnx model here"
            )
        try:
            session = InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
                sess_options=session_options(sess_options or self._sess_options),
            )
        except Exception as e:  # onnxruntime's InvalidProtobuf, Fail, ... (not RuntimeError)
            raise ValueError(f"{model_path} is not a valid ONNX model: {e}") from e
        if "chunk" not in {i.name for i in session.get_inputs()}:
            raise ValueError(
                f"{model_path} is not a streaming wake word model; v1 classifier models are no "
                "longer supported, retrain with `livekit-wakeword train`"
            )
        name = model_name or model_path.stem
        self._models[name] = _Stream(session)
        logger.info("Loaded wake word model '%s' from %s", name, model_path)

    def unload_model(self, model_name: str) -> bool:
        """Remove a model; returns whether it was loaded."""
        return self._models.pop(model_name, None) is not None

    @property
    def model_names(self) -> list[str]:
        return list(self._models)

    def reset(self) -> None:
        """Forget all audio heard so far (e.g. after a detection)."""
        if self._resampler is not None:
            self._resampler.reset()
        for m in self._models.values():
            m.reset()

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        """Feed the next audio (mono at `sample_rate`, int16 or float in [-1, 1], any length).
        Returns each model's highest score among the 80 ms steps this call completed (0.0 if
        none)."""
        if not self._models:
            return {}
        x = _as_float(audio)
        if self._resampler is not None:
            x = self._resampler(x)
        return {name: m.feed(x) for name, m in self._models.items()}

    def predict_clip(self, audio: np.ndarray) -> dict[str, float]:
        """Each model's highest score over a whole clip, starting fresh."""
        self.reset()
        return self.predict(audio)


def _as_float(audio: np.ndarray) -> np.ndarray:
    """Mono int16 or float PCM -> float32 in [-1, 1]."""
    audio = np.asarray(audio)
    if audio.ndim > 1 and sum(n > 1 for n in audio.shape) <= 1:
        audio = audio.reshape(-1)  # (n, 1) or (1, n)
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio (one channel), got shape {audio.shape}")
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)
    raise TypeError(f"expected int16 or float PCM, got {audio.dtype}")
