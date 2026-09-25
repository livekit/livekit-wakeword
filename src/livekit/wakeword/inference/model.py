"""Wake word detection model with optional per-stream caching."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from onnxruntime.capi.onnxruntime_pybind11_state import SessionOptions

from ..models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from ..resources import get_embedding_model_path, get_mel_model_path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
EMBEDDING_WINDOW = 76  # mel frames per embedding
EMBEDDING_STRIDE = 8  # mel frames between embeddings
MIN_EMBEDDINGS = 16  # classifier input length


class WakeWordModel:
    """Stateless wake word detection model.

    The model is a pure function: pass an audio chunk (~2 seconds at 16 kHz)
    and receive confidence scores.  No internal audio state is maintained.

    Example usage:
        from livekit.wakeword import WakeWordModel

        model = WakeWordModel(models=["path/to/model.onnx"])

        # Pass ~2 seconds of 16 kHz audio
        scores = model.predict(audio_chunk)
        # Returns: {"model_name": 0.95, ...}
    """

    def __init__(
        self,
        models: list[str | Path] | None = None,
        sess_options: SessionOptions | None = None,
    ):
        """Initialize the wake word detection model.

        Args:
            models: List of paths to wake word ONNX classifier models.
                If None, no models are loaded (call load_model() later).
        """
        mel_path = get_mel_model_path()
        embedding_path = get_embedding_model_path()

        if not mel_path.exists():
            raise FileNotFoundError(
                f"Bundled mel model not found: {mel_path}\n"
                "This should not happen - please reinstall livekit-wakeword."
            )
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Bundled embedding model not found: {embedding_path}\n"
                "This should not happen - please reinstall livekit-wakeword."
            )

        self._mel_frontend = MelSpectrogramFrontend(onnx_path=mel_path, sess_options=sess_options)
        self._speech_embedding = SpeechEmbedding(
            onnx_path=embedding_path,
            sess_options=sess_options,
        )

        # name -> (onnx_session, input_name)
        self._classifiers: dict[str, tuple] = {}

        if models:
            for model_path in models:
                self.load_model(model_path, sess_options=sess_options)

    def load_model(
        self,
        model_path: str | Path,
        model_name: str | None = None,
        sess_options: SessionOptions | None = None,
    ) -> None:
        """Load a wake word classifier model.

        Args:
            model_path: Path to the ONNX wake word classifier.
            model_name: Optional name for the model. If None, derived from filename.
        """
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Wake word model not found: {model_path}")

        if model_name is None:
            model_name = model_path.stem

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )
        input_name = session.get_inputs()[0].name
        self._classifiers[model_name] = (session, input_name)
        logger.info(f"Loaded wake word model '{model_name}' from {model_path}")

    def create_stream(self) -> WakeWordStream:
        """Create an isolated predictor for overlapping audio windows.

        The returned stream caches speech embeddings. The model itself remains
        stateless, and each stream owns independent cache state.
        """
        return WakeWordStream(self)

    def predict(self, audio_chunk: np.ndarray) -> dict[str, float]:
        """Get wake word predictions for an audio chunk.

        The model is stateless — pass a complete audio window each time.
        ~2 seconds of 16 kHz audio is recommended (yields exactly 16
        embeddings for the classifier).  Shorter chunks that lack enough
        data return zero scores.

        Args:
            audio_chunk: Audio samples at 16 kHz. Can be int16 or float32.

        Returns:
            Dictionary mapping model names to prediction scores (0-1).
        """
        if not self._classifiers:
            return {}

        windows = self._mel_windows(audio_chunk)
        if windows is None:
            return self._zero_scores()

        embeddings = np.stack(
            [self._speech_embedding(window[np.newaxis, :, :])[0] for window in windows],
            axis=0,
        )
        return self._classify(embeddings)

    def _mel_windows(self, audio_chunk: np.ndarray) -> np.ndarray | None:
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        all_mel = self._mel_frontend(audio_chunk.flatten())
        if all_mel.ndim == 3:
            all_mel = all_mel[0]

        starts = range(
            0,
            all_mel.shape[0] - EMBEDDING_WINDOW + 1,
            EMBEDDING_STRIDE,
        )
        windows = [all_mel[start : start + EMBEDDING_WINDOW] for start in starts]
        if len(windows) < MIN_EMBEDDINGS:
            return None
        return np.stack(windows[-MIN_EMBEDDINGS:], axis=0)

    def _classify(self, embeddings: np.ndarray) -> dict[str, float]:
        emb_input = embeddings[np.newaxis, :, :].astype(np.float32)
        predictions: dict[str, float] = {}
        for name, (session, input_name) in self._classifiers.items():
            outputs = session.run(None, {input_name: emb_input})
            predictions[name] = float(outputs[0][0, 0])
        return predictions

    def _zero_scores(self) -> dict[str, float]:
        return {name: 0.0 for name in self._classifiers}


class WakeWordStream:
    """Prediction state for a sequence of overlapping full audio windows.

    Each call still computes the full mel spectrogram because that model's
    output can depend on the complete audio window. Speech embeddings are reused
    only when their 76-frame mel inputs exactly match the shifted prior inputs.
    """

    def __init__(self, model: WakeWordModel):
        self._model = model
        self._mel_windows: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None

    def reset(self) -> None:
        """Clear cached windows and embeddings."""
        self._mel_windows = None
        self._embeddings = None

    def predict(self, audio_chunk: np.ndarray) -> dict[str, float]:
        """Predict from the next complete audio window in the stream."""
        if not self._model._classifiers:
            return {}

        windows = self._model._mel_windows(audio_chunk)
        if windows is None:
            self.reset()
            return self._model._zero_scores()

        embeddings = self._embedding_sequence(windows)
        self._mel_windows = windows
        self._embeddings = embeddings
        return self._model._classify(embeddings)

    def _embedding_sequence(self, windows: np.ndarray) -> np.ndarray:
        changed_indexes: list[int] = []
        changed_windows: list[np.ndarray] = []
        reused: dict[int, np.ndarray] = {}

        for index, window in enumerate(windows):
            previous_index = index + 1
            if (
                self._mel_windows is not None
                and self._embeddings is not None
                and previous_index < len(self._mel_windows)
                and np.array_equal(window, self._mel_windows[previous_index])
            ):
                reused[index] = self._embeddings[previous_index]
            else:
                changed_indexes.append(index)
                changed_windows.append(window)

        fresh = self._model._speech_embedding(np.stack(changed_windows, axis=0))
        embeddings = np.empty(
            (len(windows), fresh.shape[1]),
            dtype=fresh.dtype,
        )
        for index, embedding in reused.items():
            embeddings[index] = embedding
        for index, embedding in zip(changed_indexes, fresh, strict=True):
            embeddings[index] = embedding
        return embeddings
