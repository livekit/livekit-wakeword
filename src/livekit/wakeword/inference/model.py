"""Stateless wake word detection model."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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

        self._mel_frontend = MelSpectrogramFrontend(onnx_path=mel_path)
        self._speech_embedding = SpeechEmbedding(onnx_path=embedding_path)

        # name -> (onnx_session, input_name)
        self._classifiers: dict[str, tuple] = {}

        if models:
            for model_path in models:
                self.load_model(model_path)

    def load_model(self, model_path: str | Path, model_name: str | None = None) -> None:
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
        )
        input_name = session.get_inputs()[0].name
        self._classifiers[model_name] = (session, input_name)
        logger.info(f"Loaded wake word model '{model_name}' from {model_path}")

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

        # Convert int16 to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        audio_chunk = audio_chunk.flatten()

        # Mel spectrogram over the full chunk
        all_mel = self._mel_frontend(audio_chunk)
        if all_mel.ndim == 3:
            all_mel = all_mel[0]

        if all_mel.shape[0] < EMBEDDING_WINDOW:
            return {name: 0.0 for name in self._classifiers}

        # Extract embeddings: 76-frame windows, stride 8
        embeddings = []
        for start in range(0, all_mel.shape[0] - EMBEDDING_WINDOW + 1, EMBEDDING_STRIDE):
            window = all_mel[start : start + EMBEDDING_WINDOW]
            emb = self._speech_embedding(window[np.newaxis, :, :])
            embeddings.append(emb[0])

        if len(embeddings) < MIN_EMBEDDINGS:
            return {name: 0.0 for name in self._classifiers}

        # Use last 16 embeddings
        emb_sequence = np.stack(embeddings[-MIN_EMBEDDINGS:], axis=0)
        emb_input = emb_sequence[np.newaxis, :, :].astype(np.float32)

        predictions = {}
        for name, (session, input_name) in self._classifiers.items():
            outputs = session.run(None, {input_name: emb_input})
            score = float(outputs[0][0, 0])
            predictions[name] = score

        return predictions
