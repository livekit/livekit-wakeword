"""Simple API for wake word detection, similar to openwakeword."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

from livewakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from livewakeword.resources import get_embedding_model_path, get_mel_model_path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class Model:
    """Simple API for wake word detection, similar to openwakeword.

    Example usage:
        from livewakeword import Model

        # Load wake word model(s)
        model = Model(wakeword_models=["path/to/model.onnx"])

        # Get audio frame (16-bit 16kHz PCM, multiples of 80ms recommended)
        frame = get_audio_frame()

        # Get predictions
        prediction = model.predict(frame)
        # Returns: {"model_name": 0.95, ...}
    """

    def __init__(
        self,
        wakeword_models: list[str | Path] | None = None,
        inference_framework: str = "onnx",
    ):
        """Initialize the wake word detection model.

        Args:
            wakeword_models: List of paths to wake word ONNX classifier models.
                If None, no models are loaded (call load_model() later).
            inference_framework: Inference framework to use (only "onnx" supported).
        """
        if inference_framework != "onnx":
            raise ValueError(f"Unsupported inference framework: {inference_framework}")

        # Load bundled feature extraction models
        mel_path = get_mel_model_path()
        embedding_path = get_embedding_model_path()

        if not mel_path.exists():
            raise FileNotFoundError(
                f"Bundled mel model not found: {mel_path}\n"
                "This should not happen - please reinstall livewakeword."
            )
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Bundled embedding model not found: {embedding_path}\n"
                "This should not happen - please reinstall livewakeword."
            )

        self._mel_frontend = MelSpectrogramFrontend(onnx_path=mel_path)
        self._speech_embedding = SpeechEmbedding(onnx_path=embedding_path)

        # Store wake word classifiers: name -> (session, input_name)
        self._classifiers: dict[str, tuple] = {}

        # Audio processing state (shared across all models)
        self._max_audio_seconds = 3.0
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._mel_frame_count = 0
        self._embedding_buffer: deque[np.ndarray] = deque(maxlen=16)
        self._mel_frames_since_embedding = 0
        self._last_scores: dict[str, float] = {}

        # Load provided models
        if wakeword_models:
            for model_path in wakeword_models:
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
        self._last_scores[model_name] = 0.0
        logger.info(f"Loaded wake word model '{model_name}' from {model_path}")

    @property
    def model_names(self) -> list[str]:
        """Return list of loaded model names."""
        return list(self._classifiers.keys())

    def predict(self, audio_frame: np.ndarray) -> dict[str, float]:
        """Get wake word predictions for an audio frame.

        Args:
            audio_frame: Audio samples at 16kHz. Can be:
                - int16 array (will be converted to float32)
                - float32 array (values in [-1, 1])
                For best efficiency, use multiples of 80ms (1280 samples).

        Returns:
            Dictionary mapping model names to prediction scores (0-1).
        """
        if len(self._classifiers) == 0:
            return {}

        # Convert int16 to float32 if needed
        if audio_frame.dtype == np.int16:
            audio_frame = audio_frame.astype(np.float32) / 32768.0

        # Ensure 1D
        audio_frame = audio_frame.flatten()

        # Accumulate audio and compute mel spectrogram
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_frame])

        all_mel = self._mel_frontend(self._audio_buffer)
        if all_mel.ndim == 3:
            all_mel = all_mel[0]

        new_mel_frames = all_mel.shape[0] - self._mel_frame_count
        self._mel_frame_count = all_mel.shape[0]
        self._mel_frames_since_embedding += new_mel_frames

        # Need at least 76 mel frames for one embedding window
        if all_mel.shape[0] < 76:
            return {name: 0.0 for name in self._classifiers}

        # Only extract a new embedding every 8 mel frames
        if self._mel_frames_since_embedding < 8:
            return dict(self._last_scores)

        # Extract embedding from latest 76-frame window
        self._mel_frames_since_embedding = 0
        window = all_mel[-76:]
        embedding = self._speech_embedding(window[np.newaxis, :, :])
        self._embedding_buffer.append(embedding[0])

        # Trim audio buffer to avoid unbounded growth
        max_audio_samples = int(self._max_audio_seconds * SAMPLE_RATE)
        if len(self._audio_buffer) > max_audio_samples:
            self._audio_buffer = self._audio_buffer[-max_audio_samples:]
            trimmed_mel = self._mel_frontend(self._audio_buffer)
            if trimmed_mel.ndim == 3:
                trimmed_mel = trimmed_mel[0]
            self._mel_frame_count = trimmed_mel.shape[0]

        # Need 16 embeddings for classifier
        if len(self._embedding_buffer) < 16:
            return {name: 0.0 for name in self._classifiers}

        # Run all classifiers
        emb_sequence = np.stack(list(self._embedding_buffer), axis=0)
        emb_input = emb_sequence[np.newaxis, :, :].astype(np.float32)

        predictions = {}
        for name, (session, input_name) in self._classifiers.items():
            outputs = session.run(None, {input_name: emb_input})
            score = float(outputs[0][0, 0])
            predictions[name] = score
            self._last_scores[name] = score

        return predictions

    def reset(self) -> None:
        """Reset internal audio buffers. Call when starting a new audio stream."""
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._mel_frame_count = 0
        self._embedding_buffer.clear()
        self._mel_frames_since_embedding = 0
        self._last_scores = {name: 0.0 for name in self._classifiers}
