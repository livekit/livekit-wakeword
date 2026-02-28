"""Bundled ONNX models for feature extraction.

Contains:
- melspectrogram.onnx: Mel spectrogram frontend
- embedding_model.onnx: Google speech embedding CNN
"""

from importlib import resources
from pathlib import Path


def get_resource_path(filename: str) -> Path:
    """Get the path to a bundled resource file."""
    # Python 3.9+ with importlib.resources.files
    return Path(str(resources.files(__package__) / filename))


def get_mel_model_path() -> Path:
    """Get path to bundled melspectrogram.onnx model."""
    return get_resource_path("melspectrogram.onnx")


def get_embedding_model_path() -> Path:
    """Get path to bundled embedding_model.onnx model."""
    return get_resource_path("embedding_model.onnx")
