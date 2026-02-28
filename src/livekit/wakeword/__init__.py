"""livekit-wakeword — Simplified pure-PyTorch wake word detection."""

from .config import WakeWordConfig, load_config
from .data.augment import run_augment
from .data.generate import run_generate
from .export.onnx import run_export
from .inference.listener import WakeWordListener
from .inference.model import WakeWordModel
from .training.trainer import run_train

__version__ = "0.1.0"
__all__ = [
    "WakeWordConfig",
    "WakeWordListener",
    "WakeWordModel",
    "load_config",
    "run_augment",
    "run_export",
    "run_generate",
    "run_train",
    "__version__",
]
