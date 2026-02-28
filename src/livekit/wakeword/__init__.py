"""livekit-wakeword — Simplified pure-PyTorch wake word detection."""

from .config import WakeWordConfig, load_config
from .data.augment import run_augment
from .data.generate import run_generate
from .inference.listener import Detection, WakeWordListener
from .inference.model import WakeWordModel
from .training.trainer import run_train

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    if name == "run_export":
        from .export.onnx import run_export

        return run_export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WakeWordConfig",
    "WakeWordListener",
    "WakeWordModel",
    "Detection",
    "load_config",
    "run_augment",
    "run_export",
    "run_generate",
    "run_train",
    "__version__",
]
