"""livekit-wakeword — Wake word detection for voice-enabled applications."""

from .inference.listener import Detection, WakeWordListener
from .inference.model import WakeWordModel

__version__ = "0.1.0"

# Training / CLI imports are lazy-loaded so that the core inference API
# works with only numpy + onnxruntime (no torch, pydantic, etc.).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "WakeWordConfig": (".config", "WakeWordConfig"),
    "load_config": (".config", "load_config"),
    "run_augment": (".data.augment", "run_augment"),
    "run_extraction": (".data.features", "run_extraction"),
    "run_generate": (".data.generate", "run_generate"),
    "run_train": (".training.trainer", "run_train"),
    "run_export": (".export.onnx", "run_export"),
    "run_eval": (".eval.evaluate", "run_eval"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WakeWordConfig",
    "WakeWordListener",
    "WakeWordModel",
    "Detection",
    "load_config",
    "run_augment",
    "run_eval",
    "run_export",
    "run_extraction",
    "run_generate",
    "run_train",
    "__version__",
]
