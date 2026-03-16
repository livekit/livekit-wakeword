"""Neural network models for wake word detection."""

# Feature extractors use ONNX only — safe to import eagerly.
from .feature_extractor import MelSpectrogramFrontend, SpeechEmbedding

# Classifier / pipeline modules require torch and are lazy-loaded so that the
# inference-only path (numpy + onnxruntime) works without torch installed.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DNNClassifier": (".classifier", "DNNClassifier"),
    "FCNBlock": (".classifier", "FCNBlock"),
    "RNNClassifier": (".classifier", "RNNClassifier"),
    "build_classifier": (".classifier", "build_classifier"),
    "WakeWordClassifier": (".pipeline", "WakeWordClassifier"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DNNClassifier",
    "FCNBlock",
    "MelSpectrogramFrontend",
    "RNNClassifier",
    "SpeechEmbedding",
    "WakeWordClassifier",
    "build_classifier",
]
