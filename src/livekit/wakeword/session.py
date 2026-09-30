"""onnxruntime session options shared by every model the package runs."""

from __future__ import annotations

from onnxruntime import SessionOptions

# onnxruntime 1.27.0 on arm64 (e.g. Apple silicon) runs batch-1 convolutions through its KleidiAI
# kernels, which return wrong results. The speech-embedding model runs one window at a time, so
# every wake word scored ~0 there. Older onnxruntime versions accept this setting and ignore it.
DISABLE_KLEIDIAI = "mlas.disable_kleidiai"


def session_options(sess_options: SessionOptions | None = None) -> SessionOptions:
    """``sess_options`` (or new defaults) with KleidiAI off, unless the caller set that entry."""
    options = sess_options if sess_options is not None else SessionOptions()
    try:
        options.get_session_config_entry(DISABLE_KLEIDIAI)
    except RuntimeError:  # not set
        options.add_session_config_entry(DISABLE_KLEIDIAI, "1")
    return options
