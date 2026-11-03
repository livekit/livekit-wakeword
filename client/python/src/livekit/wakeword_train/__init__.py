"""livekit-wakeword training (the `train` extra): type a wake word, get a streaming model."""

from .config import WakeWordConfig, load_config
from .pipeline import run_eval, run_setup, run_train

__all__ = ["WakeWordConfig", "load_config", "run_eval", "run_setup", "run_train"]
