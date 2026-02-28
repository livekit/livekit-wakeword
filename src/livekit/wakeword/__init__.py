"""livekit-wakeword — Simplified pure-PyTorch wake word detection."""

from .inference.listener import WakeWordListener
from .inference.model import WakeWordModel

__version__ = "0.1.0"
__all__ = ["WakeWordListener", "WakeWordModel", "__version__"]
