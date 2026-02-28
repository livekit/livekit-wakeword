"""livekit-wakeword — Simplified pure-PyTorch wake word detection."""

from livekit.wakeword.inference.listener import WakeWordListener
from livekit.wakeword.inference.model import WakeWordModel

__version__ = "0.1.0"
__all__ = ["WakeWordListener", "WakeWordModel", "__version__"]
