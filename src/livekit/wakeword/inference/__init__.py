"""Real-time inference engine."""

from .listener import Detection, WakeWordListener
from .model import WakeWordModel

__all__ = ["Detection", "WakeWordListener", "WakeWordModel"]
