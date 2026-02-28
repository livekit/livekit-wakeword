"""Real-time inference engine."""

from livekit.wakeword.inference.listener import Detection, WakeWordListener
from livekit.wakeword.inference.model import WakeWordModel

__all__ = ["Detection", "WakeWordListener", "WakeWordModel"]
