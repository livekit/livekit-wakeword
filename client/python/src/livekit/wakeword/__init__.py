"""livekit-wakeword: run wake word models (training: `livekit.wakeword_train`)."""

from importlib.metadata import version

from .listener import Detection, WakeWordListener
from .model import WakeWordModel

__version__ = version("livekit-wakeword")

__all__ = ["Detection", "WakeWordListener", "WakeWordModel", "__version__"]
