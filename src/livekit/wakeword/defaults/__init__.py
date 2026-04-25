"""Per-backend TTS defaults.

Lives outside ``livekit.wakeword.data`` so ``config`` can import these
without pulling in the ``data`` package (which would cycle back through
``config`` via ``data/__init__.py``).
"""

from . import piper, voxcpm

__all__ = ["piper", "voxcpm"]
