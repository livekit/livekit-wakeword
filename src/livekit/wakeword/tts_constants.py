"""TTS path fragments shared by config and data code.

Kept outside ``livekit.wakeword.data`` so ``config`` does not import the ``data``
package (which would circularly import ``config`` via ``data/__init__.py``).
"""

from __future__ import annotations

PIPER_DATA_SUBDIR = "piper"
DEFAULT_CHECKPOINT_STEM = "en-us-libritts-high"
DEFAULT_PIPER_STATE_DICT_FILENAME = f"{DEFAULT_CHECKPOINT_STEM}.pt"
DEFAULT_CHECKPOINT_RELPATH = f"{PIPER_DATA_SUBDIR}/{DEFAULT_PIPER_STATE_DICT_FILENAME}"
