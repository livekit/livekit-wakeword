"""Default Piper/VITS checkpoint layout and download URLs."""

from __future__ import annotations

from pathlib import Path

from livekit.wakeword.tts_constants import (
    DEFAULT_CHECKPOINT_RELPATH,
    DEFAULT_CHECKPOINT_STEM,
    DEFAULT_PIPER_STATE_DICT_FILENAME,
)

# On-disk basename for the default checkpoint (same as release asset base name + .pt).
DEFAULT_STATE_DICT_FILENAME = DEFAULT_PIPER_STATE_DICT_FILENAME
DEFAULT_CONFIG_JSON_FILENAME = f"{DEFAULT_CHECKPOINT_STEM}.json"

# GitHub release asset names (may differ from on-disk filenames).
RELEASE_STATE_DICT_ASSET = f"{DEFAULT_CHECKPOINT_STEM}.state_dict.pt"
RELEASE_CONFIG_JSON_ASSET = f"{DEFAULT_CHECKPOINT_STEM}.config.json"

DEFAULT_RELEASE_TAG = "v0.1.0"
DEFAULT_RELEASE_BASE_URL = (
    f"https://github.com/livekit/livekit-wakeword/releases/download/{DEFAULT_RELEASE_TAG}"
)


def default_checkpoint_path(
    data_path: Path,
    *,
    checkpoint_relpath: str | None = None,
) -> Path:
    """Resolve VITS state_dict path under *data_path* (used by tools without full config)."""
    rel = checkpoint_relpath or DEFAULT_CHECKPOINT_RELPATH
    return (data_path / Path(rel)).resolve()
