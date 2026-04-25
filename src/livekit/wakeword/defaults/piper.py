"""Default Piper VITS checkpoint layout, release URLs, and path resolution."""

from __future__ import annotations

from pathlib import Path

# On-disk layout under ``data_dir``.
DATA_SUBDIR = "piper"
CHECKPOINT_STEM = "en-us-libritts-high"
STATE_DICT_FILENAME = f"{CHECKPOINT_STEM}.pt"
CONFIG_JSON_FILENAME = f"{CHECKPOINT_STEM}.json"
CHECKPOINT_RELPATH = f"{DATA_SUBDIR}/{STATE_DICT_FILENAME}"

# GitHub release asset names (differ from on-disk filenames).
RELEASE_TAG = "v0.1.0"
RELEASE_BASE_URL = (
    f"https://github.com/livekit/livekit-wakeword/releases/download/{RELEASE_TAG}"
)
RELEASE_STATE_DICT_ASSET = f"{CHECKPOINT_STEM}.state_dict.pt"
RELEASE_CONFIG_JSON_ASSET = f"{CHECKPOINT_STEM}.config.json"


def checkpoint_path(
    data_path: Path,
    *,
    checkpoint_relpath: str | None = None,
) -> Path:
    """Resolve VITS state_dict path under *data_path* (used by tools without full config)."""
    rel = checkpoint_relpath or CHECKPOINT_RELPATH
    return (data_path / Path(rel)).resolve()
