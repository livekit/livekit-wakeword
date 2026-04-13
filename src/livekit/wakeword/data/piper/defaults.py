"""Default Piper/VITS checkpoint layout and download URLs."""

from __future__ import annotations

from pathlib import Path

# Directory under data_path where Piper artifacts live (matches ``livekit-wakeword setup``).
PIPER_DATA_SUBDIR = "piper"

# Default English LibriTTS-high checkpoint from project releases.
DEFAULT_CHECKPOINT_STEM = "en-us-libritts-high"
DEFAULT_STATE_DICT_FILENAME = f"{DEFAULT_CHECKPOINT_STEM}.pt"
DEFAULT_CONFIG_JSON_FILENAME = f"{DEFAULT_CHECKPOINT_STEM}.json"

# GitHub release asset names (may differ from on-disk filenames).
RELEASE_STATE_DICT_ASSET = f"{DEFAULT_CHECKPOINT_STEM}.state_dict.pt"
RELEASE_CONFIG_JSON_ASSET = f"{DEFAULT_CHECKPOINT_STEM}.config.json"

DEFAULT_RELEASE_TAG = "v0.1.0"
DEFAULT_RELEASE_BASE_URL = (
    f"https://github.com/livekit/livekit-wakeword/releases/download/{DEFAULT_RELEASE_TAG}"
)


def default_checkpoint_path(data_path: Path) -> Path:
    """Path to the default VITS state_dict used by setup and generation."""
    return data_path / PIPER_DATA_SUBDIR / DEFAULT_STATE_DICT_FILENAME
