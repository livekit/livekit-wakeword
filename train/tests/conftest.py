"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from livekit.wakeword_train.config import WakeWordConfig


@pytest.fixture
def sample_config(tmp_path: Path) -> WakeWordConfig:
    """A config whose files all live under tmp_path."""
    return WakeWordConfig(
        wakeword="hey test",
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
    )
