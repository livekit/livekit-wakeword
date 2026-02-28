"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from livekit.wakeword.config import WakeWordConfig


@pytest.fixture
def sample_config(tmp_path: Path) -> WakeWordConfig:
    """Minimal config for testing."""
    return WakeWordConfig(
        model_name="test_wakeword",
        target_phrases=["hey test"],
        n_samples=10,
        n_samples_val=5,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def sample_audio() -> torch.Tensor:
    """Random 2-second 16kHz audio batch."""
    return torch.randn(2, 32000)


@pytest.fixture
def sample_embeddings() -> torch.Tensor:
    """Random (batch, 16, 96) embedding sequence."""
    return torch.randn(4, 16, 96)


@pytest.fixture
def sample_features_file(tmp_path: Path) -> Path:
    """Create a temporary .npy features file."""
    features = np.random.randn(100, 16, 96).astype(np.float32)
    path = tmp_path / "test_features.npy"
    np.save(str(path), features)
    return path
