"""Tests for data generation resume support."""

from __future__ import annotations

from pathlib import Path

import pytest

from livekit.wakeword.data.generate import _count_original_clips, synthesize_clips


class TestCountOriginalClips:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert _count_original_clips(tmp_path) == 0

    def test_missing_dir(self, tmp_path: Path) -> None:
        assert _count_original_clips(tmp_path / "nonexistent") == 0

    def test_with_files(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"clip_{i:06d}.wav").touch()
        assert _count_original_clips(tmp_path) == 5

    def test_ignores_augmented(self, tmp_path: Path) -> None:
        # Original clips
        for i in range(3):
            (tmp_path / f"clip_{i:06d}.wav").touch()
        # Augmented variants (should be excluded)
        (tmp_path / "clip_000000_r1.wav").touch()
        (tmp_path / "clip_000001_r2.wav").touch()
        (tmp_path / "clip_000002_r1.wav").touch()
        assert _count_original_clips(tmp_path) == 3

    def test_ignores_unrelated_files(self, tmp_path: Path) -> None:
        (tmp_path / "clip_000000.wav").touch()
        (tmp_path / "metadata.json").touch()
        (tmp_path / "features.npy").touch()
        assert _count_original_clips(tmp_path) == 1


class TestSynthesizeClipsNoModel:
    def test_raises_when_model_path_is_none(self, tmp_path: Path) -> None:
        """Must raise FileNotFoundError instead of generating silent placeholders."""
        with pytest.raises(FileNotFoundError, match="VITS model not found"):
            synthesize_clips(
                phrases=["hello"],
                output_dir=tmp_path,
                n_samples=5,
                vits_model_path=None,
            )

    def test_raises_when_model_path_missing(self, tmp_path: Path) -> None:
        """Must raise FileNotFoundError for a non-existent model path."""
        with pytest.raises(FileNotFoundError, match="VITS model not found"):
            synthesize_clips(
                phrases=["hello"],
                output_dir=tmp_path,
                n_samples=5,
                vits_model_path=tmp_path / "nonexistent.pt",
            )
