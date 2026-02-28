"""Tests for data generation resume support."""

from __future__ import annotations

from pathlib import Path

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


class TestSynthesizeClipsResume:
    def test_silence_fallback_starts_from_start_index(self, tmp_path: Path) -> None:
        """Silence fallback should only generate clips from start_index onward."""
        # Pre-create first 3 clips (simulating prior run)
        for i in range(3):
            (tmp_path / f"clip_{i:06d}.wav").write_text("existing")

        # Resume from index 3, targeting 5 total
        result = synthesize_clips(
            phrases=["hello"],
            output_dir=tmp_path,
            n_samples=5,
            vits_model_path=None,
            start_index=3,
        )

        # Should only generate clips 3 and 4
        assert len(result) == 2
        assert result[0] == tmp_path / "clip_000003.wav"
        assert result[1] == tmp_path / "clip_000004.wav"

        # Pre-existing clips should still exist
        assert (tmp_path / "clip_000000.wav").exists()

    def test_silence_fallback_skip_when_complete(self, tmp_path: Path) -> None:
        """When start_index == n_samples, no clips should be generated."""
        result = synthesize_clips(
            phrases=["hello"],
            output_dir=tmp_path,
            n_samples=5,
            vits_model_path=None,
            start_index=5,
        )
        assert len(result) == 0
