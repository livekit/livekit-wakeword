"""Tests for custom positive sample injection."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from livekit.wakeword.config import CustomPositiveSource, WakeWordConfig
from livekit.wakeword.data.generate import _copy_custom_positives


def _make_wav(
    path: Path,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(duration_s * sample_rate)
    shape = (n_frames,) if channels == 1 else (n_frames, channels)
    data = (np.random.randn(*shape) * 0.1).astype(np.float32)
    sf.write(str(path), data, sample_rate)


class TestCustomPositiveSourceModel:
    def test_default_multiplier_is_one(self) -> None:
        src = CustomPositiveSource(path="/tmp/anywhere")
        assert src.multiplier == 1

    def test_rejects_zero_multiplier(self) -> None:
        with pytest.raises(ValueError):
            CustomPositiveSource(path="/tmp/anywhere", multiplier=0)

    def test_rejects_negative_multiplier(self) -> None:
        with pytest.raises(ValueError):
            CustomPositiveSource(path="/tmp/anywhere", multiplier=-1)


class TestCopyCustomPositives:
    def test_empty_sources_is_noop(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "positive_train"
        written = _copy_custom_positives(split_dir, [], start_index=10)
        assert written == 0
        # Empty sources should not even create the output directory.
        assert not split_dir.exists()

    def test_basic_copy_appends_at_start_index(self, tmp_path: Path) -> None:
        src = tmp_path / "recordings"
        _make_wav(src / "sample1.wav")
        _make_wav(src / "sample2.wav")

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(src), multiplier=1)
        written = _copy_custom_positives(split_dir, [source], start_index=5)

        assert written == 2
        assert (split_dir / "clip_000005.wav").exists()
        assert (split_dir / "clip_000006.wav").exists()
        # Numbering does not collide with pre-existing range.
        assert not (split_dir / "clip_000004.wav").exists()

    def test_multiplier_duplicates_each_file(self, tmp_path: Path) -> None:
        src = tmp_path / "recordings"
        _make_wav(src / "only.wav")

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(src), multiplier=3)
        written = _copy_custom_positives(split_dir, [source], start_index=0)

        assert written == 3
        names = sorted(p.name for p in split_dir.glob("clip_*.wav"))
        assert names == ["clip_000000.wav", "clip_000001.wav", "clip_000002.wav"]

    def test_multiple_sources_continuous_numbering(self, tmp_path: Path) -> None:
        src1 = tmp_path / "first"
        src2 = tmp_path / "second"
        _make_wav(src1 / "a.wav")
        _make_wav(src2 / "b.wav")

        split_dir = tmp_path / "positive_train"
        sources = [
            CustomPositiveSource(path=str(src1), multiplier=2),
            CustomPositiveSource(path=str(src2), multiplier=3),
        ]
        written = _copy_custom_positives(split_dir, sources, start_index=100)

        assert written == 5
        for i in range(100, 105):
            assert (split_dir / f"clip_{i:06d}.wav").exists()

    def test_resume_skips_existing_outputs(self, tmp_path: Path) -> None:
        src = tmp_path / "recordings"
        _make_wav(src / "a.wav")
        _make_wav(src / "b.wav")

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(src), multiplier=2)

        first = _copy_custom_positives(split_dir, [source], start_index=0)
        assert first == 4
        # A second call with identical inputs should be a no-op.
        second = _copy_custom_positives(split_dir, [source], start_index=0)
        assert second == 0
        assert len(list(split_dir.glob("clip_*.wav"))) == 4

    def test_resume_fills_gap_when_partial(self, tmp_path: Path) -> None:
        src = tmp_path / "recordings"
        _make_wav(src / "a.wav")
        _make_wav(src / "b.wav")
        _make_wav(src / "c.wav")

        split_dir = tmp_path / "positive_train"
        split_dir.mkdir()
        # Simulate a partial previous run: first two targets are already on disk.
        (split_dir / "clip_000010.wav").write_bytes(b"existing-a")
        (split_dir / "clip_000011.wav").write_bytes(b"existing-b")

        source = CustomPositiveSource(path=str(src), multiplier=1)
        written = _copy_custom_positives(split_dir, [source], start_index=10)

        # Only the third slot should be newly written.
        assert written == 1
        assert (split_dir / "clip_000010.wav").read_bytes() == b"existing-a"
        assert (split_dir / "clip_000011.wav").read_bytes() == b"existing-b"
        assert (split_dir / "clip_000012.wav").exists()

    def test_rejects_wrong_sample_rate(self, tmp_path: Path) -> None:
        src = tmp_path / "recordings"
        _make_wav(src / "bad.wav", sample_rate=48000)

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(src), multiplier=1)
        with pytest.raises(ValueError, match="sample rate 48000"):
            _copy_custom_positives(split_dir, [source], start_index=0)
        # No partial output should be left behind.
        assert not any(split_dir.glob("clip_*.wav"))

    def test_rejects_stereo(self, tmp_path: Path) -> None:
        src = tmp_path / "recordings"
        _make_wav(src / "stereo.wav", channels=2)

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(src), multiplier=1)
        with pytest.raises(ValueError, match="2 channels"):
            _copy_custom_positives(split_dir, [source], start_index=0)

    def test_missing_source_path_raises(self, tmp_path: Path) -> None:
        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(tmp_path / "nope"), multiplier=1)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _copy_custom_positives(split_dir, [source], start_index=0)

    def test_source_path_is_file_raises(self, tmp_path: Path) -> None:
        file_as_source = tmp_path / "single.wav"
        _make_wav(file_as_source)

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(file_as_source), multiplier=1)
        with pytest.raises(NotADirectoryError, match="must be a directory"):
            _copy_custom_positives(split_dir, [source], start_index=0)

    def test_warns_and_skips_non_wav_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        src = tmp_path / "recordings"
        _make_wav(src / "good.wav")
        (src / "notes.txt").write_text("ignored")
        (src / "cover.mp3").touch()

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(src), multiplier=1)
        with caplog.at_level(logging.WARNING):
            written = _copy_custom_positives(split_dir, [source], start_index=0)

        assert written == 1
        assert any(
            "non-.wav" in rec.message
            and "notes.txt" in rec.message
            or "non-.wav" in rec.message
            and "cover.mp3" in rec.message
            for rec in caplog.records
        )

    def test_empty_source_dir_warns_no_files(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        src = tmp_path / "recordings"
        src.mkdir()

        split_dir = tmp_path / "positive_train"
        source = CustomPositiveSource(path=str(src), multiplier=1)
        with caplog.at_level(logging.WARNING):
            written = _copy_custom_positives(split_dir, [source], start_index=0)
        assert written == 0
        assert any("No .wav files" in rec.message for rec in caplog.records)


class _FakeTtsBackend:
    """Minimal TTS stub that writes placeholder clips for integration testing."""

    def validate_artifacts(self) -> None:
        return None

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        start_index: int = 0,
        batch_size: int = 50,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for i in range(start_index, n_samples):
            p = output_dir / f"clip_{i:06d}.wav"
            p.touch()  # empty placeholder, distinguishable from real copies by size
            written.append(p)
        return written


class TestRunGenerateIntegration:
    """End-to-end: verify run_generate wires custom_positive_samples correctly."""

    def test_injection_happens_after_positive_train_tts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec_dir = tmp_path / "recordings"
        _make_wav(rec_dir / "voice_a.wav")
        _make_wav(rec_dir / "voice_b.wav")

        cfg = WakeWordConfig(
            model_name="hey_integration",
            target_phrases=["hey integration"],
            n_samples=3,
            n_samples_val=1,
            n_background_samples=0,
            n_background_samples_val=0,
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "output"),
            custom_positive_samples=[
                CustomPositiveSource(path=str(rec_dir), multiplier=4),
            ],
        )

        from livekit.wakeword.data import generate as gen_mod

        monkeypatch.setattr(gen_mod, "get_tts_backend", lambda c: _FakeTtsBackend())
        gen_mod.run_generate(cfg)

        pos_train = cfg.model_output_dir / "positive_train"
        all_clips = sorted(p.name for p in pos_train.glob("clip_*.wav"))
        # 3 TTS placeholders + 2 files × multiplier 4 = 11 total clips
        assert len(all_clips) == 11
        assert all_clips[0] == "clip_000000.wav"
        assert all_clips[-1] == "clip_000010.wav"

        # TTS clips (indices 0..2) are empty placeholders from _FakeTtsBackend.
        # Custom clips (indices 3..10) are real wav copies with non-zero size.
        for i in range(3):
            assert (pos_train / f"clip_{i:06d}.wav").stat().st_size == 0
        for i in range(3, 11):
            assert (pos_train / f"clip_{i:06d}.wav").stat().st_size > 0

        # positive_test is unaffected (no custom injection there in v1)
        pos_test_clips = list((cfg.model_output_dir / "positive_test").glob("clip_*.wav"))
        assert len(pos_test_clips) == cfg.n_samples_val

    def test_run_generate_is_idempotent_with_custom_positives(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec_dir = tmp_path / "recordings"
        _make_wav(rec_dir / "voice.wav")

        cfg = WakeWordConfig(
            model_name="hey_resume",
            target_phrases=["hey resume"],
            n_samples=2,
            n_samples_val=1,
            n_background_samples=0,
            n_background_samples_val=0,
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "output"),
            custom_positive_samples=[
                CustomPositiveSource(path=str(rec_dir), multiplier=3),
            ],
        )

        from livekit.wakeword.data import generate as gen_mod

        monkeypatch.setattr(gen_mod, "get_tts_backend", lambda c: _FakeTtsBackend())

        gen_mod.run_generate(cfg)
        first_listing = sorted(
            (p.name, p.stat().st_size)
            for p in (cfg.model_output_dir / "positive_train").glob("clip_*.wav")
        )

        # Second invocation should be a complete no-op on positive_train
        gen_mod.run_generate(cfg)
        second_listing = sorted(
            (p.name, p.stat().st_size)
            for p in (cfg.model_output_dir / "positive_train").glob("clip_*.wav")
        )

        assert first_listing == second_listing
        assert len(first_listing) == 2 + 3  # 2 TTS + 1 file × multiplier 3


class TestConfigIntegration:
    def test_config_accepts_custom_positive_samples(self, tmp_path: Path) -> None:
        cfg = WakeWordConfig(
            model_name="hey_test",
            target_phrases=["hey test"],
            custom_positive_samples=[
                CustomPositiveSource(path=str(tmp_path), multiplier=10),
            ],
        )
        assert len(cfg.custom_positive_samples) == 1
        assert cfg.custom_positive_samples[0].multiplier == 10

    def test_config_default_is_empty_list(self) -> None:
        cfg = WakeWordConfig(
            model_name="hey_test",
            target_phrases=["hey test"],
        )
        assert cfg.custom_positive_samples == []

    def test_config_yaml_parsing(self, tmp_path: Path) -> None:
        rec_dir = tmp_path / "recordings"
        rec_dir.mkdir()
        yaml_text = f"""
model_name: hey_test
target_phrases: ["hey test"]
custom_positive_samples:
  - path: {rec_dir}
    multiplier: 25
"""
        yaml_path = tmp_path / "cfg.yaml"
        yaml_path.write_text(yaml_text)

        from livekit.wakeword.config import load_config

        cfg = load_config(yaml_path)
        assert len(cfg.custom_positive_samples) == 1
        assert cfg.custom_positive_samples[0].path == str(rec_dir)
        assert cfg.custom_positive_samples[0].multiplier == 25
