"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from livekit.wakeword.config import (
    MODEL_SIZE_PRESETS,
    ModelConfig,
    ModelSize,
    ModelType,
    TtsBackend,
    WakeWordConfig,
    load_config,
)


def test_default_config():
    config = WakeWordConfig(model_name="test", target_phrases=["hey test"])
    assert config.model_name == "test"
    assert config.steps == 50000
    assert config.model.model_type == ModelType.conv_attention
    assert config.model.model_size == ModelSize.small


def test_model_size_presets():
    assert MODEL_SIZE_PRESETS[ModelSize.tiny] == (16, 1)
    assert MODEL_SIZE_PRESETS[ModelSize.small] == (32, 1)
    assert MODEL_SIZE_PRESETS[ModelSize.medium] == (128, 2)
    assert MODEL_SIZE_PRESETS[ModelSize.large] == (256, 3)


def test_model_config_properties():
    cfg = ModelConfig(model_type=ModelType.dnn, model_size=ModelSize.medium)
    assert cfg.layer_dim == 128
    assert cfg.n_blocks == 2


def test_config_output_dir():
    config = WakeWordConfig(
        model_name="hey_jarvis",
        target_phrases=["hey jarvis"],
        output_dir="./output",
    )
    assert config.model_output_dir == Path("./output/hey_jarvis")


def test_load_config_from_yaml(tmp_path: Path):
    yaml_data = {
        "model_name": "test_word",
        "target_phrases": ["test word"],
        "steps": 1000,
        "model": {"model_type": "rnn", "model_size": "large"},
    }
    yaml_path = tmp_path / "test.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    config = load_config(yaml_path)
    assert config.model_name == "test_word"
    assert config.steps == 1000
    assert config.model.model_type == ModelType.rnn
    assert config.model.model_size == ModelSize.large


def test_batch_n_per_class_default():
    config = WakeWordConfig(model_name="test", target_phrases=["test"])
    assert config.batch_n_per_class["positive"] == 50
    assert config.batch_n_per_class["adversarial_negative"] == 50
    assert config.batch_n_per_class["ACAV100M_sample"] == 1024


def test_piper_checkpoint_path_default(tmp_path: Path) -> None:
    data = tmp_path / "data"
    config = WakeWordConfig(
        model_name="test",
        target_phrases=["hey"],
        data_dir=str(data),
    )
    assert config.piper_checkpoint_path == (data / "piper" / "en-us-libritts-high.pt").resolve()


def test_piper_checkpoint_path_custom_relpath(tmp_path: Path) -> None:
    data = tmp_path / "data"
    config = WakeWordConfig(
        model_name="test",
        target_phrases=["hey"],
        data_dir=str(data),
        piper_tts={"checkpoint_relpath": "models/custom.pt"},
    )
    assert config.piper_checkpoint_path == (data / "models" / "custom.pt").resolve()


def test_tts_backend_enum_in_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "model_name: t\n"
        "target_phrases: [a]\n"
        "tts_backend: piper_vits\n",
        encoding="utf-8",
    )
    cfg = load_config(yaml_path)
    assert cfg.tts_backend is TtsBackend.piper_vits


def test_tts_backend_voxcpm_in_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "model_name: t\n"
        "target_phrases: [a]\n"
        "data_dir: ./d\n"
        "tts_backend: voxcpm\n",
        encoding="utf-8",
    )
    cfg = load_config(yaml_path)
    assert cfg.tts_backend is TtsBackend.voxcpm


def test_voxcpm_local_model_path_default_cache(tmp_path: Path) -> None:
    data = tmp_path / "data"
    cfg = WakeWordConfig(
        model_name="t",
        target_phrases=["hey"],
        data_dir=str(data),
        tts_backend=TtsBackend.voxcpm,
    )
    assert "voxcpm" in str(cfg.voxcpm_local_model_path)
    assert cfg.voxcpm_local_model_path == (data / "voxcpm" / "VoxCPM2").resolve()


def test_voxcpm_local_model_path_override(tmp_path: Path) -> None:
    data = tmp_path / "data"
    cfg = WakeWordConfig(
        model_name="t",
        target_phrases=["hey"],
        data_dir=str(data),
        tts_backend=TtsBackend.voxcpm,
        voxcpm_tts={"local_model_path": "models/vox"},
    )
    assert cfg.voxcpm_local_model_path == (data / "models" / "vox").resolve()
