"""Config, environment checks and the CLI surface (no training, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from livekit.wakeword_train.config import WakeWordConfig, default_config_path, load_config, slug


def test_only_the_wakeword_is_required(tmp_path: Path) -> None:
    cfg = WakeWordConfig(wakeword="Hey LiveKit!")
    assert cfg.name == "hey_livekit"
    assert cfg.model_path("standard").name == "hey_livekit.onnx"
    assert cfg.model_path("micro").name == "hey_livekit_micro.tflite"
    assert cfg.steps == 5000 and cfg.plan == "build" and cfg.concurrency is None
    assert (
        slug("你好小智") == "你好小智"
        and default_config_path("hey jarvis").name == "hey_jarvis.yaml"
    )


def test_config_round_trip(sample_config: WakeWordConfig, tmp_path: Path) -> None:
    cfg = sample_config.model_copy(update={"plan": "ship", "concurrency": 12, "steps": 1000})
    again = load_config(cfg.save(tmp_path / "c.yaml"))
    assert again == cfg


def test_environment_problems_are_reported(
    sample_config: WakeWordConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.wakeword_train.pipeline import check_environment

    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    problems = check_environment(sample_config.model_copy(update={"plan": "nope"}))
    assert any("LIVEKIT_URL" in p for p in problems)
    assert any("unknown plan" in p for p in problems)


def test_cli_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.wakeword_train import cli

    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "livekit.wakeword_train.pipeline.run_setup",
        lambda cfg, eval=False: calls.append(("setup", cfg)) or [],
    )
    monkeypatch.setattr(
        "livekit.wakeword_train.pipeline.run_train",
        lambda cfg, type="standard": calls.append((type, cfg)) or Path("x.onnx"),
    )
    runner = CliRunner()
    r = runner.invoke(cli.app, ["setup", "--wakeword", "hey robot", "--plan", "ship"])
    assert r.exit_code == 0, r.output
    assert load_config("hey_robot.yaml").plan == "ship"  # setup wrote the default config
    r = runner.invoke(
        cli.app, ["train", "--wakeword", "hey robot", "--type", "micro", "--steps", "7"]
    )
    assert r.exit_code == 0, r.output
    kind, cfg = calls[-1]
    assert kind == "micro" and cfg.plan == "ship" and cfg.micro_steps == 7  # read the saved config
    r = runner.invoke(cli.app, ["train", "--wakeword", "hey robot", "--type", "large"])
    assert r.exit_code != 0


def test_cli_clips_and_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.wakeword_train import cli

    monkeypatch.chdir(tmp_path)
    calls: list[WakeWordConfig] = []
    monkeypatch.setattr(
        "livekit.wakeword_train.pipeline.run_setup", lambda cfg, eval=False: calls.append(cfg) or []
    )
    monkeypatch.setattr(
        "livekit.wakeword_train.pipeline.run_train",
        lambda cfg, type="standard": calls.append(cfg) or Path("x.onnx"),
    )
    runner = CliRunner()
    r = runner.invoke(cli.app, ["setup", "-w", "hey robot", "--clips", "4"])
    assert r.exit_code == 0, r.output
    saved = load_config("hey_robot.yaml")
    assert (saved.n_positive, saved.n_confusable, saved.n_sentence) == (2000, 2000, 600)
    r = runner.invoke(cli.app, ["train", "-w", "hey robot", "--out", "models"])
    assert r.exit_code == 0, r.output
    assert calls[-1].model_path("standard") == Path("models/hey_robot.onnx")
    assert calls[-1].n_positive == 2000  # train reads the counts setup saved
    assert "--config" not in runner.invoke(cli.app, ["setup", "--help"]).output


def test_setup_regenerates_when_clip_counts_change(
    sample_config: WakeWordConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from livekit.wakeword_train import pipeline

    made: list[tuple[str, bool, int]] = []

    def fake_generate(config, out, splits, overwrite=False, **knobs):  # type: ignore[no-untyped-def]
        made.append((out.name, overwrite, knobs["splits__train__n_positive"]))
        (out / "data").mkdir(parents=True, exist_ok=True)
        (out / "data" / "train-00000.parquet").touch()
        split = {
            "n_positive": knobs["splits__train__n_positive"],
            "n_adversarial": knobs["splits__train__n_adversarial"],
            "n_speech": knobs["splits__train__n_speech"],
        }
        (out / "bench_config.yaml").write_text(yaml.safe_dump({"splits": {"train": split}}))

    monkeypatch.setattr("livekit.wakeword_train.config.CACHE", tmp_path / "cache")
    monkeypatch.setattr(pipeline, "check_environment", lambda config: [])
    monkeypatch.setattr(
        "livekit.wakeword_train.encoders.encoder_dir", lambda tier: tmp_path / "enc" / tier
    )
    monkeypatch.setattr(pipeline, "natural_positives_help", lambda phrase, language="en": True)
    monkeypatch.setattr(pipeline, "_generate", fake_generate)
    cfg = sample_config.model_copy(update={"data_dir": tmp_path / "data"})
    pipeline.run_setup(cfg)
    assert made == [("steered", False, 500), ("natural", False, 500)]
    pipeline.run_setup(cfg)  # cached: nothing new
    assert len(made) == 2
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.runs_dir / "standard.pt").touch()
    pipeline.run_setup(cfg.model_copy(update={"n_positive": 2000}))
    assert made[2:] == [("steered", True, 2000), ("natural", True, 2000)]
    assert not (cfg.runs_dir / "standard.pt").exists()  # the old teacher learned from other speech
