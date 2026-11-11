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
    monkeypatch.setattr("livekit.wakeword_train.pipeline.check_environment", lambda cfg: [])
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
    monkeypatch.setattr("livekit.wakeword_train.pipeline.check_environment", lambda cfg: [])
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
        bench = {
            "wake_word": config.wakeword,
            "language": config.language,
            "pronunciations": dict(config.pronunciations),
            "splits": {"train": split},
        }
        (out / "bench_config.yaml").write_text(yaml.safe_dump(bench))

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
    # a changed pronunciation: the old clips say the phrase differently
    bigger = cfg.model_copy(update={"n_positive": 2000})
    pipeline.run_setup(bigger)
    assert len(made) == 4
    pipeline.run_setup(bigger.model_copy(update={"pronunciations": {"test": "T EH1 S T"}}))
    assert [m[:2] for m in made[4:]] == [("steered", True), ("natural", True)]


def test_cli_overrides_are_validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.wakeword_train import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("livekit.wakeword_train.pipeline.check_environment", lambda cfg: [])
    monkeypatch.setattr("livekit.wakeword_train.pipeline.run_setup", lambda cfg, eval=False: [])
    runner = CliRunner()
    r = runner.invoke(cli.app, ["setup", "-w", "hey robot", "--concurrency", "0"])
    assert r.exit_code == 1 and "concurrency" in r.output
    assert not Path("hey_robot.yaml").exists()
    r = runner.invoke(cli.app, ["train", "-w", "hey robot", "--steps", "0"])
    assert r.exit_code == 1 and "steps" in r.output
    Path("hey_robot.yaml").write_text("wakeword: hey robot\nmicro_step: 10\n")  # typo
    r = runner.invoke(cli.app, ["train", "-w", "hey robot"])
    assert r.exit_code == 1 and "micro_step" in r.output and "hey_robot.yaml" in r.output


def test_language_codes() -> None:
    from livekit.wakeword_train.config import is_english
    from livekit.wakeword_train.pipeline import natural_positives_help, phrase_problems

    assert is_english("en") and is_english("en-US") and is_english("EN_gb")
    assert not is_english("es") and not is_english("eng-x")
    assert "zork" in phrase_problems(WakeWordConfig(wakeword="hey zork", language="en-US"))[0]
    assert not natural_positives_help("hey livekit", language="en-US")


def test_clip_cache_follows_the_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from livekit.wakeword_train import data

    ds = tmp_path / "ds"
    (ds / "data").mkdir(parents=True)
    parquet = ds / "data" / "train-00000.parquet"
    parquet.touch()
    cdir = tmp_path / "cache"
    assert not data._cache_fresh(cdir, ds)
    for c in data._CATS:
        (cdir / c).mkdir(parents=True)
        (cdir / c / "ends.npy").touch()
    assert data._cache_fresh(cdir, ds)
    later = (cdir / "speech" / "ends.npy").stat().st_mtime + 10
    os.utime(parquet, (later, later))  # regenerated after the cache was built
    assert not data._cache_fresh(cdir, ds)


def test_cli_setup_checks_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.wakeword_train import cli

    monkeypatch.chdir(tmp_path)
    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    runner = CliRunner()
    r = runner.invoke(cli.app, ["setup", "-w", "hey zork"])
    assert r.exit_code == 1
    assert "LIVEKIT_URL" in r.output and "--pronounce" in r.output and "zork" in r.output
    assert not Path("hey_zork.yaml").exists()  # nothing written when setup cannot run
    assert runner.invoke(cli.app, ["--version"]).output.startswith("livekit-wakeword ")
    r = runner.invoke(cli.app, ["eval", "-w", "hey zork", "-m", "missing.onnx"])
    assert r.exit_code == 1 and "no model at missing.onnx" in r.output


def test_cli_pronounce_is_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.wakeword_train import cli, pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "run_setup", lambda cfg, eval=False: [])
    seen: list[list[str]] = []
    monkeypatch.setattr(
        pipeline, "check_environment", lambda cfg: seen.append(pipeline.phrase_problems(cfg)) or []
    )
    r = CliRunner().invoke(cli.app, ["setup", "-w", "hey zork", "--pronounce", "Zork=Z AO1 R K"])
    assert r.exit_code == 0, r.output
    assert seen == [[]]  # the pronunciation covers the unknown word
    assert load_config("hey_zork.yaml").pronunciations == {"zork": "Z AO1 R K"}


def test_phrase_pronunciations() -> None:
    from livekit.wakeword_train.pipeline import _wanted, phoneme_steered, phrase_problems

    assert phrase_problems(WakeWordConfig(wakeword="hey livekit")) == []  # live + kit
    zork = WakeWordConfig(wakeword="hey zork")
    assert "zork" in phrase_problems(zork)[0] and not phoneme_steered(zork)
    fixed = zork.model_copy(update={"pronunciations": {"zork": "Z AO1 R K"}})
    assert phrase_problems(fixed) == [] and phoneme_steered(fixed)
    bad = zork.model_copy(update={"pronunciations": {"zork": "zork"}})
    assert "not ARPAbet" in phrase_problems(bad)[0]
    # other languages: no error, plain-text TTS only, one set with every clip
    vi = WakeWordConfig(wakeword="xin chào", language="vi")
    assert phrase_problems(vi) == [] and not phoneme_steered(vi)
    assert _wanted(vi) == {"natural": (1000, 500, 150)}
    es = WakeWordConfig(wakeword="hola amigo", language="es")  # dictionary words, still Spanish
    assert not phoneme_steered(es) and _wanted(es) == {"natural": (1000, 500, 150)}


def test_generate_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import livekit_wakeword_bench

    from livekit.wakeword_train import pipeline

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(livekit_wakeword_bench, "generate", lambda *a, **kw: calls.append(kw) or {})
    cfg = WakeWordConfig(wakeword="hey zork", pronunciations={"zork": "Z AO1 R K"})
    pipeline._generate(cfg, tmp_path, ["train"])
    assert calls[-1]["pronunciations__zork"] == '"Z AO1 R K"' and "tts__phonemes" not in calls[-1]
    pipeline._generate(WakeWordConfig(wakeword="xin chào", language="vi"), tmp_path, ["train"])
    assert calls[-1]["tts__phonemes"] is False and "pronunciations__xin" in calls[-1]


def test_default_aug_follows_the_log_mel_floor() -> None:
    import torch

    from livekit.wakeword_train.augment import AugConfig, robust_aug
    from livekit.wakeword_train.loop import default_aug
    from livekit.wakeword_train.nn import MEL_DEFAULT, MEL_LOW_FLOOR, LogMel

    def det(params: tuple[float, float, float]) -> torch.nn.Module:
        m = torch.nn.Module()
        m.encoder = torch.nn.Module()
        m.encoder.frontend = LogMel(params=params)
        return m

    assert default_aug(det(MEL_DEFAULT)) == AugConfig(p_codec=0.25)  # float32 params
    assert default_aug(det(MEL_LOW_FLOOR)) == robust_aug()


def test_hard_negatives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand-written confusables: the only ones for languages other than English (#118, #119)."""
    import livekit_wakeword_bench
    import yaml
    from pydantic import ValidationError

    from livekit.wakeword_train import pipeline

    zh = WakeWordConfig(
        wakeword="嘿小机", language="zh", hard_negatives=[" 喂小机", "嘿小七", "喂小机", ""]
    )
    assert zh.hard_negatives == ["喂小机", "嘿小七"]
    en = WakeWordConfig(wakeword="hey robot", hard_negatives=["hey rabbit", "hey zork"])
    assert pipeline.unknown_hard_negative_words(en) == ["zork"]
    assert any("'zork' in hard_negatives" in p for p in pipeline.phrase_problems(en))
    fixed = en.model_copy(update={"pronunciations": {"zork": "Z AO1 R K"}})
    assert pipeline.phrase_problems(fixed) == []
    for said in ("嘿小机", "嘿，小机啊"):
        with pytest.raises(ValidationError, match="wake word"):
            WakeWordConfig(wakeword="嘿小机", language="zh", hard_negatives=[said])
    for said in ("Hey Zork!", "hey zork robot", "ok, hey-zork"):
        with pytest.raises(ValidationError, match="wake word"):
            WakeWordConfig(wakeword="hey zork", hard_negatives=[said])
    WakeWordConfig(wakeword="hey zork", hard_negatives=["hey zorkington", "say zork", "hey"])
    WakeWordConfig(wakeword="привет", language="ru", hard_negatives=["приветствую"])
    with pytest.raises(ValidationError, match="wake word"):
        WakeWordConfig(wakeword="ねえ ロボット", language="ja", hard_negatives=["ねえロボット君"])

    # listed (empty) in a non-English config to fill in, left out of an English one
    assert (
        "hard_negatives: []"
        in WakeWordConfig(wakeword="hola", language="es").save(tmp_path / "es.yaml").read_text()
    )
    assert (
        "hard_negatives"
        not in WakeWordConfig(wakeword="hey robot").save(tmp_path / "en.yaml").read_text()
    )
    assert load_config(zh.save(tmp_path / "zh.yaml")) == zh

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(livekit_wakeword_bench, "generate", lambda *a, **kw: calls.append(kw) or {})
    pipeline._generate(zh, tmp_path, ["train"])
    assert yaml.safe_load(str(calls[-1]["hard_negatives"])) == ["喂小机", "嘿小七"]
    pipeline._generate(WakeWordConfig(wakeword="hola", language="es"), tmp_path, ["train"])
    assert "hard_negatives" not in calls[-1]
    pipeline._generate(en, tmp_path, ["train"])  # English: with its automatic confusables
    assert yaml.safe_load(str(calls[-1]["hard_negatives"])) == ["hey rabbit", "hey zork"]
    pipeline._generate(en, tmp_path, ["train"], hard_negatives=False)
    assert "hard_negatives" not in calls[-1]  # a set without confusables

    # a dataset made with other hard negatives is out of date
    bench = {"wake_word": "嘿小机", "language": "zh", "hard_negatives": ["嘿小七", "喂小机"]}
    (tmp_path / "bench_config.yaml").write_text(yaml.safe_dump(bench, allow_unicode=True))
    assert pipeline._config_change(tmp_path, zh) is None
    other = zh.model_copy(update={"hard_negatives": ["小机"]})
    assert pipeline._config_change(tmp_path, other) == "hard_negatives changed"
    none = zh.model_copy(update={"hard_negatives": []})
    assert pipeline._config_change(tmp_path, none) == "hard_negatives changed"
    assert pipeline._config_change(tmp_path, none, confusables=False) is None


def test_setup_hard_negative_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.wakeword_train import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("livekit.wakeword_train.pipeline.check_environment", lambda cfg: [])
    monkeypatch.setattr("livekit.wakeword_train.pipeline.run_setup", lambda cfg, eval=False: [])
    runner = CliRunner()
    args = ["setup", "-w", "嘿小机", "-l", "zh", "--hard-negative", "喂小机"]
    assert runner.invoke(cli.app, [*args, "--hard-negative", "嘿小七"]).exit_code == 0
    path = default_config_path("嘿小机")
    assert load_config(path).hard_negatives == ["喂小机", "嘿小七"]
    r = runner.invoke(cli.app, ["setup", "-w", "嘿小机", "--hard-negative", "小机"])  # adds
    assert r.exit_code == 0, r.output
    assert load_config(path).hard_negatives == ["喂小机", "嘿小七", "小机"]
    r = runner.invoke(cli.app, ["setup", "-w", "hey robot", "--hard-negative", "hey rabbit"])
    assert r.exit_code == 0, r.output
    assert load_config("hey_robot.yaml").hard_negatives == ["hey rabbit"]
    r = runner.invoke(cli.app, ["setup", "-w", "hey robot", "--hard-negative", "hey robot"])
    assert r.exit_code == 1 and "wake word" in r.output
