"""The `livekit-wakeword` entry point: `listen` runs without the trainer (the `train` extra)."""

from __future__ import annotations

import sys
import types

import pytest

from livekit.wakeword._cli import main


@pytest.fixture
def no_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `livekit.wakeword_train` unimportable, as in an install without the train extra."""
    for name in list(sys.modules):
        if name.startswith("livekit.wakeword_train"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "livekit.wakeword_train", None)


@pytest.mark.usefixtures("no_trainer")
def test_listen_runs_without_the_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyaudio", types.ModuleType("pyaudio"))
    with pytest.raises(SystemExit) as e:
        main(["listen", "no_such_model.onnx"])  # gets as far as loading the model
    assert "no_such_model.onnx" in str(e.value.code)
    assert "train extra" not in str(e.value.code)


@pytest.mark.usefixtures("no_trainer")
def test_listen_explains_missing_pyaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyaudio", None)
    with pytest.raises(SystemExit) as e:
        main(["listen", "model.onnx"])
    assert "PyAudio" in str(e.value.code)


@pytest.mark.usefixtures("no_trainer")
def test_listen_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["listen", "--help"])
    assert e.value.code == 0
    assert "--threshold" in capsys.readouterr().out


@pytest.mark.usefixtures("no_trainer")
@pytest.mark.skipif(sys.version_info < (3, 12), reason="3.11 gets the Python version message")
def test_training_commands_need_the_train_extra() -> None:
    with pytest.raises(SystemExit) as e:
        main(["train", "--wakeword", "hey robot"])
    assert "train extra" in str(e.value.code)
