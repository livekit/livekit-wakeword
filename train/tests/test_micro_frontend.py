"""The ESP32 C front end (lkww_frontend.c) reproduces the PyTorch log-mel, streamed per 80 ms."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from livekit.wakeword_train.nn import FRAME, MEL_DEFAULT, MEL_LOW_FLOOR, LogMel

ESP32 = Path(__file__).resolve().parents[2] / "client" / "esp32"


@pytest.fixture(scope="module")
def frontend_cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    exe = tmp_path_factory.mktemp("micro") / "frontend_cli"
    subprocess.run(
        [
            cc,
            "-O2",
            "-std=c99",
            "-Wall",
            "-Werror",
            "-o",
            str(exe),
            str(ESP32 / "tools/frontend_cli.c"),
            str(ESP32 / "livekit_wakeword/src/lkww_frontend.c"),
            "-lm",
        ],
        check=True,
    )
    return exe


@pytest.mark.parametrize("params", [None, MEL_DEFAULT, MEL_LOW_FLOOR])
def test_matches_torch_logmel(
    frontend_cli: Path, params: tuple[float, float, float] | None
) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(FRAME * 25) / 16000
    audio = 0.3 * np.sin(2 * np.pi * (200 + 900 * t) * t) + 0.05 * rng.standard_normal(t.size)
    audio[FRAME * 10 : FRAME * 12] = 0.0  # digital silence exercises the log floor
    audio = audio.astype(np.float32)

    audio[FRAME * 15 : FRAME * 17] *= 1e-3  # quiet input: only the low floor resolves it
    args = [str(frontend_cli)] + ([repr(v) for v in params] if params else [])
    out = subprocess.run(args, input=audio.tobytes(), capture_output=True, check=True)
    c_mel = np.frombuffer(out.stdout, np.float32).reshape(-1, 40)

    with torch.no_grad():
        fe = LogMel(params=params or MEL_LOW_FLOOR)  # no arguments: the C default (current encoder)
        ref = fe(torch.from_numpy(audio)[None])[0].numpy()
    assert c_mel.shape == ref.shape
    np.testing.assert_allclose(c_mel, ref, atol=1e-4)
