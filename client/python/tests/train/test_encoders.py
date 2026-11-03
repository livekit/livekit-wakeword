"""Pretrained encoders: baking a phrase equals modulating the live layers; save/load round trip."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from livekit.wakeword_train import encoders  # noqa: E402
from livekit.wakeword_train.micro import MicroEncoder, MicroHead  # noqa: E402
from livekit.wakeword_train.nn import FRAME, Detector, Encoder, Head  # noqa: E402


def _tiny(tier: str) -> encoders.PretrainedEncoder:
    torch.manual_seed(0)
    if tier == "micro":
        enc = MicroEncoder(dim=16, mult=1.0)
        det = Detector(enc, MicroHead(16, ch=8), norm=False)
        top = 4
    else:
        enc = Encoder(dim=16, d=32, layers=3, heads=2, ff=2, k=5, left=8, sub_c=8)
        det = Detector(enc, Head(16, 16, 1, n_heads=2))
        top = 2
    pe = encoders.PretrainedEncoder(det, top=top, mode="bytes").eval()
    with torch.no_grad():  # a non-trivial modulation
        pe.phrase.mlp[-1].weight.normal_(0, 0.05)
    return pe


def _modulated(pe: encoders.PretrainedEncoder, phrase: str, x: torch.Tensor) -> torch.Tensor:
    """Reference: scale and shift the live layer outputs with forward hooks."""
    p = pe.phrase(phrase)
    gs, bs = p.split([sum(pe.sizes)] * 2)
    targets = encoders.film_targets(pe.det, pe.top)
    hooks = []
    for m, g, b in zip(targets, gs.split(pe.sizes), bs.split(pe.sizes)):
        channels_last = isinstance(m, torch.nn.LayerNorm) and len(m.normalized_shape) == 1

        def hook(_m, _i, out, g=g, b=b, last=channels_last):  # type: ignore[no-untyped-def]
            if last:
                return out * (1 + g) + b
            shape = (1, -1, *([1] * (out.dim() - 2)))
            return out * (1 + g.view(shape)) + b.view(shape)

        hooks.append(m.register_forward_hook(hook))
    try:
        return pe.det(x, last=4)
    finally:
        for h in hooks:
            h.remove()


@pytest.mark.parametrize("tier", ["standard", "micro"])
@pytest.mark.parametrize("phrase", ["hey jarvis", "привет джарвис", "你好"])
def test_bake_matches_modulation(tier: str, phrase: str) -> None:
    pe = _tiny(tier)
    x = torch.randn(2, FRAME * 24) * 0.05
    with torch.no_grad():
        ref = _modulated(pe, phrase, x)
        baked = pe.bake(phrase).eval()(x, last=4)
    assert torch.allclose(ref, baked, atol=1e-5)
    assert not isinstance(pe.bake(phrase), encoders.PretrainedEncoder)  # a plain Detector


def test_phrases_differ() -> None:
    pe = _tiny("standard")
    x = torch.randn(1, FRAME * 24) * 0.05
    with torch.no_grad():
        a, b = pe.bake("hey jarvis")(x), pe.bake("ok computer")(x)
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("tier", ["standard", "micro"])
def test_save_load_round_trip(tmp_path, monkeypatch, tier: str) -> None:
    pe = _tiny(tier)
    encoders.save_pretrained(pe, tmp_path / tier, tier)  # type: ignore[arg-type]
    monkeypatch.setattr(encoders, "build", lambda cfg: _tiny(str(cfg["tier"])))
    monkeypatch.setenv("LIVEKIT_WAKEWORD_ENCODERS", str(tmp_path))
    loaded = encoders.load_pretrained(tier)  # type: ignore[arg-type]
    for k, v in pe.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k


def test_token_ids() -> None:
    assert encoders.token_ids("Hey", "chars") == [10, 7, 27]
    assert encoders.token_ids("é", "bytes") == [2 + 0xC3, 2 + 0xA9]
    assert encoders.token_ids("", "bytes") == [1]


def _tarball(tmp_path: Path, tiers: tuple[str, ...] = ("standard", "micro")) -> tuple[Path, str]:
    import hashlib
    import tarfile

    src = tmp_path / "src" / "encoders"
    for t in tiers:
        (src / t).mkdir(parents=True)
        (src / t / "config.json").write_text("{}")
        (src / t / "model.safetensors").write_bytes(t.encode())
    tb = tmp_path / "encoders.tar.gz"
    with tarfile.open(tb, "w:gz") as tar:
        tar.add(src, arcname="encoders")
    return tb, hashlib.sha256(tb.read_bytes()).hexdigest()


def test_install_encoders_checks_the_tarball(tmp_path: Path) -> None:
    install_encoders = encoders.install_encoders
    tb, sha = _tarball(tmp_path)
    root = install_encoders(tb, sha, tmp_path / "cache" / "v1")
    assert (root / "micro" / "model.safetensors").read_bytes() == b"micro"
    with pytest.raises(RuntimeError, match="sha256"):
        install_encoders(tb, "0" * 64, tmp_path / "cache" / "v2")
    tb1, sha1 = _tarball(tmp_path / "one", ("standard",))
    with pytest.raises(RuntimeError, match="encoders/micro"):
        install_encoders(tb1, sha1, tmp_path / "cache" / "v3")


def test_encoder_dir_downloads_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_WAKEWORD_ENCODERS", raising=False)
    monkeypatch.setattr("livekit.wakeword_train.config.CACHE", tmp_path / "cache")
    tb, sha = _tarball(tmp_path)
    calls: list[Path] = []

    def fake_download(root: Path) -> Path:
        calls.append(root)
        return encoders.install_encoders(tb, sha, root)

    monkeypatch.setattr(encoders, "download_encoders", fake_download)
    d = encoders.encoder_dir("standard")
    assert d == tmp_path / "cache" / "encoders" / encoders.ENCODERS_RELEASE[1] / "standard"
    encoders.encoder_dir("micro")
    assert len(calls) == 1  # cached after the first download
    monkeypatch.setenv("LIVEKIT_WAKEWORD_ENCODERS", str(tmp_path / "mine"))
    assert encoders.encoder_dir("micro") == tmp_path / "mine" / "micro"
