"""Pretrained encoders and phrase baking.

A phrase encoder turns the wake phrase into a per-channel scale and shift for some of the
detector's layers; `PretrainedEncoder.bake` folds them into the weights and returns a plain
`Detector` (the phrase encoder never ships inside a wake word model)::

    det = load_pretrained("standard").bake("hey jarvis")

Weights come from ``$LIVEKIT_WAKEWORD_ENCODERS/<tier>/`` (``config.json`` + ``model.safetensors``)
when set, else from the GitHub release in `ENCODERS_RELEASE`, downloaded once (checksum verified)
into ``~/.cache/livekit-wakeword/encoders/<tag>/``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn as nn

from .nn import ConformerBlock, Detector, Encoder

logger = logging.getLogger(__name__)

Tier = Literal["standard", "micro"]
# repo, tag, asset, sha256: encoders.tar.gz holds encoders/standard/ and encoders/micro/
ENCODERS_RELEASE = (
    "livekit/livekit-wakeword",
    "v2.0.0",
    "encoders.tar.gz",
    "a13a83c34752f6e1c20c47f1b048c02217f6d147ee721ea101eb58e59d4922e7",
)
CHARS = " abcdefghijklmnopqrstuvwxyz'-"


def token_ids(text: str, mode: str) -> list[int]:
    """Phrase -> token ids: UTF-8 bytes of the NFC lowercase text, or `CHARS` (else unknown)."""
    if mode == "bytes":
        b = unicodedata.normalize("NFC", text.lower()).encode("utf-8")[:96]
        return [2 + x for x in b] or [1]
    return [CHARS.find(c) + 2 if c in CHARS else 1 for c in text.lower()][:48] or [1]


class PhraseEncoder(nn.Module):
    """Tokens -> 2-layer bidirectional GRU -> mean + last state -> (scale, shift) per channel."""

    def __init__(self, n_out: int, mode: str = "bytes", d: int = 192, emb: int = 64) -> None:
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(258 if mode == "bytes" else len(CHARS) + 2, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, d, num_layers=2, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(4 * d, 512), nn.SiLU(), nn.Linear(512, n_out))

    def forward(self, text: str) -> torch.Tensor:  # -> (n_out,)
        ids = torch.tensor([token_ids(text, self.mode)], device=self.emb.weight.device)
        h, _ = self.rnn(self.emb(ids))
        out: torch.Tensor = self.mlp(torch.cat([h.mean(1), h[:, -1]], -1))[0]
        return out


def film_targets(det: Detector, top: int) -> list[nn.Module]:
    """The modules a phrase modulates, in weight order."""
    out: list[nn.Module] = []
    head = det.head
    head_conv = cast(nn.Sequential, head.conv)
    if isinstance(det.encoder, Encoder):
        for blk in list(det.encoder.blocks)[-top:]:
            b = cast(ConformerBlock, blk)
            out += [b.ff1[0], b.ln_att, b.ln_conv, b.conv_norm, b.ff2[0], b.ln_out]
        if isinstance(det.norm, nn.LayerNorm):
            out.append(det.norm)
        out += [m for m in head_conv if isinstance(m, nn.LayerNorm)]
        out.append(cast(nn.Module, head.attn_norm))
        return out
    for blk in list(det.encoder.blocks)[-top:]:
        out += [m for m in blk.modules() if isinstance(m, nn.BatchNorm2d)]
    out += [m for m in det.encoder.head if isinstance(m, nn.BatchNorm2d)]
    out += [m for m in head_conv if isinstance(m, nn.Conv1d)]
    return out


def _channels(m: nn.Module) -> int:
    if isinstance(m, nn.LayerNorm):
        return int(m.normalized_shape[0])
    if isinstance(m, nn.BatchNorm2d):
        return int(m.num_features)
    assert isinstance(m, nn.Conv1d)
    return int(m.out_channels)


class PretrainedEncoder(nn.Module):
    """A detector plus the phrase encoder that adapts it to any wake phrase."""

    def __init__(self, det: Detector, top: int, mode: str = "bytes") -> None:
        super().__init__()
        self.det, self.top = det, top
        self.sizes = [_channels(m) for m in film_targets(det, top)]
        self.phrase = PhraseEncoder(2 * sum(self.sizes), mode=mode)

    @torch.no_grad()
    def bake(self, phrase: str) -> Detector:
        """A plain `Detector` for `phrase`: the phrase's scale and shift folded into the weights."""
        det = copy.deepcopy(self.det).cpu()
        p = self.phrase.cpu().eval()(phrase)
        gs, bs = p.split([sum(self.sizes)] * 2)
        for m, g, b in zip(film_targets(det, self.top), gs.split(self.sizes), bs.split(self.sizes)):
            w, bias = cast(torch.Tensor, m.weight), cast(torch.Tensor, m.bias)
            w.mul_((1 + g).view(-1, *([1] * (w.dim() - 1))))
            bias.mul_((1 + g).view(-1, *([1] * (bias.dim() - 1))))
            bias.add_(b.view(-1, *([1] * (bias.dim() - 1))))
        return det


def build(config: dict[str, object]) -> PretrainedEncoder:
    """An untrained `PretrainedEncoder` with the architecture `config` describes."""
    if config["tier"] == "micro":
        from .micro import MicroEncoder, micro_detector

        det = micro_detector(MicroEncoder())
    else:
        det = Detector(Encoder())
    return PretrainedEncoder(det, top=int(str(config["top"])), mode=str(config["text_mode"]))


def encoder_dir(tier: Tier) -> Path:
    """`$LIVEKIT_WAKEWORD_ENCODERS/<tier>`, else the cached release, downloaded on first use."""
    local = os.environ.get("LIVEKIT_WAKEWORD_ENCODERS")
    if local:
        return Path(local) / tier
    from .config import CACHE

    root = CACHE / "encoders" / ENCODERS_RELEASE[1]
    if not all((root / t / "model.safetensors").exists() for t in ("standard", "micro")):
        download_encoders(root)
    return root / tier


def _github_token() -> str | None:
    """GH_TOKEN / GITHUB_TOKEN, else the GitHub CLI's login (`gh auth login`)."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token or not shutil.which("gh"):
        return token
    r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None  # handled by hand: the storage URL must not get the GitHub token


def _fetch_asset(repo: str, tag: str, name: str, dest: Path) -> None:
    """Download a release asset to `dest` (with a token if one is available: rate limits)."""
    headers = {"User-Agent": "livekit-wakeword", "Accept": "application/vnd.github+json"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(urllib.request.Request(api, headers=headers), timeout=30) as r:
        assets = {a["name"]: a["url"] for a in json.load(r)["assets"]}
    req = urllib.request.Request(
        assets[name], headers={**headers, "Accept": "application/octet-stream"}
    )
    try:
        resp = urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        resp = urllib.request.urlopen(e.headers["Location"], timeout=60)  # signed URL, no token
    with resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, 1 << 20)


def install_encoders(tarball: Path, sha256: str, root: Path) -> Path:
    """Check `tarball` against `sha256` and unpack its encoders/standard and encoders/micro into
    `root` (replaced atomically)."""
    h = hashlib.sha256()
    with tarball.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != sha256:
        raise RuntimeError(f"{tarball.name}: sha256 {h.hexdigest()}, expected {sha256}")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        with tarfile.open(tarball) as tar:
            tar.extractall(tmp, filter="data")
        src = Path(tmp) / "encoders"
        if not all((src / t / "model.safetensors").exists() for t in ("standard", "micro")):
            raise RuntimeError(f"{tarball.name} has no encoders/standard and encoders/micro")
        if root.exists():
            shutil.rmtree(root)
        src.rename(root)
    return root


def download_encoders(root: Path) -> Path:
    """Fetch the release's encoders.tar.gz (`ENCODERS_RELEASE`) and install it into `root`."""
    repo, tag, name, sha256 = ENCODERS_RELEASE
    logger.info("downloading the pretrained encoders (%s %s, 37 MB, once)", repo, tag)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        tarball = Path(tmp) / name
        try:
            _fetch_asset(repo, tag, name, tarball)
        except (urllib.error.URLError, KeyError) as e:
            raise RuntimeError(
                f"cannot download {name} from {repo} release {tag} ({e}). Check your network "
                "(GH_TOKEN helps with GitHub rate limits), or set LIVEKIT_WAKEWORD_ENCODERS to an "
                "unpacked copy (standard/ and micro/)"
            ) from e
        return install_encoders(tarball, sha256, root)


def load_pretrained(tier: Tier = "standard") -> PretrainedEncoder:
    """The released pretrained encoder for `tier`."""
    from safetensors.torch import load_file

    d = encoder_dir(tier)
    enc = build(json.loads((d / "config.json").read_text()))
    enc.load_state_dict(load_file(str(d / "model.safetensors")))
    return enc.eval()


def save_pretrained(enc: PretrainedEncoder, out: str | Path, tier: Tier) -> Path:
    """Write the layout `load_pretrained` reads."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"tier": tier, "top": enc.top, "text_mode": enc.phrase.mode, "format": 1}
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    state = {k: v.contiguous() for k, v in enc.state_dict().items()}
    save_file(state, str(out / "model.safetensors"))
    return out
