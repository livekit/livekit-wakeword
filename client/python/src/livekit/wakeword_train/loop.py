"""Training loop (GPU augmentation on CUDA or MPS, streaming labels, EMA) and score calibration.

Each row is a 2.56 s window scored at its last 8 positions (80 ms apart). In positive rows,
positions up to 0.5 s after the wake word ends are positive and positions more than 0.25 s before
its end are negative; every other row is negative everywhere.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import data as D
from .augment import AugConfig, GpuNoise, augment, robust_aug
from .nn import FRAME, MEL_DEFAULT, SR, Detector

logger = logging.getLogger(__name__)

W = 40960  # 2.56 s window
K = 8  # scored positions per window


def default_device() -> str:
    """CUDA, else Apple silicon (MPS), else CPU (very slow)."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def default_aug(det: nn.Module) -> AugConfig:
    """Augmentation for `det`: encoders with the low log-mel floor also get quiet/device audio."""
    fe = getattr(getattr(det, "encoder", None), "frontend", None)
    # the params are float32 buffers: 1e-6 comes back as 9.99999997e-07, so compare loosely
    if fe is not None and not all(
        math.isclose(float(a), b, rel_tol=1e-5) for a, b in zip(fe.params, MEL_DEFAULT, strict=True)
    ):
        return robust_aug()
    return AugConfig(p_codec=0.25)


@dataclass
class TrainConfig:
    steps: int = 5000
    n_pos: int = 64
    n_adv: int = 16
    n_speech: int = 144
    n_noise: int = 32
    lr_head: float = 1e-3
    lr_encoder: float = 1e-4
    ft_top_blocks: int | None = 2  # fine-tune the last k encoder blocks; 0 freezes, None: all
    kd_weight: float = 1.0  # with a teacher: its scores on the same batch as soft targets
    kd_temp: float = 2.0
    weight_decay: float = 0.01
    warmup: int = 500
    neg_weight: tuple[float, float] = (1.0, 4.0)
    ema: float = 0.999
    p_speed: float = 0.7
    speed: tuple[float, float] = (0.8, 1.2)
    aug: AugConfig = field(default_factory=lambda: AugConfig(p_codec=0.25))
    workers: int = 6
    seed: int = 0


def _speed(
    c: np.ndarray, e: int, rng: np.random.Generator, cfg: TrainConfig
) -> tuple[np.ndarray, int]:
    if cfg.p_speed <= 0 or rng.random() >= cfg.p_speed:
        return c, e
    f = rng.uniform(*cfg.speed)
    n = max(1, int(len(c) / f))
    y = np.interp(np.arange(n) * f, np.arange(len(c)), c.astype(np.float32)).astype(np.int16)
    return y, min(n, int(e / f))


class _Stream(IterableDataset):  # type: ignore[type-arg]
    def __init__(self, dataset: str | list[str], cfg: TrainConfig) -> None:
        self.dataset, self.cfg = dataset, cfg

    def __iter__(self):  # type: ignore[no-untyped-def]
        cfg = self.cfg
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng(cfg.seed * 1000 + wid + int(time.time()) % 100000)
        sets = D.load_clipsets(self.dataset)
        pos, adv, tts = sets["positive"], sets["adversarial"], sets["speech"]
        if cfg.n_adv and not len(adv) and wid == 0:  # e.g. languages without confusable phrases
            logger.warning(
                "dataset %r has no confusable clips (list some under `hard_negatives`): the %d "
                "confusable rows per batch use %s",
                self.dataset,
                cfg.n_adv,
                "its sentence clips" if len(tts) else "background speech",
            )
        files = D.negative_speech_files()
        speech = D.StreamPool(files[wid::nw] or files, seed=wid)
        B = cfg.n_pos + cfg.n_adv + cfg.n_speech + cfg.n_noise
        while True:
            x = np.zeros((B, W), np.int16)
            pend = np.full(B, -1, np.int64)
            r = 0
            for _ in range(cfg.n_pos):
                i = int(rng.integers(len(pos)))
                c, e = _speed(pos.clip(i), int(pos.ends[i]), rng, cfg)
                end_at = W - int(rng.uniform(0.0, 0.6) * SR)
                st = end_at - e
                a, b = max(0, st), min(W, st + len(c))
                x[r, a:b] = c[a - st : b - st]
                pend[r] = end_at
                r += 1
            for _ in range(cfg.n_adv):
                src = tts if len(tts) and (not len(adv) or rng.random() < 0.2) else adv
                if not len(src):  # no confusables and no sentence clips: background speech
                    x[r] = speech.crop(rng, W)
                    r += 1
                    continue
                i = int(rng.integers(len(src)))
                c, e = _speed(src.clip(i), int(src.ends[i]), rng, cfg)
                st = W - int(rng.uniform(-0.3, 1.2) * SR) - e
                a, b = max(0, st), min(W, st + len(c))
                if b > a:
                    x[r, a:b] = c[a - st : b - st]
                r += 1
            for _ in range(cfg.n_speech):
                x[r] = speech.crop(rng, W)
                r += 1
            babble = np.stack([speech.crop(rng, W) for _ in range(B)])
            yield torch.from_numpy(x), torch.from_numpy(pend), torch.from_numpy(babble)


def _labels(
    pend: torch.Tensor, ends: torch.Tensor, pos_s: float = 0.5, margin_s: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    t, p = ends[None].float(), pend[:, None].float()
    ispos = (pend >= 0)[:, None]
    y = ((t >= p) & (t <= p + pos_s * SR)).float()
    valid = torch.where(ispos, (y > 0) | (t < p - margin_s * SR), torch.ones_like(ispos))
    return torch.where(ispos, y, torch.zeros_like(y)), valid.float()


def train(
    det: Detector,
    dataset: str | list[str],
    cfg: TrainConfig,
    device: str | None = None,
    log_every: int = 500,
    teacher: nn.Module | None = None,
) -> Detector:
    """Train `det` on a bench dataset (dir or HF repo; several are merged) and return the EMA
    weights. With `teacher`, its scores on each augmented batch are soft targets (NaN: unscored);
    confusable rows keep only their hard labels."""
    device = device or default_device()
    torch.manual_seed(cfg.seed)
    noise = GpuNoise(np.asarray(D.noise_pool()), device)
    rirs = torch.from_numpy(D.rir_pool()).to(device)
    det.to(device).train()
    top = cfg.ft_top_blocks
    frozen = top is not None and top <= 0
    for p in det.encoder.parameters():
        p.requires_grad_(top is None)
    if frozen:
        det.encoder.eval()
    elif top is not None:
        for m in [*list(det.encoder.blocks)[-top:], det.encoder.out]:
            for p in m.parameters():
                p.requires_grad_(True)
    if teacher is not None:
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    head_params = [*det.head.parameters(), *det.norm.parameters()]
    groups = [{"params": head_params, "lr": cfg.lr_head}]
    enc_params = [p for p in det.encoder.parameters() if p.requires_grad]
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps)))
        ),
    )
    D.load_clipsets(dataset)  # build the clip caches before the data workers start
    ema = copy.deepcopy(det).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    dl = DataLoader(
        _Stream(dataset, cfg),
        batch_size=None,
        num_workers=cfg.workers,
        prefetch_factor=4,
        persistent_workers=cfg.workers > 0,
        timeout=900 if cfg.workers > 0 else 0,
        pin_memory=device == "cuda",
    )
    B = cfg.n_pos + cfg.n_adv + cfg.n_speech + cfg.n_noise
    noise_only = torch.zeros(B, dtype=torch.bool, device=device)
    if cfg.n_noise:
        noise_only[-cfg.n_noise :] = True
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    ends = torch.from_numpy(det.encoder.frame_end_samples(W // FRAME)[-K:]).to(device)
    kd_rows = torch.ones(B, device=device)
    kd_rows[cfg.n_pos : cfg.n_pos + cfg.n_adv] = 0.0  # rows: pos, adv, speech, noise
    t0, hist = time.time(), []
    for step, (x, pend, bab) in enumerate(dl):
        if step >= cfg.steps:
            break
        x = x.to(device, non_blocking=True).float() / 32767
        bab = bab.to(device, non_blocking=True).float() / 32767
        pend = pend.to(device, non_blocking=True)
        xa, _ = augment(x, cfg.aug, noise, rirs, babble=bab, gen=gen, noise_only=noise_only)
        if frozen:
            with torch.no_grad():
                f = det.encoder(xa)
        else:
            f = det.encoder(xa)
        logits = det.head_windows(f, last=K)
        y, valid = _labels(pend, ends)
        wneg = cfg.neg_weight[0] + (cfg.neg_weight[1] - cfg.neg_weight[0]) * step / cfg.steps
        w = valid * torch.where(y > 0, torch.ones_like(y), torch.full_like(y, wneg))
        bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
        loss = (bce * w).sum() / w.sum()
        if teacher is not None:
            with torch.no_grad():
                tl = teacher(xa, last=K).float()
                scored = torch.isfinite(tl).float()
                soft = torch.sigmoid(torch.nan_to_num(tl) / cfg.kd_temp)
            kd = F.binary_cross_entropy_with_logits(
                logits.float() / cfg.kd_temp, soft, reduction="none"
            )
            kd_mask = kd_rows[:, None] * scored
            kd = (kd * kd_mask).sum() / kd_mask.sum().clamp(min=1.0)
            loss = loss + cfg.kd_weight * cfg.kd_temp**2 * kd
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 5.0)
        opt.step()
        sched.step()
        d = min(cfg.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for (k, e), p in zip(ema.state_dict().items(), det.state_dict().values()):
                if e.dtype.is_floating_point and not k.split(".")[-1].startswith("mel_"):
                    e.mul_(d).add_(p.detach(), alpha=1 - d)
                else:  # counters and the fixed log-mel settings
                    e.copy_(p)
        hist.append(loss.detach().item())
        if (step + 1) % log_every == 0:
            logger.info(
                "step %d/%d  loss %.4f  %.2fs/step",
                step + 1,
                cfg.steps,
                np.mean(hist[-log_every:]),
                (time.time() - t0) / (step + 1),
            )
    return ema


# held-out (validation) speech for calibration
CALIBRATION_SPEECH = [
    (
        "google/fleurs",
        [
            f"parquet-data/{x}/validation-*.parquet"
            for x in ("en_us", "es_419", "de_de", "fr_fr", "hi_in", "cmn_hans_cn", "ja_jp", "ar_eg")
        ],
    ),
    (
        "fixie-ai/common_voice_17_0",
        [
            f"{x}/validation-0000[0-1]-*.parquet"
            for x in ("en", "es", "de", "fr", "it", "pt", "ru", "ja")
        ],
    ),
]


def calibration_speech_files() -> list[tuple[str, str]]:
    path = D.CACHE / "calibration_speech_files.json"
    if path.exists():
        return [tuple(x) for x in json.loads(path.read_text())]
    import fnmatch

    from huggingface_hub import HfApi

    out: list[tuple[str, str]] = []
    for repo, globs in CALIBRATION_SPEECH:
        files = sorted(HfApi().list_repo_files(repo, repo_type="dataset"))
        for g in globs:
            out += [(repo, f) for f in files if fnmatch.fnmatch(f, g)][:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


@torch.no_grad()
def calibration_logit(
    det: Detector,
    target_fpph: float = 0.5,
    minutes: float = 120.0,
    device: str | None = None,
    debounce_s: float = 2.0,
    aug: AugConfig | None = None,
    noise_minutes: float = 40.0,
) -> float:
    """Logit threshold giving `target_fpph` false triggers per hour, met separately on held-out
    speech and on noise-only audio, both augmented like training negatives."""
    device = device or default_device()
    det = det.to(device).eval()
    rng = np.random.default_rng(0)
    seg, n_seg = 60 * SR, int(minutes)
    # stopped (joined) before anything forks: workers would inherit its HTTP locks and deadlock
    pool = D.StreamPool(
        calibration_speech_files(),
        buffer_s=minutes * 60,
        min_fill_s=minutes * 60 * 0.9,
        refresh_s=0,
    )
    try:
        crops = np.stack([pool.crop(rng, seg) for _ in range(n_seg)]).astype(np.float32) / 32767
    finally:
        pool.stop(join=True)
    cfg = aug or AugConfig(p_codec=0.25)
    noise = GpuNoise(np.asarray(D.noise_pool()), device)
    rirs = torch.from_numpy(D.rir_pool()).to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    n_noise = int(noise_minutes)
    crops = np.concatenate([crops, np.zeros((n_noise, seg), np.float32)])
    n_all = n_seg + n_noise
    logits, times = [], []
    for k in range(0, n_all, 8):
        x = torch.from_numpy(crops[k : k + 8]).to(device)
        babble = torch.roll(x, 1, 0)
        only = torch.arange(k, k + len(x), device=device) >= n_seg
        xa, _ = augment(x, cfg, noise, rirs, babble=babble, gen=gen, noise_only=only)
        lg = det(xa).float().cpu().numpy()
        for j, row in enumerate(lg):
            logits.append(row)
            times.append((k + j) * 60 + np.arange(len(row)) * FRAME / SR)
    lg, t = np.concatenate(logits), np.concatenate(times)
    speech_part = t < n_seg * 60
    thr = _threshold_at(lg[speech_part], t[speech_part], n_seg / 60, target_fpph, debounce_s)
    if n_noise:
        thr_n = _threshold_at(
            lg[~speech_part], t[~speech_part], n_noise / 60, target_fpph, debounce_s
        )
        thr = max(thr, thr_n)
    return thr


def _threshold_at(
    lg: np.ndarray, t: np.ndarray, hours: float, target_fpph: float, debounce_s: float
) -> float:
    """Lowest logit threshold whose debounced trigger rate stays within `target_fpph`."""
    for thr in np.sort(np.unique(np.round(lg, 2)))[::-1]:
        idx = np.flatnonzero(lg >= thr)
        events, last = 0, -1e9
        for i in idx:
            if t[i] - last >= debounce_s:
                events, last = events + 1, t[i]
        if events / hours > target_fpph:
            return float(thr) + 0.01
    return float(lg.min())
