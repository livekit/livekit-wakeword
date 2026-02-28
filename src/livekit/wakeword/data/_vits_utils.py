"""Vendored VITS utilities from piper_train.vits.commons.

Only the functions needed for inference-time generation are included.
Source: https://github.com/rhasspy/piper (piper_train/vits/commons.py)
License: MIT
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Create boolean mask from sequence lengths."""
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build monotonic alignment path from duration predictions.

    Args:
        duration: Shape ``[b, 1, t_x]``.
        mask: Shape ``[b, 1, t_y, t_x]``.
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).type_as(mask)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def slerp(
    v1: torch.Tensor,
    v2: torch.Tensor,
    t: float,
    dot_thr: float = 0.9995,
    zdim: int = -1,
) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two tensors.

    Args:
        v1: First vector.
        v2: Second vector.
        t: Interpolation weight in ``[0, 1]``.
        dot_thr: When dot product exceeds this, fall back to linear interp.
        zdim: Feature dimension for norms and angles.
    """
    v1_norm = v1 / torch.norm(v1, dim=zdim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=zdim, keepdim=True)
    dot = (v1_norm * v2_norm).sum(zdim)

    if (torch.abs(dot) > dot_thr).any():
        return (1 - t) * v1 + t * v2

    theta = torch.acos(dot)
    theta_t = theta * t
    sin_theta = torch.sin(theta)
    sin_theta_t = torch.sin(theta_t)

    s1 = torch.sin(theta - theta_t) / sin_theta
    s2 = sin_theta_t / sin_theta

    return (s1.unsqueeze(zdim) * v1) + (s2.unsqueeze(zdim) * v2)


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range."""
    audio_norm = audio * (max_wav_value / max(0.01, float(np.max(np.abs(audio)))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    return audio_norm.astype(np.int16)
