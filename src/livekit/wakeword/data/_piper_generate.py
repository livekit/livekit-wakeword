"""Vendored and adapted VITS generation from dscripka/piper-sample-generator.

Generates synthetic speech with SLERP speaker blending across 904 speakers.
Source: https://github.com/dscripka/piper-sample-generator (generate_samples.py)
License: MIT

Adaptations from dscripka's original:
- Import from local ``_vits_utils`` instead of ``piper_train.vits.commons``
- Use ``espeak-ng`` CLI for phonemization (cross-platform, no C binding issues)
- Add MPS device support (Apple Silicon)
- Load via state_dict + config JSON (no pickle, no piper_train dependency)
- Add type annotations for mypy strict mode
- Remove argparse CLI, auto_reduce_batch_size, file_names params
"""

from __future__ import annotations

import itertools as it
import json
import logging
import shutil
import subprocess
import unicodedata
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from ..utils import get_device
from ._vits.models import SynthesizerTrn
from ._vits_utils import audio_float_to_int16, generate_path, sequence_mask, slerp

logger = logging.getLogger(__name__)


def _load_vits_model(model_path: Path, device: torch.device) -> SynthesizerTrn:
    """Load VITS SynthesizerTrn from state_dict + config JSON.

    Expects two files:
    - ``model_path`` — state_dict saved with ``torch.save(model.state_dict(), ...)``
    - ``model_path.with_suffix(".json")`` — config with a ``"synthesizer"`` key
      containing the ``SynthesizerTrn`` constructor kwargs.
    """
    config_path = model_path.with_suffix(".json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    synth_config: dict[str, Any] = config["synthesizer"]
    model = SynthesizerTrn(**synth_config)

    # The state_dict was saved with weight_norm removed on dec (Generator)
    # and flow, but NOT on enc_q.  Match that before loading.
    model.dec.remove_weight_norm()
    for flow in model.flow.flows:
        remove_fn = getattr(flow, "remove_weight_norm", None)
        if remove_fn is not None:
            remove_fn()

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device)


def _find_espeak_ng() -> str:
    """Find the espeak-ng binary, raising if not found."""
    path = shutil.which("espeak-ng")
    if path is None:
        raise FileNotFoundError(
            "espeak-ng not found. Install it:\n"
            "  macOS:  brew install espeak-ng\n"
            "  Linux:  sudo apt install espeak-ng"
        )
    return path


def _espeak_phonemize(text: str, voice: str = "en-us") -> str:
    """Phonemize text using the espeak-ng CLI.

    This avoids the ``espeak_phonemizer`` C binding which has
    ``libespeak-ng.so`` vs ``.dylib`` issues on macOS.
    """
    espeak = _find_espeak_ng()
    result = subprocess.run(
        [espeak, "--ipa", "-q", "-v", voice, text],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _consume(iterator: Any, n: int) -> None:
    """Advance *iterator* by *n* steps, discarding values."""
    for _ in range(n):
        next(iterator)


def generate_samples(
    text: list[str],
    output_dir: str | Path,
    max_samples: int | None = None,
    model: str | Path = "",
    batch_size: int = 1,
    slerp_weights: list[float] | None = None,
    length_scales: list[float] | None = None,
    noise_scales: list[float] | None = None,
    noise_scale_ws: list[float] | None = None,
    max_speakers: int | None = None,
    start_index: int = 0,
) -> list[Path]:
    """Generate synthetic speech clips with SLERP speaker blending.

    Cycles through all speaker pairs, blending each pair with SLERP at the
    given weights while varying prosody parameters. Outputs are resampled
    to 16 kHz and silence-trimmed via webrtcvad.

    Args:
        text: Phrases to synthesize.
        output_dir: Directory for output ``.wav`` files.
        max_samples: Stop after this many clips (default: ``len(text)``).
        model: Path to VITS state_dict ``.pt`` file (with matching ``.json`` config).
        batch_size: Inference batch size.
        slerp_weights: SLERP interpolation weights (0=speaker1, 1=speaker2).
        length_scales: Speaking rate multipliers.
        noise_scales: Overall speech variability.
        noise_scale_ws: Phoneme duration variability.
        max_speakers: Cap on speaker IDs to use from the model.
        start_index: Resume generation from this clip index (skips already-generated clips).

    Returns:
        List of paths to generated ``.wav`` files.
    """
    if slerp_weights is None:
        slerp_weights = [0.5]
    if length_scales is None:
        length_scales = [0.75, 1.0, 1.25]
    if noise_scales is None:
        noise_scales = [0.667]
    if noise_scale_ws is None:
        noise_scale_ws = [0.8]

    if max_samples is None:
        max_samples = len(text)

    # Validate espeak-ng is available before doing anything expensive
    _find_espeak_ng()

    device = get_device()

    logger.debug("Loading VITS model from %s", model)
    model_path = Path(model)
    vits_model = _load_vits_model(model_path, device)
    logger.info("VITS model loaded on %s", device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = model_path.with_suffix(".json")
    with open(config_path, encoding="utf-8") as config_file:
        config: dict[str, Any] = json.load(config_file)

    voice: str = config["espeak"]["voice"]
    num_speakers: int = config["num_speakers"]
    if max_speakers is not None:
        num_speakers = min(num_speakers, max_speakers)

    # 22050 → 16000 Hz resampler (Kaiser best params from torchaudio tutorial)
    resampler = torchaudio.transforms.Resample(
        22050,
        16000,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

    settings_iter = it.cycle(it.product(slerp_weights, length_scales, noise_scales, noise_scale_ws))
    speakers_iter = it.cycle(it.product(range(num_speakers), range(num_speakers)))
    texts_iter = it.cycle(text)

    # Advance cyclic iterators to maintain speaker/text diversity on resume
    # speakers and texts are consumed once per sample; settings once per batch
    if start_index > 0:
        logger.info("Resuming generation from clip %d / %d", start_index, max_samples)
        _consume(settings_iter, (start_index + batch_size - 1) // batch_size)
        _consume(speakers_iter, start_index)
        _consume(texts_iter, start_index)

    from tqdm import tqdm

    generated: list[Path] = []
    sample_idx = start_index
    failed_batches = 0
    max_consecutive_failures = 5
    consecutive_failures = 0
    pbar = tqdm(total=max_samples, initial=start_index, desc="Synthesizing clips", unit="clip")

    while sample_idx < max_samples:
        speakers_batch = list(it.islice(speakers_iter, batch_size))
        if not speakers_batch:
            break

        current_batch_size = len(speakers_batch)
        sw, ls, ns, nsw = next(settings_iter)

        try:
            with torch.no_grad():
                # Consume texts for this batch (must happen even on failure to keep iterator in sync)
                batch_texts = [next(texts_iter) for _ in range(current_batch_size)]

                phoneme_ids = [
                    get_phonemes(config, t, voice) for t in batch_texts
                ]
                phoneme_lengths = [len(ids) for ids in phoneme_ids]
                phoneme_ids = _right_pad_lists(phoneme_ids)

                speaker_1 = _to_device(torch.LongTensor([s[0] for s in speakers_batch]), device)
                speaker_2 = _to_device(torch.LongTensor([s[1] for s in speakers_batch]), device)

                audio = _generate_audio(
                    vits_model,
                    speaker_1,
                    speaker_2,
                    phoneme_ids,
                    phoneme_lengths,
                    sw,
                    ns,
                    nsw,
                    ls,
                    device,
                )

                # Resample 22050 → 16000
                audio_16k = resampler(audio.cpu()).numpy()
                audio_int16 = audio_float_to_int16(audio_16k)

                for audio_idx in range(audio_int16.shape[0]):
                    if sample_idx >= max_samples:
                        break
                    trimmed = remove_silence(audio_int16[audio_idx].flatten())
                    wav_path = output_path / f"clip_{sample_idx:06d}.wav"
                    with wave.open(str(wav_path), "wb") as wav_file:
                        wav_file.setframerate(16000)
                        wav_file.setsampwidth(2)
                        wav_file.setnchannels(1)
                        wav_file.writeframes(trimmed.tobytes())
                    generated.append(wav_path)
                    sample_idx += 1
                    pbar.update(1)

            consecutive_failures = 0
            logger.debug("Batch complete — %d / %d clips generated", sample_idx, max_samples)

        except Exception as e:
            failed_batches += 1
            consecutive_failures += 1
            logger.warning(
                "Batch failed (texts=%r): %s. Skipping %d clips.",
                batch_texts, e, current_batch_size,
            )
            # Advance sample_idx past this batch so we don't retry the same indices
            sample_idx += current_batch_size
            pbar.update(current_batch_size)

            if consecutive_failures >= max_consecutive_failures:
                pbar.close()
                raise RuntimeError(
                    f"{consecutive_failures} consecutive batches failed. "
                    f"Last error: {e}"
                ) from e

    pbar.close()

    expected = max_samples - start_index
    if failed_batches > 0:
        logger.warning(
            "%d/%d batches failed — generated %d/%d clips",
            failed_batches, failed_batches + len(generated), len(generated), expected,
        )

    if len(generated) == 0 and expected > 0:
        raise RuntimeError(
            f"Generated 0/{expected} clips — all batches failed. "
            "Check espeak-ng installation and input phrases."
        )

    logger.info("Generated %d clips in %s", len(generated), output_path)
    return generated


def _right_pad_lists(lists: list[list[int]]) -> list[list[int]]:
    """Right-pad phoneme ID lists to equal length (pad token = 1 / '^')."""
    max_length = max(len(lst) for lst in lists)
    return [lst + [1] * (max_length - len(lst)) for lst in lists]


def _generate_audio(
    model: Any,
    speaker_1: torch.Tensor,
    speaker_2: torch.Tensor,
    phoneme_ids: list[list[int]],
    phoneme_lengths: list[int],
    slerp_weight: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Run a single VITS forward pass with SLERP-blended speaker embedding."""
    x = _to_device(torch.LongTensor(phoneme_ids), device)
    x_lengths = _to_device(torch.LongTensor(phoneme_lengths), device)

    x_enc, m_p_orig, logs_p_orig, x_mask = model.enc_p(x, x_lengths)
    emb0 = model.emb_g(speaker_1)
    emb1 = model.emb_g(speaker_2)
    g = slerp(emb0, emb1, slerp_weight).unsqueeze(-1)

    if model.use_sdp:
        logw = model.dp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
        logw = model.dp(x_enc, x_mask, g=g)

    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(sequence_mask(y_lengths, int(y_lengths.max().item())), 1).type_as(
        x_mask
    )
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p_orig.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p_orig.transpose(1, 2)).transpose(1, 2)

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = model.flow(z_p, y_mask, g=g, reverse=True)
    audio: torch.Tensor = model.dec((z * y_mask), g=g)
    return audio


def get_phonemes(config: dict[str, Any], text: str, voice: str = "en-us") -> list[int]:
    """Convert text to phoneme IDs using espeak-ng CLI.

    Args:
        config: VITS model config dict (must contain ``phoneme_id_map``).
        text: Input text to phonemize.
        voice: espeak-ng voice name.

    Returns:
        List of integer phoneme IDs.
    """
    phonemes_str = _espeak_phonemize(text, voice)
    phonemes = list(unicodedata.normalize("NFD", phonemes_str))

    id_map: dict[str, list[int]] = config["phoneme_id_map"]
    phoneme_ids: list[int] = list(id_map["^"])
    for phoneme in phonemes:
        p_ids = id_map.get(phoneme)
        if p_ids is not None:
            phoneme_ids.extend(p_ids)
            phoneme_ids.extend(id_map["_"])
    phoneme_ids.extend(id_map["$"])
    return phoneme_ids


def remove_silence(
    x: np.ndarray,
    frame_duration: float = 0.030,
    sample_rate: int = 16000,
    min_start: int = 2000,
) -> np.ndarray:
    """Trim silence from audio using WebRTC VAD.

    Args:
        x: Audio samples (int16 or float).
        frame_duration: VAD frame length in seconds.
        sample_rate: Audio sample rate.
        min_start: Always keep this many leading samples.

    Returns:
        Trimmed int16 audio array.
    """
    import webrtcvad

    vad = webrtcvad.Vad(0)
    if x.dtype in (np.float32, np.float64):
        x = (x * 32767).astype(np.int16)

    x_new = x[:min_start].tolist()
    step_size = int(sample_rate * frame_duration)
    for i in range(min_start, x.shape[0] - step_size, step_size):
        if vad.is_speech(x[i : i + step_size].tobytes(), sample_rate):
            x_new.extend(x[i : i + step_size].tolist())

    result = np.array(x_new, dtype=np.int16)

    # If VAD stripped too much, return the original — don't produce near-silent clips
    min_speech_samples = int(sample_rate * 0.15)  # at least 150ms of content
    if len(result) <= min_start + min_speech_samples:
        logger.debug("VAD stripped too aggressively (%d samples left), keeping original", len(result))
        if x.dtype != np.int16:
            x = (x * 32767).astype(np.int16)
        return x

    return result
