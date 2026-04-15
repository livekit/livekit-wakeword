"""VoxCPM2 TTS backend: voice-design diversification (persona × cfg × diffusion steps)."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...config import WakeWordConfig

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000


def diversification_triple_at_index(
    prompts: list[str],
    cfg_values: list[float],
    timesteps: list[int],
    index: int,
) -> tuple[str, float, int]:
    """Return the (prompt, cfg, steps) triple for global clip *index* (resume-safe).

    Ordering matches ``itertools.product(prompts, cfg_values, timesteps)``:
    innermost dimension is *timesteps*, then *cfg_values*, then *prompts*.
    """
    np_ = len(prompts)
    nc = len(cfg_values)
    nt = len(timesteps)
    n = np_ * nc * nt
    if n == 0:
        raise ValueError("voxcpm diversification lists must be non-empty")
    flat = index % n
    ti = flat % nt
    flat //= nt
    ci = flat % nc
    pi = flat // nc
    return prompts[pi], cfg_values[ci], timesteps[ti]


class VoxCpmBackend:
    """VoxCPM2 with strong default diversification; loads weights from local snapshot only."""

    def __init__(
        self,
        *,
        model_dir: Path,
        load_denoiser: bool,
        voice_design_prompts: list[str],
        cfg_values: list[float],
        inference_timesteps_list: list[int],
    ) -> None:
        self._model_dir = model_dir
        self._load_denoiser = load_denoiser
        self._prompts = voice_design_prompts
        self._cfg_values = cfg_values
        self._timesteps = inference_timesteps_list
        self._model: Any = None

    @classmethod
    def from_config(cls, config: WakeWordConfig) -> VoxCpmBackend:
        vt = config.voxcpm_tts
        return cls(
            model_dir=config.voxcpm_local_model_path,
            load_denoiser=vt.load_denoiser,
            voice_design_prompts=list(vt.voice_design_prompts),
            cfg_values=list(vt.cfg_values),
            inference_timesteps_list=list(vt.inference_timesteps_list),
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if importlib.util.find_spec("voxcpm") is None:
            raise ImportError(
                "VoxCPM is not installed. Install with: uv sync --extra train --extra voxcpm"
            )
        voxcpm_mod = importlib.import_module("voxcpm")
        VoxCPM = getattr(voxcpm_mod, "VoxCPM")
        logger.info("Loading VoxCPM from %s", self._model_dir)
        self._model = VoxCPM.from_pretrained(
            str(self._model_dir),
            load_denoiser=self._load_denoiser,
        )
        return self._model

    def validate_artifacts(self) -> None:
        if not self._model_dir.is_dir():
            raise FileNotFoundError(
                f"VoxCPM model directory not found: {self._model_dir}. "
                "Run: livekit-wakeword setup --config <your.yaml>"
            )
        if not any(self._model_dir.iterdir()):
            raise FileNotFoundError(
                f"VoxCPM model directory is empty: {self._model_dir}. "
                "Run: livekit-wakeword setup --config <your.yaml>"
            )
        if not self._prompts or not self._cfg_values or not self._timesteps:
            raise ValueError(
                "voxcpm_tts.voice_design_prompts, cfg_values, and "
                "inference_timesteps_list must be non-empty"
            )
        if importlib.util.find_spec("voxcpm") is None:
            raise ImportError(
                "VoxCPM is not installed. Install with: uv sync --extra train --extra voxcpm"
            )

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
    ) -> list[Path]:
        del batch_size  # sequential generation only
        if not phrases:
            raise ValueError("phrases must be non-empty")
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._ensure_model()
        src_sr = int(model.tts_model.sample_rate)

        import librosa
        import soundfile as sf  # type: ignore[import-untyped]
        from tqdm import tqdm  # type: ignore[import-untyped]

        generated: list[Path] = []
        pbar = tqdm(
            range(start_index, n_samples),
            desc="VoxCPM clips",
            unit="clip",
            initial=start_index,
            total=n_samples,
        )
        for sample_idx in pbar:
            phrase = phrases[sample_idx % len(phrases)]
            prompt, cfg_v, steps = diversification_triple_at_index(
                self._prompts,
                self._cfg_values,
                self._timesteps,
                sample_idx,
            )
            text = f"({prompt}){phrase}"
            try:
                wav = model.generate(
                    text=text,
                    cfg_value=cfg_v,
                    inference_timesteps=steps,
                )
            except Exception as e:
                logger.warning("VoxCPM generate failed at clip %d: %s", sample_idx, e)
                continue

            audio = np.asarray(wav, dtype=np.float32).flatten()
            if audio.size == 0:
                logger.warning("VoxCPM returned empty audio at clip %d", sample_idx)
                continue

            if src_sr != TARGET_SAMPLE_RATE:
                audio = librosa.resample(
                    audio,
                    orig_sr=src_sr,
                    target_sr=TARGET_SAMPLE_RATE,
                )
            peak = float(np.max(np.abs(audio))) or 1.0
            audio_i16 = (audio * (32767.0 / peak)).astype(np.int16)

            out_path = output_dir / f"clip_{sample_idx:06d}.wav"
            sf.write(str(out_path), audio_i16, TARGET_SAMPLE_RATE)
            generated.append(out_path)

        logger.info("Generated %d clips in %s", len(generated), output_dir)
        return generated
