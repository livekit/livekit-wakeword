# Augmentation Pipeline

The augmentation stage applies realistic audio transformations to synthetic TTS clips and aligns them within detection windows.

**Source:** `src/livekit/wakeword/data/augment.py`
**CLI:** `livekit-wakeword augment <config>`

## Overview

```
Generated .wav clips
    │
    ├──► Per-sample augmentations (audiomentations)
    │    EQ, distortion
    │
    ├──► RIR convolution
    │    Room impulse responses
    │
    ├──► Background mixing
    │    Real-world noise at random SNR
    │
    ├──► Alignment
    │    Positive: end-of-window
    │    Negative: center-padded
```

## AudioAugmentor

The `AudioAugmentor` class manages all audio augmentations.

### Initialization

```python
AudioAugmentor(
    background_paths: list[Path],  # Directories with background noise .wav files
    rir_paths: list[Path],         # Directories with room impulse response .wav files
    sample_rate: int = 16000
)
```

All `.wav` files are collected recursively from the provided directories.

### Per-Sample Augmentations

Applied via the `audiomentations` library to individual clips:

| Transform | Probability | Description |
|-----------|------------|-------------|
| `SevenBandParametricEQ` | 0.25 | 7-band parametric equalizer |
| `TanhDistortion` | 0.25 | Tanh-based distortion |

### RIR Convolution

`apply_rir(audio, p=0.5)` convolves audio with a randomly selected room impulse response using FFT convolution (`scipy.signal.fftconvolve`). The RIR is normalized by its maximum absolute value before convolution. Output is cropped to the original audio length.

### Background Mixing

`mix_with_background(audio, snr_db_range=(5.0, 15.0))` mixes audio with a random background noise clip at a randomly selected SNR within the given range. The batch augmentations use an SNR range of **0–15 dB** for `AddBackgroundNoise` (signal is always at least as loud as the noise).

The background clip is looped (tiled) if shorter than the audio and randomly cropped to a starting position. The mixing formula scales the background based on:

```
scale = sqrt(audio_power / (background_power * 10^(snr_db / 10)))
output = audio + scale * background
```

## Clip Alignment

Positive and negative clips are aligned differently within the target window (default 2.0 seconds = 32,000 samples).

### Positive Clips — End-Aligned

`align_clip_to_end(audio, target_length, jitter_samples=3200)`

Positive clips are placed at the **end** of the window with random jitter of up to 3200 samples (200ms at 16kHz). This simulates the real detection scenario where the wake word appears at the trailing edge of the audio buffer.

```
[    zero padding    |  wake word  | jitter ]
                              ◄── target_length ──►
```

### Negative Clips — Center-Padded

Negative clips are centered within the target window. If longer than the target, they are center-cropped; if shorter, they are center-padded with zeros.

## Augmentation Rounds

The augmentation pipeline runs `config.augmentation.rounds` passes over all four directories (positive train/test, negative train/test). Round 0 overwrites the original `.wav` files in-place. Subsequent rounds write new files (e.g. `clip_000000_r1.wav`) but read from the round-0 (already augmented) files, not the raw TTS output.

Only original clips (`clip_######.wav`) are read as input — augmented variants (`clip_000000_r1.wav`, etc.) are excluded via a regex filter to prevent compounding augmentation effects across rounds.

## Per-Clip Processing Order

For each WAV file in a directory:

1. Read audio, convert to float32, take first channel if stereo
2. Apply per-sample augmentations (EQ, distortion)
3. Apply RIR convolution (50% probability)
4. Mix with background noise
5. Align to window (end-aligned for positives, center-padded for negatives)
6. Write back to the same file path

## Output

After augmentation:

```
output/<model_name>/
├── positive_train/                 # Augmented .wav files
├── positive_test/
├── negative_train/
└── negative_test/
```

Feature extraction is a separate step — see [Feature Extraction](feature-extraction.md).
