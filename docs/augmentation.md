# Augmentation Pipeline

The augmentation stage applies realistic audio transformations to synthetic TTS clips and aligns them within detection windows.

**Source:** `src/livekit/wakeword/data/augment.py`
**CLI:** `livekit-wakeword augment <config>`

## Overview

```
Original TTS clips (clip_000000.wav)
    │
    ▼  Round 0: reads originals
    ├──► Per-sample augmentations (EQ, distortion)
    ├──► RIR convolution
    ├──► Background mixing
    ├──► Alignment
    └──► clip_000000_r0.wav
              │
              ▼  Round 1: reads r0 output (stacks)
              ├──► Per-sample augmentations
              ├──► RIR convolution
              ├──► Background mixing
              └──► clip_000000_r1.wav
                        │
                        ▼  ... Round N reads r(N-1)
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

`mix_with_background(audio, snr_db_range=(5.0, 15.0))` mixes audio with a random background noise clip at a randomly selected SNR within the given range.

The background clip is looped (tiled) if shorter than the audio and randomly cropped to a starting position. The mixing formula scales the background based on:

```
scale = sqrt(audio_power / (background_power * 10^(snr_db / 10)))
output = audio + scale * background
```

> **Note:** Background noise files serve double duty — they are used here as augmentation overlays *and* also generated as standalone background clips during the [data generation step](data-generation.md#background-noise-clip-generation). Those background clips then pass through this same augmentation pipeline.

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

The augmentation pipeline runs `config.augmentation.rounds` passes over all six directories (positive train/test, negative train/test, background train/test). Each round writes to a separate file (`clip_000000_r0.wav`, `clip_000000_r1.wav`, etc.) — originals are never modified.

Rounds **stack**: round 0 reads the clean TTS originals, round 1 reads round 0's output, round 2 reads round 1's output, and so on. This produces progressively more degraded audio as augmentation effects compound across rounds. Old augmented files (`_rN.wav`) are cleaned up at the start of each run so re-running is idempotent.

## Per-Clip Processing Order

For each WAV file in a directory:

1. Read audio, convert to float32, take first channel if stereo
2. Apply per-sample augmentations (EQ, distortion)
3. Apply RIR convolution (50% probability)
4. Mix with background noise
5. Align to window — round 0 only (end-aligned for positives, center-padded for negatives)
6. Write to `clip_NNNNNN_r{round}.wav` (originals preserved)

## Output

After augmentation:

```
output/<model_name>/
├── positive_train/
│   ├── clip_000000.wav             # Original TTS (preserved, not used for training)
│   ├── clip_000000_r0.wav          # Round 0 augmented
│   ├── clip_000000_r1.wav          # Round 1 (stacked on r0)
│   └── ...
├── positive_test/
├── negative_train/
├── negative_test/
├── background_train/
└── background_test/
```

Only `_rN.wav` files are fed to feature extraction — clean TTS originals are excluded from training since they don't match real microphone audio.

Feature extraction is a separate step — see [Feature Extraction](feature-extraction.md).
