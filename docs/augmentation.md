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

## Parallel Execution (`n_workers`)

The per-clip loop in `_augment_directory` is a pure Python `for` over `soundfile.read`, `scipy.signal.fftconvolve`, and audiomentations transforms. Because of the GIL, adding CPU cores to the process does nothing on its own — each clip is processed sequentially on a single core. On a 32-CPU host, augmenting a 25k-clip dataset this way takes ~3 hours even though the work is embarrassingly parallel.

`AugmentationConfig.n_workers` opts into a `multiprocessing.Pool` that runs the loop across worker processes. Each worker constructs its own `AudioAugmentor` via the pool's `initializer` callback — the parent's lazy-loaded audiomentations instance is never pickled, which keeps the setup robust even as upstream transforms evolve.

Measured on a 32-CPU Modal container augmenting a 60k-clip dataset (25k positive_train + 5k positive_test + 25k negative_train + ~5k negative_test + ~2.5k backgrounds) end-to-end in **~6 minutes**:

| Split | Throughput | Wall-clock |
|---|---|---|
| `positive_train` (25k) | 178 clips/sec | 2:20 |
| `positive_test` (5k) | 174 clips/sec | 0:28 |
| `negative_train` (25k) | 130 clips/sec | 3:12 |
| `negative_test` (~5k) | 91 clips/sec | 0:53 |
| `background_train` (2k) | 83 clips/sec | 0:24 |
| `background_test` (500) | 62 clips/sec | 0:08 |

For reference, the single-threaded path on the same host processes ~2.3 clips/sec, so the full 60k dataset would otherwise take ~7 hours.

Semantics:

- `n_workers: 1` (default) — the legacy single-threaded code path, unchanged.
- `n_workers: 0` — auto, uses `os.cpu_count()`.
- `n_workers: N` (any positive integer) — explicit worker count.

`mp_context` controls the start method: `"auto"` picks `fork` on Linux/macOS and `spawn` on Windows. Override only if a fork-unsafe audio backend is crashing workers.

Output file names, round-0 alignment, padding, and RIR / background mixing behave identically to the single-threaded path. Per-worker random state means the *exact* audio content differs run-to-run across paths (different SNR draws, different RIR picks), but the output shape, count, and naming are byte-for-byte the same — which is what the downstream feature extractor depends on.
