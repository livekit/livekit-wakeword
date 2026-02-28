# Feature Extraction Pipeline

Feature extraction converts augmented audio clips into fixed-size embedding arrays using two frozen ONNX models: a mel-spectrogram frontend and a speech-embedding CNN.

**Source:** `src/livekit/wakeword/data/features.py`, `src/livekit/wakeword/models/feature_extractor.py`

## Overview

```
.wav files (16kHz, ~2s)
    │
    ▼
MelSpectrogramFrontend (ONNX)
    │  (batch, time_frames, 32)
    ▼
SpeechEmbedding (ONNX)
    │  window=76, stride=8
    │  (batch, n_windows, 96)
    ▼
Take last 16 timesteps (or left-pad)
    │  (batch, 16, 96)
    ▼
Save as .npy
```

## MelSpectrogramFrontend

**Source:** `src/livekit/wakeword/models/feature_extractor.py`
**ONNX model:** `melspectrogram.onnx` (from openWakeWord)

Converts raw audio to normalized mel-spectrogram features.

### What is a Mel Spectrogram?

A mel spectrogram is a visual representation of audio that mimics how humans perceive sound. The calculation proceeds in stages:

1. **Short-Time Fourier Transform (STFT)**: The audio is sliced into overlapping windows, and FFT is applied to each window to extract frequency content.

2. **Power Spectrogram**: Square the magnitude of the FFT output to get energy at each frequency bin.

3. **Mel Filterbank**: Apply triangular filters spaced on the **mel scale**—a perceptual scale where equal distances sound equally spaced to humans. Low frequencies get finer resolution (where speech formants live), high frequencies are coarser. The mel scale approximates: `mel = 2595 * log10(1 + f/700)`.

4. **Log Compression**: Convert to decibels (`10 * log10`), mimicking how human ears perceive loudness logarithmically.

5. **Normalization**: Clip to 80dB dynamic range and scale to match the pretrained model's expected input range.

### Parameters

| Parameter            | Value                   |
| -------------------- | ----------------------- |
| Sample rate          | 16,000 Hz               |
| FFT size (`n_fft`)   | 512                     |
| Hop length           | 160 samples (10ms)      |
| Window length        | 400 samples (25ms)      |
| Mel bands (`n_mels`) | 32                      |
| Frequency range      | 60–3,800 Hz             |
| Center padding       | Disabled                |
| Power                | 2.0 (power spectrogram) |

### How Time Frames Are Calculated

The number of output mel frames depends on the audio length and STFT parameters:

```
time_frames = floor((samples - window_length) / hop_length) + 1
            = floor((samples - 400) / 160) + 1
```

**Example**: 2 seconds of audio at 16kHz = 32,000 samples:

```
time_frames = (32000 - 400) / 160 + 1 = 198 frames
```

Each frame represents 10ms of audio (the hop length). Center padding is disabled, so the first frame starts at sample 0.

### Normalization

The mel spectrogram is converted to decibels and normalized:

```
mel_db = 10 * log10(mel_power)         # power_to_db
mel_db = clip(mel_db, max - 80, max)   # top_db = 80
output = mel_db / 10 + 2               # match openWakeWord scaling
```

### Input/Output

- **Input:** `(batch, samples)` or `(samples,)` — float32 audio at 16kHz
- **Output:** `(batch, time_frames, 32)` — normalized mel features

### Torchaudio Fallback

If the ONNX model file is not found, `MelSpectrogramFrontend` falls back to `torchaudio.transforms.MelSpectrogram` with matching parameters. The ONNX model was built with torchlibrosa (not torchaudio), so slight numerical differences exist in the fallback path.

## SpeechEmbedding

**Source:** `src/livekit/wakeword/models/feature_extractor.py`
**ONNX model:** `embedding_model.onnx` (Google speech_embedding)

Converts mel-spectrogram windows into 96-dimensional embedding vectors.

### Model Architecture

Google's `speech_embedding` CNN (~330k parameters), originally from TensorFlow Hub (`google/speech_embedding/1`). The model uses a 5-block architecture:

- Separable convolutions (1x3 depthwise + 3x1 pointwise)
- BatchNorm + LeakyReLU
- MaxPool for downsampling

### Input/Output

- **Input:** `(batch, 76, 32, 1)` — channels-last mel window (76 frames of 32 mels)
- **Output:** `(batch, 96)` — 96-dimensional embedding (squeezed from `(batch, 1, 1, 96)`)

### Sliding Window Extraction

`extract_embeddings()` processes a full mel spectrogram with a sliding window:

| Parameter     | Default | Description                     |
| ------------- | ------- | ------------------------------- |
| `window_size` | 76      | Mel frames per window           |
| `stride`      | 8       | Hop between windows (~80ms)     |
| `batch_size`  | 64      | Samples per ONNX inference call |

### How n_windows Is Calculated

The speech embedding model requires exactly 76 mel frames as input (representing 760ms of audio context). To process longer audio, a sliding window extracts overlapping chunks:

```
n_windows = floor((time_frames - window_size) / stride) + 1
          = floor((time_frames - 76) / 8) + 1
```

**Example**: With 198 mel frames from 2 seconds of audio:

```
n_windows = (198 - 76) / 8 + 1 = 16.25 → 16 windows
```

Each window produces one 96-dimensional embedding vector. The stride of 8 frames = 80ms between consecutive embeddings.

### Output

- `(batch, n_windows, 96)` — one 96-dim embedding per sliding window position
- Returns `(batch, 0, 96)` if the mel spectrogram has fewer than 76 frames

## Timestep Selection

After extracting embeddings, each clip's embedding sequence is normalized to exactly 16 timesteps:

- **16 or more windows:** Take the last 16: `embeddings[-16:]`
- **Fewer than 16 windows:** Left-pad with zeros to reach 16

The constant `N_EMBEDDING_TIMESTEPS = 16` is defined in `features.py`.

### Why ~2 Seconds?

The classifier requires exactly 16 embedding timesteps. Working backwards from this requirement:

```
To get 16 windows:
  (time_frames - 76) / 8 + 1 = 16
  time_frames = 76 + 8 * 15 = 196 frames

To get 196 mel frames:
  samples = (196 - 1) * 160 + 400 = 31,600 samples
  duration = 31,600 / 16000 = ~1.98 seconds
```

So **~2 seconds is the minimum audio length to produce exactly 16 embedding windows** without any padding.

| Audio Duration | Mel Frames | Windows | Action              |
| -------------- | ---------- | ------- | ------------------- |
| 2.0s           | ~198       | 16      | Use last 16         |
| 2.5s           | ~248       | 22      | Crop to last 16     |
| 1.5s           | ~148       | 10      | Left-pad with 6 zeros |
| 0.5s           | ~48        | 0       | All 16 are zeros    |

### Why Left-Padding?

Left-padding places zeros at the **beginning** of the sequence, so the real audio features occupy the **end** (most recent positions). This design choice has important implications:

1. **Causal alignment**: The model learns that the pattern "silence → wake word" corresponds to a detection. Real audio naturally fills from the right as new frames arrive during streaming inference.

2. **Streaming compatibility**: At inference time, you're always asking "did the wake word just happen in the most recent audio?" Having the actual audio at the end of the sequence matches this temporal expectation.

3. **Consistent anchor point**: Regardless of clip length, the wake word utterance is always anchored to the end of the feature sequence, giving the classifier a consistent reference point.

## Feature Extraction for Training

`run_extraction(config)` processes all four data splits:

| Clip Directory    | Output File                   |
| ----------------- | ----------------------------- |
| `positive_train/` | `positive_features_train.npy` |
| `positive_test/`  | `positive_features_test.npy`  |
| `negative_train/` | `negative_features_train.npy` |
| `negative_test/`  | `negative_features_test.npy`  |

Each `.npy` file contains a float32 array of shape `(N_clips, 16, 96)`.

Audio files are read via `soundfile`, converted to float32, reduced to mono if stereo, and processed one clip at a time.

## Memory-Mapped Dataset

**Source:** `src/livekit/wakeword/data/dataset.py`

The training pipeline loads `.npy` features via memory mapping for efficient access without loading entire arrays into memory.

### mmap_batch_generator

```python
mmap_batch_generator(
    data_files: dict[str, Path],       # {"class_name": path_to_npy}
    n_per_class: dict[str, int],       # {"class_name": samples_per_batch}
    label_funcs: dict[str, Callable]   # {"class_name": label_function}
)
```

- Memory-maps all `.npy` files in read-only mode (`mmap_mode="r"`)
- Yields infinite shuffled batches with specified class composition
- Each batch is a `(features, labels)` tuple with shapes `(batch_size, 16, 96)` and `(batch_size,)`
- Wraps around each class independently using modulo arithmetic

### WakeWordDataset

A PyTorch `IterableDataset` wrapper around `mmap_batch_generator` that converts numpy arrays to torch tensors.

### Default Batch Composition

From `WakeWordConfig.batch_n_per_class`:

| Class                  | Samples per Batch |
| ---------------------- | ----------------- |
| `positive`             | 50                |
| `adversarial_negative` | 50                |
| `ACAV100M_sample`      | 1024              |

Total batch size: 1,124 samples (50 positive + 50 adversarial + 1,024 general negative).
