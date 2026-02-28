# Architecture Overview

livekit-wakeword uses a hybrid ONNX + PyTorch architecture. Two frozen ONNX models handle feature extraction (mel spectrogram and speech embeddings), while a lightweight PyTorch classifier head is trained per wake word.

## System Architecture

**Training Pipeline:** Synthetic speech is generated via VITS TTS with SLERP speaker blending, augmented with noise/reverb, then passed through frozen ONNX models (mel spectrogram → speech embedding) to produce `.npy` feature files. These features train a lightweight classifier head, which is saved as a `.pt` model.

**Inference Pipeline:** Raw 16kHz audio flows through the same frozen ONNX feature extractors (32-band mel spectrogram → 96-dim speech embedding), then through the trained classifier (exported to ONNX) to produce a detection score between 0 and 1.

## Why ONNX + PyTorch?

The mel-spectrogram and speech-embedding models are pre-trained and frozen — they never change between wake words. Running them through ONNX Runtime provides:

- **Fast numpy-based inference** without loading PyTorch at detection time
- **Shared feature extractors** across all wake words
- **Minimal training data** needed since only the small classifier head is trained

The classifier head is trained in PyTorch for flexibility (custom loss, hard example mining, learning rate schedules), then exported to ONNX for deployment.

## Data Flow

### Feature Dimensions at Each Stage

| Stage | Output Shape | Description |
|-------|-------------|-------------|
| Raw audio | `(samples,)` | 16kHz float32 mono |
| MelSpectrogramFrontend | `(batch, time_frames, 32)` | 32 mel bands, ~5 frames per 80ms |
| SpeechEmbedding | `(batch, n_windows, 96)` | 76-frame sliding window, stride 8 |
| Timestep selection | `(batch, 16, 96)` | Last 16 windows (or left-padded) |
| Classifier | `(batch, 1)` | Sigmoid confidence score |

### Mel Spectrogram Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| FFT size | 512 |
| Hop length | 160 (10ms) |
| Window length | 400 (25ms) |
| Mel bands | 32 |
| Frequency range | 60–3800 Hz |
| Centering | Disabled |
| Normalization | `power_to_db / 10 + 2` |

### Speech Embedding Model

Google's `speech_embedding` CNN (~330k parameters), originally from TensorFlow Hub. Input: `(batch, 76, 32, 1)` channels-last mel windows. Output: `(batch, 96)` embedding vectors.

The embedding model uses a 5-block CNN with separable convolutions (1x3 + 3x1), BatchNorm, LeakyReLU, and MaxPool.

## Module Map

```
src/livekit/wakeword/
├── config.py                    Pydantic config models + YAML loading
├── cli.py                       Typer CLI (setup, generate, augment, train, export, run)
├── models/
│   ├── feature_extractor.py     MelSpectrogramFrontend + SpeechEmbedding (ONNX)
│   ├── classifier.py            DNNClassifier, RNNClassifier, build_classifier()
│   └── pipeline.py              WakeWordClassifier (training wrapper for classifier head)
├── data/
│   ├── generate.py              VITS TTS + SLERP speaker blending + adversarial negatives
│   ├── augment.py               AudioAugmentor + clip alignment
│   ├── dataset.py               WakeWordDataset (memory-mapped .npy batch generator)
│   └── features.py              ONNX feature extraction → .npy files
├── training/
│   ├── trainer.py               WakeWordTrainer (3-phase training + checkpoint averaging)
│   └── metrics.py               FPPH, recall, balanced accuracy
├── export/
│   └── onnx.py                  ONNX export + INT8 quantization
└── inference/
    ├── model.py                 WakeWordModel class (simple prediction API)
    └── listener.py              WakeWordListener class (async microphone detection)
```

## Pipeline Stages

1. **[Data Generation](data-generation.md)** — Synthesize positive clips via VITS TTS with SLERP speaker blending + adversarial negatives
2. **[Augmentation](augmentation.md)** — Apply pitch shift, EQ, RIR convolution, background mixing; align clips to training windows
3. **[Feature Extraction](feature-extraction.md)** — Extract mel spectrograms and speech embeddings through frozen ONNX models
4. **[Training](training.md)** — 3-phase adaptive training with hard example mining and checkpoint averaging
5. **[Export & Inference](export-and-inference.md)** — Export classifier to ONNX and run real-time streaming detection
