# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

livekit-wakeword — wake word detection library using frozen ONNX feature extraction with trainable PyTorch classifiers. Hybrid architecture: ONNX mel spectrogram + speech embedding → PyTorch DNN/RNN classifier head.

## Commands

**Always use `uv` for package management. Always use `git` for version separation (branches, commits).**

```bash
# Install
uv sync                              # All deps including optional groups
uv sync --group dev                  # Dev only

# Test
uv run pytest tests/                 # All 32 tests
uv run pytest tests/test_config.py   # Single file
uv run pytest -k "test_name"         # Single test
uv run pytest --cov=src/livekit/wakeword tests/  # With coverage

# Lint & format
uv run ruff check src/ tests/       # Lint (rules: E, F, I, UP)
uv run ruff format src/ tests/      # Auto-format
uv run mypy src/livekit/wakeword/       # Type check (strict mode)

# CLI (entry point: livekit-wakeword = livekit.wakeword.cli:app)
uv run livekit-wakeword setup            # Download frozen ONNX models + VITS TTS checkpoint
uv run livekit-wakeword generate <config> # VITS TTS + SLERP speaker blending + adversarial negatives
uv run livekit-wakeword augment <config>  # Augment + extract features → .npy
uv run livekit-wakeword train <config>    # 3-phase adaptive training
uv run livekit-wakeword export <config>   # Export classifier to ONNX
uv run livekit-wakeword run <config>      # Full pipeline (generate→augment→train→export)
```

## Architecture

### Processing Pipeline (inference)
```
Raw audio (16kHz) → MelSpectrogramFrontend (ONNX) → SpeechEmbedding (ONNX) → Classifier (PyTorch) → [0,1]
                    n_fft=512, hop=160, n_mels=32     76×32×1 → 96-dim         16×96 → 1 score
```

### Source Layout (`src/livekit/wakeword/`)

- **`config.py`** — Pydantic models + YAML loading (`WakeWordConfig.load_config()`)
- **`cli.py`** — Typer CLI with all commands
- **`models/`**
  - `feature_extractor.py` — `MelSpectrogramFrontend` (ONNX primary, torchaudio fallback) and `SpeechEmbedding` (ONNX only)
  - `classifier.py` — `DNNClassifier` (FC+LayerNorm), `RNNClassifier` (Bi-LSTM), `build_classifier()` factory
  - `pipeline.py` — `WakeWordClassifier` (training wrapper for classifier head)
- **`data/`**
  - `generate.py` — VITS TTS synthesis with SLERP speaker blending + adversarial negatives
  - `_piper_generate.py` — Vendored VITS generation from dscripka/piper-sample-generator (904-speaker SLERP)
  - `_vits_utils.py` — Vendored VITS utilities (sequence_mask, generate_path, slerp, audio_float_to_int16)
  - `augment.py` — `AudioAugmentor` (pitch, EQ, RIR, backgrounds); positives aligned to END of window, negatives center-padded
  - `dataset.py` — `WakeWordDataset` (memory-mapped .npy, mixed-class batch generator)
  - `features.py` — Extract features through ONNX pipeline → .npy files
- **`training/`**
  - `trainer.py` — `WakeWordTrainer` with 3-phase training (full → refinement → fine-tuning), hard example mining, adaptive negative weighting, checkpoint averaging
  - `metrics.py` — FPPH (false positives per hour), recall, balanced accuracy
- **`export/onnx.py`** — Export classifier to ONNX with optional INT8 quantization
- **`inference/`**
  - `model.py` — `WakeWordModel` class for simple prediction API
  - `listener.py` — `WakeWordListener` class for async microphone detection

### Key Design Decisions

- **Feature extraction is numpy-based** (ONNX runtime), not torch tensors. Both frozen models (`melspectrogram.onnx`, `embedding_model.onnx`) are bundled with the package via `importlib.resources` (see `resources/__init__.py`).
- **Embedding shape**: always `(batch, 16, 96)` — 16 timesteps of 96-dim vectors. Last 16 steps taken or left-padded.
- **Model sizes** (tiny/small/medium/large) map to `layer_dim` and `n_blocks` in config. Factory: `build_classifier(model_type, model_size)`.
- **Training loss**: BCE with hard example mining (only non-trivial predictions contribute) and linearly increasing negative class weight.
- **Checkpoint averaging**: final model averages top checkpoints by 90th-pct accuracy and 10th-pct FPPH.
- **Config format**: YAML loaded via `WakeWordConfig.load_config(path)`. See `configs/hey_livekit.yaml` for reference.

## Documentation

For detailed documentation on each pipeline stage, see `docs/`:

- [docs/overview.md](docs/overview.md) — Architecture and data flow
- [docs/data-generation.md](docs/data-generation.md) — TTS synthesis and adversarial negatives
- [docs/augmentation.md](docs/augmentation.md) — Audio transforms and alignment
- [docs/feature-extraction.md](docs/feature-extraction.md) — Mel spectrograms and embeddings
- [docs/training.md](docs/training.md) — 3-phase training and checkpoint averaging
- [docs/export-and-inference.md](docs/export-and-inference.md) — ONNX export and Python API

## Code Style

- Python 3.11+, line length 100
- Ruff for linting/formatting, mypy strict mode
- Build system: hatchling, src layout (`src/livekit/wakeword/`)
