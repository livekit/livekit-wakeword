<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# livekit-wakeword

[![CI](https://github.com/livekit/livekit-wakeword/actions/workflows/ci.yml/badge.svg)](https://github.com/livekit/livekit-wakeword/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An open-source wake word library for creating voice-enabled applications. Based on [openWakeWord](https://github.com/dscripka/openWakeWord) with streamlined training — generate synthetic data, augment, train, and export from a single YAML config.

**Features:**

- **Backward compatible** with openWakeWord models and library
- **Train anywhere** — local machine, cloud, or spawn [SkyPilot](https://github.com/skypilot-org/skypilot) jobs
- **Zero dependency headaches** — uv handles everything

**Quick Links:**

- [Using Existing Models](#using-existing-models-and-library)
- [Training New Models](#training-new-models)

## Quick Start

### Using Existing Models and Library

**System dependencies (for microphone listener):**

```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt install portaudio19-dev
```

**Installation:**

```bash
pip install git+https://github.com/livekit/livekit-wakeword.git
# or
uv add git+https://github.com/livekit/livekit-wakeword
```

**Basic inference:**

```python
from livekit.wakeword import WakeWordModel

model = WakeWordModel(models=["hey_livekit.onnx"])

# Feed audio frames (16kHz, int16 or float32)
scores = model.predict(audio_frame)
if scores["hey_livekit"] > 0.5:
    print("Wake word detected!")
```

**Async listener with microphone:**

```python
import asyncio
from livekit.wakeword import WakeWordModel, WakeWordListener

model = WakeWordModel(models=["hey_livekit.onnx"])

async def main():
    async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
        while True:
            detection = await listener.wait_for_detection()
            print(f"Detected {detection.name}! ({detection.confidence:.2f})")

asyncio.run(main())
```

### Training New Models

**System dependencies:**

```bash
# macOS
brew install espeak-ng ffmpeg portaudio

# Ubuntu/Debian
sudo apt install espeak-ng libsndfile1 ffmpeg sox portaudio19-dev
```

**Installation:**

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/livekit/livekit-wakeword
cd livekit-wakeword
uv sync --all-extras
```

**Download models and data:**

```bash
uv run livekit-wakeword setup
```

**Train a wake word:**

```bash
uv run livekit-wakeword run configs/hey_livekit.yaml
```

Or run stages individually:

```bash
uv run livekit-wakeword generate configs/hey_livekit.yaml  # TTS synthesis + adversarial negatives
uv run livekit-wakeword augment configs/hey_livekit.yaml   # Add noise, reverb, pitch shifts
uv run livekit-wakeword train configs/hey_livekit.yaml     # 3-phase adaptive training
uv run livekit-wakeword export configs/hey_livekit.yaml    # Export to ONNX
```

**Config:**

See [configs/hey_livekit.yaml](configs/hey_livekit.yaml) for all options.

```yaml
model_name: hey_livekit
target_phrases:
  - "hey livekit"

n_samples: 10000 # training samples per class
model:
  model_type: dnn # dnn or rnn
  model_size: medium # tiny, small, medium, large
steps: 50000
target_fp_per_hour: 0.2
```

**Train on cloud GPUs with SkyPilot:**

See [skypilot/train.yaml](skypilot/train.yaml) for SkyPilot's example training job on Nebius.

```bash
sky launch skypilot/train.yaml
```

## Detailed Documentation

If you want to understand more about how this library works:

- [Architecture Overview](docs/overview.md) — system design and data flow
- [Data Generation](docs/data-generation.md) — TTS synthesis and adversarial negatives
- [Augmentation](docs/augmentation.md) — audio transforms and alignment
- [Feature Extraction](docs/feature-extraction.md) — mel spectrograms and embeddings
- [Training](docs/training.md) — 3-phase training and checkpoint averaging
- [Export & Inference](docs/export-and-inference.md) — ONNX export and Python API

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
