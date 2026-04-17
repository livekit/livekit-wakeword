<a href="https://livekit.io/">
  <img src="https://raw.githubusercontent.com/livekit/livekit-wakeword/main/.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# livekit-wakeword

[![CI](https://github.com/livekit/livekit-wakeword/actions/workflows/ci.yml/badge.svg)](https://github.com/livekit/livekit-wakeword/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1-green)](https://github.com/livekit/livekit-wakeword)

An open-source wake word library for creating voice-enabled applications. Based on [openWakeWord](https://github.com/dscripka/openWakeWord) with streamlined training: generate synthetic data, augment, train, and export from a single YAML config.

**Features:**

- **Conv-Attention classifier**: 1D temporal convolutions + multi-head self-attention replace openWakeWord's flat DNN head, preserving temporal structure across the 16-frame embedding window for better accuracy and fewer false positives (see [comparison](#why-livekit-wakeword) below)
- **Backward compatible** with openWakeWord models and library
- **Multilingual support**: over 30 languages via VoxCPM synthetic data generation
- **Train anywhere**: local machine, cloud, or spawn [SkyPilot](https://github.com/skypilot-org/skypilot) jobs
- **Zero dependency headaches**: uv handles everything

**Quick Links:**

- [Why livekit-wakeword](#why-livekit-wakeword)
- [Using a Pre-trained Model](#using-a-pre-trained-model)
- [Training a Custom Wake Word](#training-a-custom-wake-word)
- [Multilingual Support](#multilingual)
- [Python API](#python-api)
- [Example: Wake Word–Triggered Agent](https://github.com/livekit-examples/hello-wakeword)

## Why livekit-wakeword

Both livekit-wakeword and openWakeWord share the same audio front-end: mel spectrograms are fed through frozen [Google speech embedding](https://github.com/google-research/google-research/tree/master/embedding_fns) and [openWakeWord embedding](https://github.com/dscripka/openWakeWord) models to produce a `(16, 96)` feature matrix (16 timesteps × 96-dim embeddings). The difference is the classification head that sits on top.

### Architecture

**openWakeWord** flattens the `(16, 96)` matrix into a 1536-d vector and feeds it through a small fully-connected DNN:

```
Flatten(16×96=1536) → Dense → Dense → Sigmoid
```

While the positional information is technically still present in the flattened vector, the dense layer has no inductive bias for temporal structure and must learn any sequential patterns from scratch.

**livekit-wakeword** introduces a **Conv-Attention** (`conv_attention`) classifier:

```
Conv1D blocks → MultiheadAttention → Mean pool → Linear(1) → Sigmoid
```

1. **1D Convolutions** (kernel size 3) slide across the 16 timesteps, capturing local temporal patterns (e.g., syllable transitions).
2. **Multi-Head Self-Attention** models long-range dependencies across the full temporal window, letting the model learn which timestep relationships matter.
3. **Mean pooling** aggregates attended features into a fixed-size vector for the final sigmoid output.

### Results

To compare, we evaluated an openWakeWord DNN, a livekit-wakeword DNN (same architecture, better training pipeline), and a livekit-wakeword conv-attention model on the same "hey livekit" validation set (15,000 positive clips, 45,084 negative clips, 25 hours of audio). The livekit-wakeword models were trained with the [prod config](configs/prod.yaml).

| Metric              | openWakeWord (DNN) | livekit-wakeword (DNN) | livekit-wakeword (conv-attention) |
| ------------------- | :----------------: | :--------------------: | :-------------------------------: |
| **AUT\***           |       0.0720       |         0.0423         |            **0.0012**             |
| **FPPH\***          |        8.50        |          3.07          |             **0.08**              |
| **Recall\***        |       68.6%        |         85.3%          |             **86.1%**             |
| Optimal Threshold\* |        0.01        |          0.01          |               0.68                |

<table>
<tr>
<td align="center"><strong>openWakeWord (DNN)</strong></td>
<td align="center"><strong>livekit-wakeword (DNN)</strong></td>
<td align="center"><strong>livekit-wakeword (conv-attention)</strong></td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/livekit/livekit-wakeword/main/.github/assets/det_openwakeword.png" alt="DET curve: openWakeWord" width="280"></td>
<td><img src="https://raw.githubusercontent.com/livekit/livekit-wakeword/main/.github/assets/det_livekit_wakeword_dnn.png" alt="DET curve: livekit-wakeword DNN" width="280"></td>
<td><img src="https://raw.githubusercontent.com/livekit/livekit-wakeword/main/.github/assets/det_livekit_wakeword.png" alt="DET curve: livekit-wakeword conv-attention" width="280"></td>
</tr>
</table>

The livekit-wakeword DNN already outperforms openWakeWord's DNN thanks to the improved training pipeline (focal loss, embedding mixup, 3-phase training, checkpoint averaging). However, both DNN models fail to meet the FPPH target: their optimal thresholds fall to 0.01, meaning no operating point can keep false positives low enough.

The conv-attention head is what unlocks the low false positive rate: **60x lower AUT** and **100x fewer false positives per hour** than openWakeWord, while detecting 17% more wake words.

_\***AUT** (Area Under the DET curve): summarizes the full DET (Detection Error Tradeoff) curve, which plots false positive rate vs false negative rate across all thresholds. Lower is better (0 = perfect). A DET curve that hugs the bottom-left corner indicates strong separation between wake words and non-wake-words._

_\***FPPH** (False Positives Per Hour): how many times the model falsely triggers per hour of non-wake-word audio. Lower is better. For production use, < 0.5 FPPH is typical._

_\***Recall**: the percentage of actual wake words correctly detected. Higher is better._

_\***Optimal Threshold**: the detection threshold that maximizes recall while keeping FPPH at or below the target (configurable, default 0.1). A threshold of 0.01 indicates no threshold could meet the FPPH target; the evaluator fell back to the highest balanced accuracy._

### Why conv-attention wins

- **Temporal awareness**: the conv-attention model sees the _order_ of speech events, not just their presence, reducing false triggers from phonetically similar but differently ordered phrases.
- **Better accuracy at the same model size**: attention lets a small model selectively focus on discriminative time regions rather than learning dense connections over the full flattened input.
- **Lower false-positive rates**: temporal structure helps reject partial or reordered matches that a flat DNN would accept.

The conv-attention head is the default. You can switch to the original DNN or an RNN head via `model_type` in your config:

```yaml
model:
  model_type: conv_attention # conv_attention (default) | dnn | rnn
  model_size: small # tiny, small, medium, large
```

## Using a Pre-trained Model

### Python

**System dependencies (for microphone listener):**

```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt install portaudio19-dev
```

**Installation:**

```bash
pip install livekit-wakeword
# or
uv add livekit-wakeword
```

For microphone listening, install with the `listener` extra:

```bash
pip install livekit-wakeword[listener]
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

### Rust

For Rust applications, use the [`livekit-wakeword`](https://crates.io/crates/livekit-wakeword) crate:

```toml
[dependencies]
livekit-wakeword = "0.1"
```

```rust
use livekit_wakeword::WakeWordModel;

let mut model = WakeWordModel::new(&["hey_livekit.onnx"], 16000)?;

// Feed ~2s PCM audio chunks (i16, at configured sample rate)
let scores = model.predict(&audio_chunk)?;
if scores["hey_livekit"] > 0.5 {
    println!("Wake word detected!");
}
```

The mel spectrogram and speech embedding models are compiled into the binary; only the wake word classifier ONNX file is loaded at runtime. Audio at supported sample rates (22050–384000 Hz) is automatically resampled to 16 kHz.

## Training a Custom Wake Word

### CLI quick start

**System dependencies:**

```bash
# macOS
brew install espeak-ng ffmpeg portaudio

# Ubuntu/Debian
sudo apt install espeak-ng libsndfile1 ffmpeg sox portaudio19-dev
```

**Installation (with pip):**

```bash
pip install livekit-wakeword[train,eval,export]
```

**Installation (with uv):**

```bash
uv tool install livekit-wakeword[train,eval,export]
```

**Installation (from source):**

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
livekit-wakeword setup --config configs/prod.yaml
```

**Train a wake word:**

```bash
livekit-wakeword run configs/prod.yaml
```

Or run stages individually:

```bash
livekit-wakeword generate configs/prod.yaml  # TTS synthesis + adversarial negatives
livekit-wakeword augment configs/prod.yaml   # Augment + extract features
livekit-wakeword train configs/prod.yaml     # 3-phase adaptive training
livekit-wakeword export configs/prod.yaml    # Export to ONNX
livekit-wakeword eval configs/prod.yaml      # Evaluate model (DET curve, AUT, FPPH)
```

You can also evaluate any compatible ONNX model (e.g., one trained with openWakeWord):

```bash
livekit-wakeword eval configs/prod.yaml -m /path/to/other_model.onnx
```

Eval produces a DET curve plot and metrics JSON in the output directory. See [Evaluation](docs/evaluation.md) for details.

### Configuration

The full pipeline runs based on a single YAML configuration file. Example configs:

| Config                                               | Wake word      | Use                                                      |
| ---------------------------------------------------- | -------------- | -------------------------------------------------------- |
| [configs/prod.yaml](configs/prod.yaml)               | "hey livekit"  | Production-scale for English with **Piper TTS** backbone |
| [configs/test.yaml](configs/test.yaml)               | "hey livekit"  | Small end-to-end test run with **Piper TTS** backbone    |
| [configs/prod_voxcpm.yaml](configs/prod_voxcpm.yaml) | "你好 livekit" | Production-scale multilingual with **VoxCPM** backbone   |
| [configs/test_voxcpm.yaml](configs/test_voxcpm.yaml) | "你好 livekit" | Small end-to-end test run with **VoxCPM** backbone       |

The bare minimum configuration required is as follows:

```yaml
model_name: hey_livekit
target_phrases:
  - "hey livekit"

n_samples: 10000 # training samples per class
model:
  model_type: conv_attention # conv_attention, dnn, or rnn
  model_size: small # tiny, small, medium, large
steps: 50000
target_fp_per_hour: 0.2
```

### Multilingual

We support training wake words in 30 languages with [VoxCPM2 TTS](https://github.com/OpenBMB/VoxCPM) synthetic data generation:

```
Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese

Chinese Dialect: 四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话
```

To use this, add `tts_backend` in your configuration YAML:

```yaml
tts_backend: voxcpm
```

And install `livekit-wakeword` with the `voxcpm` optional dependency:

```bash
pip install livekit-wakeword[train,eval,export,voxcpm]
```

More detail: [docs/data-generation.md](docs/data-generation.md) (Piper, VoxCPM, setup).

### Python API

The full training pipeline is available as a Python API, so you can import and drive it from your own code instead of using the CLI:

```python
from livekit.wakeword import (
    WakeWordConfig,
    load_config,
    run_generate,
    run_augment,
    run_extraction,
    run_train,
    run_export,
    run_eval,
)

# Load from YAML or construct directly
config = load_config("configs/prod.yaml")

# Or build a config programmatically
config = WakeWordConfig(
    model_name="hey_robot",
    target_phrases=["hey robot"],
    n_samples=5000,
    steps=30000,
)

# Run individual stages
run_generate(config)     # TTS synthesis + adversarial negatives
run_augment(config)      # Add noise, reverb, pitch shifts
run_extraction(config)   # Extract mel spectrograms + speech embeddings → .npy
run_train(config)        # 3-phase adaptive training
onnx_path = run_export(config)       # Export to ONNX

# Evaluate the exported model
results = run_eval(config, onnx_path)
print(f"AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  Recall={results['recall']:.1%}")
```

This is useful for integrating wake word training into larger pipelines, automating model iteration, or building custom tooling on top of the data generation and training stages.

### Cloud GPUs with SkyPilot

See [skypilot/train.yaml](skypilot/train.yaml) for an example training job on Nebius.

```bash
sky launch skypilot/train.yaml
```

## Further Reading

- [Architecture Overview](docs/overview.md): system design and data flow
- [Data Generation](docs/data-generation.md): TTS synthesis and adversarial negatives
- [Augmentation](docs/augmentation.md): audio transforms and alignment
- [Feature Extraction](docs/feature-extraction.md): mel spectrograms and embeddings
- [Training](docs/training.md): 3-phase training and checkpoint averaging
- [Export & Inference](docs/export-and-inference.md): ONNX export and Python API
- [Evaluation](docs/evaluation.md): DET curves, AUT, and model comparison

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
