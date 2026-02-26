# Export & Inference

The export stage converts the trained PyTorch classifier to ONNX for deployment. The inference API provides `Model` for prediction and `Listener` for async microphone detection.

**Source:** `src/livewakeword/export/onnx.py`, `src/livewakeword/inference/model.py`, `src/livewakeword/inference/listener.py`
**CLI:** `livewakeword export <config>`

## ONNX Export

### Classifier Export

`export_classifier()` exports the trained PyTorch classifier head to ONNX format.

| Property | Value |
|----------|-------|
| Input name | `embeddings` |
| Input shape | `(1, 16, 96)` with dynamic batch axis |
| Output name | `score` |
| Output shape | `(1, 1)` with dynamic batch axis |
| Opset version | 13 |

### Full Pipeline Export

`export_full_pipeline()` assembles all three ONNX models into one directory:

```
output/<model_name>/
├── <model_name>.onnx          # Trained classifier head
├── melspectrogram.onnx         # Frozen mel-spectrogram frontend
└── embedding_model.onnx        # Frozen speech-embedding CNN
```

The mel-spectrogram and speech-embedding models are copied from `data/models/` (where `livewakeword setup` placed them). They cannot be fused into a single ONNX graph because they are separate pre-trained models.

### INT8 Quantization

`quantize_onnx()` applies dynamic INT8 quantization using `onnxruntime.quantization`:

- Weight type: `QuantType.QInt8`
- Output filename: `<model_name>.int8.onnx`

Enable via the `--quantize` flag:

```bash
livewakeword export configs/hey_jarvis.yaml --quantize
```

### Export Entry Point

`run_export()` loads the trained model from `output/<model_name>/<model_name>.pt`, exports it to ONNX, and optionally quantizes it. Raises `FileNotFoundError` if the trained model doesn't exist.

## Inference API

**Source:** `src/livewakeword/inference/model.py`, `src/livewakeword/inference/listener.py`

### Model

The `Model` class provides a simple prediction API for wake word detection.

```python
from livewakeword import Model

model = Model(wakeword_models=["hey_livekit.onnx"])

# Feed audio frames (16kHz, int16 or float32)
scores = model.predict(audio_frame)
# Returns: {"hey_livekit": 0.95}
```

#### Initialization

```python
Model(
    wakeword_models: list[str | Path] | None = None,  # Paths to ONNX classifiers
    inference_framework: str = "onnx"                  # Only "onnx" supported
)
```

Feature extraction models (`melspectrogram.onnx`, `embedding_model.onnx`) are bundled with the package and loaded automatically.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(audio_frame)` | `dict[str, float]` | Scores for each loaded model (0-1) |
| `load_model(path, name)` | `None` | Load additional wake word model |
| `reset()` | `None` | Clear internal buffers |

#### Audio Input

- **Format:** 16kHz mono, int16 or float32
- **Frame size:** Multiples of 80ms (1280 samples) recommended
- **Buffering:** Internal sliding window handles accumulation

### Listener

The `Listener` class provides async microphone detection with debouncing.

```python
import asyncio
from livewakeword import Model, Listener

model = Model(wakeword_models=["hey_livekit.onnx"])

async def main():
    async with Listener(model, threshold=0.5, debounce=2.0) as listener:
        while True:
            detection = await listener.wait_for_detection()
            print(f"Detected {detection.name}! ({detection.confidence:.2f})")

asyncio.run(main())
```

#### Initialization

```python
Listener(
    model: Model,            # Model instance with loaded classifiers
    threshold: float = 0.5,  # Detection threshold (0-1)
    debounce: float = 2.0    # Minimum seconds between detections
)
```

#### Detection Result

```python
@dataclass
class Detection:
    name: str        # Model name that triggered
    confidence: float  # Score (0-1)
    timestamp: float   # Monotonic time
```

#### Audio Capture

Uses PyAudio to capture from the default microphone:

| Parameter | Value |
|-----------|-------|
| Format | int16 (paInt16) |
| Channels | 1 (mono) |
| Sample rate | 16,000 Hz |
| Buffer size | 1,280 samples (80ms) |
