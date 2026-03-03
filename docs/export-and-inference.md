# Export & Inference

The export stage converts the trained PyTorch classifier to ONNX for deployment. The inference API provides `WakeWordModel` for prediction and `WakeWordListener` for async microphone detection.

**Source:** `src/livekit/wakeword/export/onnx.py`, `src/livekit/wakeword/inference/model.py`, `src/livekit/wakeword/inference/listener.py`
**CLI:** `livekit-wakeword export <config>`

## ONNX Export

### Classifier Export

`export_classifier()` exports the trained PyTorch classifier head to ONNX format.

| Property | Value |
|----------|-------|
| Input name | `embeddings` |
| Input shape | `(1, 16, 96)` with dynamic batch axis |
| Output name | `score` |
| Output shape | `(1, 1)` with dynamic batch axis |
| Opset version | 18 |

### INT8 Quantization

`quantize_onnx()` applies dynamic INT8 quantization using `onnxruntime.quantization`:

- Weight type: `QuantType.QInt8`
- Output filename: `<model_name>.int8.onnx`

Enable via the `--quantize` flag:

```bash
livekit-wakeword export configs/hey_jarvis.yaml --quantize
```

### Export Entry Point

`run_export()` loads the trained model from `output/<model_name>/<model_name>.pt`, exports it to ONNX, and optionally quantizes it. Raises `FileNotFoundError` if the trained model doesn't exist.

## Inference API

**Source:** `src/livekit/wakeword/inference/model.py`, `src/livekit/wakeword/inference/listener.py`

### WakeWordModel

The `WakeWordModel` class is a stateless prediction API for wake word detection. Pass a complete audio window (~2 seconds) and receive confidence scores.

```python
from livekit.wakeword import WakeWordModel

model = WakeWordModel(models=["hey_livekit.onnx"])

# Pass ~2 seconds of 16kHz audio
scores = model.predict(audio_chunk)
# Returns: {"hey_livekit": 0.95}
```

#### Initialization

```python
WakeWordModel(
    models: list[str | Path] | None = None,  # Paths to ONNX classifiers
)
```

Feature extraction models (`melspectrogram.onnx`, `embedding_model.onnx`) are bundled with the package and loaded automatically.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(audio_chunk)` | `dict[str, float]` | Scores for each loaded model (0-1) |
| `load_model(path, name)` | `None` | Load additional wake word model |

#### Audio Input

- **Format:** 16kHz mono, int16 or float32
- **Chunk size:** ~2 seconds (32,000 samples) recommended — yields 16 embeddings for the classifier
- **Stateless:** No internal audio buffering; the caller manages the audio window

### WakeWordListener

The `WakeWordListener` class provides async microphone detection with debouncing.

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

#### Initialization

```python
WakeWordListener(
    model: WakeWordModel,    # WakeWordModel instance with loaded classifiers
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
