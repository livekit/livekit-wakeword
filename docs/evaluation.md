# Evaluation

The evaluation stage runs the exported ONNX model against held-out validation data and produces a DET (Detection Error Tradeoff) curve, AUT score, and summary metrics.

**Source:** `src/livekit/wakeword/eval/evaluate.py`
**CLI:** `livekit-wakeword eval <config>` or as step 6/6 of `livekit-wakeword run`

## Overview

```
ONNX model (.onnx) + validation features (.npy)
    │
    ▼
Run inference on positive & negative samples
    │
    ▼
Compute DET curve (FPR vs FNR across thresholds)
    │
    ▼
Compute AUT (Area Under the DET curve)
    │
    ▼
Find optimal threshold (maximize recall at target FPPH)
    │
    ▼
Save DET plot (.png) + metrics (.json)
```

## CLI Usage

**Evaluate after a full pipeline run:**

```bash
uv run livekit-wakeword eval configs/hey_livekit.yaml
```

**Evaluate a specific ONNX model** (e.g., from a different training run or an openWakeWord model):

```bash
uv run livekit-wakeword eval configs/hey_livekit.yaml -m /path/to/model.onnx
```

If `--model` is not specified, the default path `output/<model_name>/<model_name>.onnx` is used.

## Python API

```python
from pathlib import Path
from livekit.wakeword import load_config
from livekit.wakeword.eval.evaluate import run_eval

config = load_config("configs/hey_livekit.yaml")
model_path = Path("output/hey_livekit/hey_livekit.onnx")

results = run_eval(config, model_path)
# results = {
#     "aut": 0.0012,
#     "fpph": 0.08,
#     "recall": 0.861,
#     "accuracy": 0.93,
#     "threshold": 0.68,
#     "n_positive": 15000,
#     "n_negative": 45084,
#     "validation_hours": 25.05,
# }
```

## Metrics

### AUT — Area Under the DET Curve

The primary aggregate metric. Computed by integrating FNR as a function of FPR using the trapezoidal rule. Lower is better (0 = perfect separation).

AUT captures the full tradeoff between false positives and false negatives across all thresholds, making it useful for comparing models without committing to a specific operating point.

### DET Curve

The Detection Error Tradeoff curve plots False Positive Rate (x-axis) against False Negative Rate (y-axis) across 1001 thresholds from 0.0 to 1.0. A perfect model hugs the origin; a random classifier falls on the diagonal.

The DET curve is saved as `output/<model_name>/<model_name>_det.png` with an annotation box showing AUT, FPPH, recall, and the optimal threshold.

### FPPH — False Positives Per Hour

The number of false triggers per hour of negative audio. Computed as:

```
FPPH = count(negative_scores >= threshold) / validation_hours
```

where `validation_hours = n_negative_clips × clip_duration / 3600`.

### Recall

True positive rate at the optimal threshold:

```
Recall = mean(positive_scores >= threshold)
```

### Threshold Optimization

The evaluation uses `find_best_threshold()` to scan thresholds from 0.01 to 0.99 and select the one that **maximizes recall** while keeping **FPPH ≤ `target_fp_per_hour`** (from config). If no threshold meets the FPPH target, it falls back to maximizing balanced accuracy.

## Validation Data

### Sources

| Source | Path | Type |
|--------|------|------|
| Positive test clips | `output/<model>/positive_features_test.npy` | Wake word samples (generated + augmented) |
| Negative test clips | `output/<model>/negative_features_test.npy` | Adversarial negatives (phonetically similar) |
| General negatives | `data/features/validation_set_features.npy` | ~11 hrs of ACAV100M speech (downloaded via `setup`) |

All feature arrays have shape `(N, 16, 96)` — N clips × 16 embedding timesteps × 96-dim speech embeddings.

The general negative validation set (`validation_set_features.npy`) is stored as 2D `(N, 96)` and reshaped to 3D during loading. Samples not divisible by 16 are dropped with a warning.

### Requirements

Evaluation requires that the data generation, augmentation, and feature extraction stages have been run first (to produce the `*_features_test.npy` files). The ACAV100M validation features are optional but recommended — without them, FPPH estimates are based only on adversarial negatives.

## Output Files

| File | Description |
|------|-------------|
| `<model_name>_det.png` | DET curve plot with metrics annotation |
| `<model_name>_eval.json` | Full metrics as JSON |

Example `_eval.json`:

```json
{
  "aut": 0.0012,
  "fpph": 0.08,
  "recall": 0.861,
  "accuracy": 0.93,
  "threshold": 0.68,
  "n_positive": 15000,
  "n_negative": 45084,
  "validation_hours": 25.05
}
```

## Comparing Models

The eval command accepts any ONNX model that takes `(1, 16, 96)` input and produces a `(1, 1)` score, making it useful for comparing models trained with different configurations or frameworks:

```bash
# Evaluate a livekit-wakeword model
uv run livekit-wakeword eval configs/hey_livekit.yaml -m models/conv_attention_medium.onnx

# Evaluate an openWakeWord model against the same validation set
uv run livekit-wakeword eval configs/hey_livekit.yaml -m models/hey_livekit_oww.onnx
```

This works because both livekit-wakeword and openWakeWord share the same frozen embedding front-end, producing identical `(16, 96)` feature matrices.
