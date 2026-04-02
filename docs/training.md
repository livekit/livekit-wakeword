# Training Pipeline

The training stage uses a 3-phase adaptive approach with focal loss, embedding mixup, AdamW optimization, and checkpoint averaging.

**Source:** `src/livekit/wakeword/training/trainer.py`, `src/livekit/wakeword/training/metrics.py`
**CLI:** `livekit-wakeword train <config>`

## Overview

```
.npy features (N, 16, 96)
    │
    ▼
Phase 1: Full Training
    LR warmup → hold → cosine decay
    Focal loss + negative weighting + embedding mixup
    │
    ▼
Phase 2: Refinement
    0.1× LR, steps/10 steps
    Adaptive negative weight doubling
    │
    ▼
Phase 3: Fine-Tuning
    0.01× LR, steps/10 steps
    │
    ▼
Checkpoint Averaging
    Select top checkpoints by FPPH + recall + accuracy
    Average their weights
    │
    ▼
Threshold Optimization
    Scan 0.01–0.99 to maximize recall at target FPPH
    │
    ▼
Final model (.pt)
```

## 3-Phase Training

### Phase 1 — Full Training

| Parameter | Value |
|-----------|-------|
| Steps | `config.steps` (default: 50,000) |
| Learning rate | `config.learning_rate` (default: 1e-4) |
| Optimizer | AdamW with `config.weight_decay` (default: 0.01) |
| Warmup | `steps // 5` linear warmup |
| Hold | `steps // 3` constant LR |
| Decay | Cosine decay to 0 |
| Negative weight | Linear 1.0 → `max_negative_weight` |
| Validation | Final quarter, every `steps // 20` steps |

### Phase 2 — Refinement

| Parameter | Value |
|-----------|-------|
| Steps | `config.steps // 10` |
| Learning rate | `config.learning_rate * 0.1` |
| Negative weight | `max_negative_weight` (doubled if FPPH > target) |

Between phase 1 and 2, the trainer validates and checks if FPPH exceeds `target_fp_per_hour`. If so, `max_negative_weight` is doubled to increase the penalty on false positives.

### Phase 3 — Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Steps | `config.steps // 10` |
| Learning rate | `config.learning_rate * 0.01` |
| Negative weight | `max_negative_weight` (doubled again if FPPH > target) |

Between phase 2 and 3, the trainer validates again and doubles `max_negative_weight` a second time if FPPH still exceeds the target.

## Learning Rate Schedule

`_cosine_warmup_schedule(step, total_steps, warmup_steps, hold_steps, base_lr)`

```
LR
 │  ╱‾‾‾‾‾‾‾‾‾╲
 │ ╱             ╲
 │╱               ╲
 └─────────────────╲──► step
   warmup  hold     cosine decay
```

- **Warmup:** Linear from 0 to `base_lr` over `warmup_steps`
- **Hold:** Constant `base_lr` for `hold_steps`
- **Decay:** Cosine annealing from `base_lr` to 0

Phases 2 and 3 have no warmup or hold (warmup and hold steps are 0).

## Optimizer

AdamW is used instead of plain Adam. AdamW decouples weight decay from the adaptive learning rate, providing proper L2 regularization. This is controlled by `config.weight_decay` (default: 0.01).

## Negative Weight Schedule

`_negative_weight_schedule(step, total_steps, max_weight)`

Linear increase from 1.0 to `max_weight` over the course of each phase:

```
weight = 1.0 + (max_weight - 1.0) * step / total_steps
```

Default `max_negative_weight` is 1500.0, meaning by the end of phase 1 the loss contribution of negative samples is weighted 1500x compared to the start.

## Loss Function

### Focal Loss

Focal loss replaces the previous BCE + hard example mining approach. It inherently down-weights well-classified examples via a `(1 - p_t)^γ` modulating factor:

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```

where `p_t` is the model's estimated probability for the correct class and `γ = 2.0`.

| γ value | Effect |
|---------|--------|
| 0 | Equivalent to standard BCE |
| 2 (default) | Well-classified examples (p_t > 0.9) contribute ~100x less to the loss |
| 5 | Even more aggressive down-weighting |

This eliminates the need for the manual hard-example mining thresholds (0.1/0.9) that previously filtered samples. Focal loss achieves the same effect smoothly and with fewer hyperparameters.

### Per-Sample Weighting

Negative samples are weighted by the current negative weight schedule value. Positive samples always have weight 1.0. The focal loss is computed per-sample, then multiplied by the per-sample weight before averaging.

## Regularization

### Label Smoothing

Training targets are softened from hard 0/1 to `ε/2` and `1 - ε/2`, where `ε = config.label_smoothing` (default: 0.05). With the default, labels become 0.025 and 0.975.

This prevents the model from producing overconfident sigmoid outputs (very close to 0.0 or 1.0), which improves score calibration and makes threshold-based detection more reliable.

### Embedding Mixup

During training, random pairs of samples within each batch are interpolated in embedding space:

```
λ ~ Beta(0.2, 0.2)
features_mixed = λ · features + (1-λ) · features[permutation]
labels_mixed = λ · labels + (1-λ) · labels[permutation]
```

The Beta(0.2, 0.2) distribution produces mixing coefficients that are usually close to 0 or 1 (light interpolation), creating virtual training examples near the original data points. This regularizes the classifier without requiring changes to the audio augmentation pipeline.

## Validation

### Data Sources

- **Positive:** `positive_features_test.npy`
- **Negative:** `negative_features_test.npy` + optional `validation_set_features.npy` (from ACAV100M)

If the external validation set has a sample count not divisible by 16 (required for the 2D→3D reshape), the remainder samples are dropped with a logged warning.

### Metrics

**Source:** `src/livekit/wakeword/training/metrics.py`

| Metric | Function | Description |
|--------|----------|-------------|
| FPPH | `false_positives_per_hour()` | Count of predictions >= threshold on negatives, divided by total hours |
| Recall | `recall_at_threshold()` | True positive rate: `mean(positive_preds >= threshold)` |
| Balanced Accuracy | `accuracy()` | `(TPR + TNR) / 2` at the given threshold |

The `evaluate_model()` function computes all metrics at once. Validation hours are computed from the actual negative clip count × clip duration (default 2.0s), not a hardcoded value. This ensures FPPH is accurate regardless of whether the external ACAV100M validation set is present.

### Validation Schedule

Validation runs during the final quarter of each phase, at intervals of `steps // 20`. Each validation produces a checkpoint.

## Checkpoint Averaging

After all three phases, the best checkpoints are averaged to produce the final model.

### Selection Criteria

A checkpoint qualifies if **all three** conditions are met:

| Metric | Threshold |
|--------|-----------|
| FPPH | <= 10th percentile of all checkpoints |
| Recall | >= 90th percentile of all checkpoints |
| Balanced Accuracy | >= 90th percentile of all checkpoints |

If no checkpoints meet all three criteria, the checkpoint with the highest recall is used as a fallback.

### Weight Averaging

For qualifying checkpoints, each parameter tensor is stacked and averaged:

```python
averaged[key] = mean(stack([ckpt[key] for ckpt in selected]))
```

This produces a smoother model that generalizes better than any single checkpoint.

## Threshold Optimization

After checkpoint averaging, the trainer searches for the optimal detection threshold on the validation set using `find_best_threshold()`:

1. Scan thresholds from 0.01 to 0.99 in steps of 0.01
2. For each threshold, compute FPPH and recall
3. Select the threshold that **maximizes recall** while keeping **FPPH ≤ `target_fp_per_hour`**
4. If no threshold meets the FPPH target, fall back to the one with the highest balanced accuracy

The optimal threshold and its metrics are logged to `metrics.json` with `"note": "optimal_threshold"`. This threshold can be used at inference time instead of the default 0.5.

## Training Data Sources

The dataloader loads features from:

| Class | Source File | Label |
|-------|-----------|-------|
| `positive` | `positive_features_train.npy` | 1 |
| `adversarial_negative` | `negative_features_train.npy` | 0 |
| `ACAV100M_sample` | `data/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | 0 |
| `background_noise` | `background_noise_features.npy` | 0 |

The ACAV100M dataset (if available) provides ~2000 hours of general audio embeddings as additional negative examples.

### Background Noise as Standalone Negatives

Background noise audio serves double duty in the pipeline:

1. **Augmentation overlay** — mixed into positive and adversarial clips at random SNR during the augmentation step (see [Augmentation](augmentation.md))
2. **Standalone negative class** — the same background WAV files are sliced into non-overlapping 2-second chunks, feature-extracted, and fed into training as their own negative class

This teaches the model that pure ambient noise (silence, HVAC, music, etc.) is not a wake word, rather than only seeing noise blended with speech.

To add custom background noise, put `.wav` files in a directory and add the path to `augmentation.background_paths` in your config:

```yaml
augmentation:
  background_paths:
    - ./data/backgrounds        # default MUSAN noise
    - ./data/my-office-noise    # your custom recordings
```

All `.wav` files are collected recursively. During feature extraction, they are chunked into `clip_duration`-length segments (default 2s) and saved as `background_noise_features.npy` in the model output directory. If no background files are found, this class is silently skipped.

## Default Training Configuration

| Field | Default |
|-------|---------|
| `steps` | 50,000 |
| `learning_rate` | 1e-4 |
| `weight_decay` | 0.01 |
| `label_smoothing` | 0.05 |
| `max_negative_weight` | 1500.0 |
| `target_fp_per_hour` | 0.2 |
| `batch_n_per_class.positive` | 50 |
| `batch_n_per_class.adversarial_negative` | 50 |
| `batch_n_per_class.ACAV100M_sample` | 1024 |
| `batch_n_per_class.background_noise` | 50 |

## Classifier Architectures

Three classifier types are available, all sharing the same ONNX I/O contract: input `embeddings` (batch, 16, 96) → output `score` (batch, 1).

### DNN (`model_type: dnn`)

Flattens the 16×96 embedding sequence into a single 1536-dim vector, then passes it through fully-connected layers. Fast and simple, but has no architectural bias for temporal structure.

```
Flatten(16×96=1536) → Linear(1536, dim) → LayerNorm → ReLU
→ N × FCNBlock(dim) → Linear(dim, 1) → Sigmoid
```

### RNN (`model_type: rnn`)

Bi-directional LSTM that processes the 16 timesteps sequentially, capturing temporal dependencies. Uses the final hidden state for classification.

```
Bi-LSTM(96→hidden, 2 layers) → Linear(hidden×2, 1) → Sigmoid
```

### Conv+Attention (`model_type: conv_attention`)

1D temporal convolutions capture local patterns across adjacent timesteps, followed by multi-head self-attention to model long-range temporal dependencies. Mean-pools over timesteps for the final prediction.

```
Conv1D(96→dim, k=3) → LayerNorm → ReLU
→ N × Conv1D(dim, k=3) → LayerNorm → ReLU
→ MultiheadAttention(dim, heads) → LayerNorm (residual)
→ MeanPool → Linear(dim, 1) → Sigmoid
```

This head is best at distinguishing wake words from phonetically similar phrases because it explicitly models the temporal ordering of phoneme embeddings.

### Size Presets

| Size | layer_dim | n_blocks |
|------|-----------|----------|
| tiny | 16 | 1 |
| small | 32 | 1 |
| medium | 128 | 2 |
| large | 256 | 3 |

## Output

The trained model is saved to `output/<model_name>/<model_name>.pt`.
