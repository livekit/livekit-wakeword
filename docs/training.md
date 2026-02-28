# Training Pipeline

The training stage uses a 3-phase adaptive approach with hard example mining, linearly increasing negative weights, and checkpoint averaging.

**Source:** `src/livekit/wakeword/training/trainer.py`, `src/livekit/wakeword/training/metrics.py`
**CLI:** `livekit-wakeword train <config>`

## Overview

```
.npy features (N, 16, 96)
    │
    ▼
Phase 1: Full Training
    LR warmup → hold → cosine decay
    Hard example mining + negative weighting
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
Final model (.pt)
```

## 3-Phase Training

### Phase 1 — Full Training

| Parameter | Value |
|-----------|-------|
| Steps | `config.steps` (default: 50,000) |
| Learning rate | `config.learning_rate` (default: 1e-4) |
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

## Negative Weight Schedule

`_negative_weight_schedule(step, total_steps, max_weight)`

Linear increase from 1.0 to `max_weight` over the course of each phase:

```
weight = 1.0 + (max_weight - 1.0) * step / total_steps
```

Default `max_negative_weight` is 1500.0, meaning by the end of phase 1 the loss contribution of negative samples is weighted 1500x compared to the start.

## Loss Function

Binary cross-entropy (BCE) with per-sample weighting and hard example mining.

### Hard Example Mining

Only non-trivial predictions contribute to the loss:

| Sample Type | Kept If | Rationale |
|-------------|---------|-----------|
| Negative (label=0) | `prediction >= 0.001` | Already-correct negatives don't help |
| Positive (label=1) | `prediction < 0.999` | Already-correct positives don't help |

Trivial predictions are masked out by zeroing their loss contribution. This focuses training on the decision boundary where mistakes happen.

### Per-Sample Weighting

Negative samples are weighted by the current negative weight schedule value. Positive samples always have weight 1.0. The loss is computed with `reduction="none"`, then multiplied by the per-sample weight mask before averaging.

## Validation

### Data Sources

- **Positive:** `positive_features_test.npy`
- **Negative:** `negative_features_test.npy` + optional `validation_set_features.npy` (from ACAV100M, ~11 hours)

### Metrics

**Source:** `src/livekit/wakeword/training/metrics.py`

| Metric | Function | Description |
|--------|----------|-------------|
| FPPH | `false_positives_per_hour()` | Count of predictions >= threshold on negatives, divided by total hours |
| Recall | `recall_at_threshold()` | True positive rate: `mean(positive_preds >= threshold)` |
| Balanced Accuracy | `accuracy()` | `(TPR + TNR) / 2` at the given threshold |

The `evaluate_model()` function computes all metrics at once with a default `validation_hours=11.0`.

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

## Training Data Sources

The dataloader loads features from:

| Class | Source File | Label |
|-------|-----------|-------|
| `positive` | `positive_features_train.npy` | 1 |
| `adversarial_negative` | `negative_features_train.npy` | 0 |
| `ACAV100M_sample` | `data/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy` | 0 |

The ACAV100M dataset (if available) provides ~2000 hours of general audio embeddings as additional negative examples.

## Default Training Configuration

| Field | Default |
|-------|---------|
| `steps` | 50,000 |
| `learning_rate` | 1e-4 |
| `max_negative_weight` | 1500.0 |
| `target_fp_per_hour` | 0.2 |
| `batch_n_per_class.positive` | 50 |
| `batch_n_per_class.adversarial_negative` | 50 |
| `batch_n_per_class.ACAV100M_sample` | 1024 |

## Output

The trained model is saved to `output/<model_name>/<model_name>.pt`.
