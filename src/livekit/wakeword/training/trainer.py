"""3-phase adaptive training loop for wake word classifiers."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import WakeWordConfig
from ..data.dataset import create_dataloader
from ..models.pipeline import WakeWordClassifier
from ..utils import get_device
from .metrics import evaluate_model, find_best_threshold

logger = logging.getLogger(__name__)


class _Checkpoint(TypedDict):
    step: int
    phase: int
    metrics: dict[str, float]
    state_dict: dict[str, torch.Tensor]


def _cosine_warmup_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    hold_steps: int,
    base_lr: float,
) -> float:
    """Learning rate with warmup → hold → cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    elif step < warmup_steps + hold_steps:
        return base_lr
    else:
        decay_steps = total_steps - warmup_steps - hold_steps
        progress = (step - warmup_steps - hold_steps) / max(1, decay_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _negative_weight_schedule(step: int, total_steps: int, max_weight: float) -> float:
    """Linear schedule for negative class weight: 1 → max_weight."""
    return 1.0 + (max_weight - 1.0) * step / max(1, total_steps)


def focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for binary classification (per-sample, unreduced).

    Down-weights well-classified examples by (1 - p_t)^gamma, automatically
    focusing training on hard examples near the decision boundary.  This
    replaces the manual hard-example mining thresholds (0.1 / 0.9).

    Args:
        predictions: Model output probabilities (after sigmoid), shape (N,).
        targets: Ground truth labels 0 or 1, shape (N,).
        gamma: Focusing parameter. 0 = standard BCE, 2 = typical focal loss.

    Returns:
        Per-sample focal loss, shape (N,).
    """
    bce = F.binary_cross_entropy(predictions, targets, reduction="none")
    p_t = predictions * targets + (1 - predictions) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return focal_weight * bce


class WakeWordTrainer:
    """Three-phase adaptive trainer for wake word classifiers.

    Phase 1: Full training with warmup + cosine decay
    Phase 2: Refinement at lower LR, adaptive negative weight
    Phase 3: Fine-tuning at lowest LR
    """

    def __init__(self, config: WakeWordConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or get_device()
        self.model = WakeWordClassifier(config).to(self.device)
        self.checkpoints: list[_Checkpoint] = []
        self._metrics_log: list[dict[str, object]] = []
        self._metrics_path = config.model_output_dir / f"{config.model_name}_metrics.json"
        self._train_start: float = 0.0

    def _build_dataloader(self) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
        model_dir = self.config.model_output_dir
        data_files: dict[str, str | Path] = {
            "positive": model_dir / "positive_features_train.npy",
            "adversarial_negative": model_dir / "negative_features_train.npy",
        }
        # Add ACAV100M if available
        acav_path = (
            self.config.data_path / "features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        )
        if acav_path.exists():
            data_files["ACAV100M_sample"] = acav_path
        else:
            logger.warning(
                "ACAV100M features not found at %s. Training without general negative "
                "speech data — the model may have a high false positive rate. "
                "Run setup without --skip-acav to download (~16 GB).",
                acav_path,
            )

        return create_dataloader(
            data_files=data_files,
            n_per_class=self.config.batch_n_per_class,
            label_funcs={
                "positive": lambda _: 1,
                "adversarial_negative": lambda _: 0,
                "ACAV100M_sample": lambda _: 0,
            },
        )

    def _load_validation_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load test features for validation."""
        model_dir = self.config.model_output_dir
        pos_path = model_dir / "positive_features_test.npy"
        neg_path = model_dir / "negative_features_test.npy"

        pos = np.load(str(pos_path)) if pos_path.exists() else np.zeros((0, 16, 96))
        neg = np.load(str(neg_path)) if neg_path.exists() else np.zeros((0, 16, 96))

        # Also load validation features if available
        val_path = self.config.data_path / "features" / "validation_set_features.npy"
        if val_path.exists():
            val_neg = np.load(str(val_path))
            # Reshape 2D (N, 96) → 3D (N//16, 16, 96) if needed
            if val_neg.ndim == 2:
                n_full = (val_neg.shape[0] // 16) * 16
                remainder = val_neg.shape[0] - n_full
                if remainder > 0:
                    logger.warning(
                        "Dropping %d/%d validation samples (not divisible by 16)",
                        remainder, val_neg.shape[0],
                    )
                val_neg = val_neg[:n_full].reshape(-1, 16, 96)
            neg = np.concatenate([neg, val_neg], axis=0) if neg.shape[0] > 0 else val_neg

        return pos, neg

    @torch.no_grad()
    def _predict(self, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Run model prediction on numpy features."""
        self.model.eval()
        all_preds: list[np.ndarray] = []
        for i in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[i : i + batch_size]).to(self.device)
            preds = self.model(batch).cpu().numpy()
            all_preds.append(preds)
        return np.concatenate(all_preds, axis=0).squeeze(-1)

    def _validate(self) -> dict[str, float]:
        """Run validation and return metrics."""
        pos_features, neg_features = self._load_validation_data()
        if pos_features.shape[0] == 0:
            return {"fpph": 0.0, "recall": 0.0, "accuracy": 0.0, "threshold": 0.5}
        pos_preds = self._predict(pos_features)
        neg_preds = self._predict(neg_features) if neg_features.shape[0] > 0 else np.array([])

        # Compute actual validation hours from clip count × clip duration
        clip_duration = self.config.augmentation.clip_duration
        validation_hours = neg_features.shape[0] * clip_duration / 3600.0
        return evaluate_model(
            pos_preds, neg_preds, threshold=0.5, validation_hours=validation_hours,
        )

    def _log_metrics(self, step: int, phase: int, metrics: dict[str, float]) -> None:
        """Append a metrics entry and flush to disk."""
        entry: dict[str, object] = {
            "step": step,
            "phase": phase,
            "elapsed_s": round(time.time() - self._train_start, 1),
            **metrics,
        }
        self._metrics_log.append(entry)
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_path.write_text(json.dumps(self._metrics_log, indent=2) + "\n")

    def _save_checkpoint(self, step: int, phase: int, metrics: dict[str, float]) -> None:
        """Save checkpoint if it meets quality criteria."""
        self.checkpoints.append(
            {
                "step": step,
                "phase": phase,
                "metrics": metrics,
                "state_dict": copy.deepcopy(self.model.state_dict()),
            }
        )

    def _train_phase(
        self,
        phase: int,
        steps: int,
        base_lr: float,
        max_negative_weight: float,
        dataloader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    ) -> None:
        """Run a single training phase."""
        logger.info(
            f"=== Phase {phase}: {steps} steps, LR={base_lr}, max_neg_w={max_negative_weight} ==="
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=self.config.weight_decay,
        )
        label_smoothing = self.config.label_smoothing

        warmup_steps = steps // 5 if phase == 1 else 0
        hold_steps = steps // 3 if phase == 1 else 0

        from tqdm import tqdm

        self.model.train()
        data_iter = iter(dataloader)
        validation_interval = max(1, steps // 20)

        pbar = tqdm(range(steps), desc=f"Phase {phase}", unit="step")
        for step in pbar:
            # Get batch
            try:
                features, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                features, labels = next(data_iter)

            features = features.to(self.device)
            labels = labels.to(self.device)

            # Embedding mixup: interpolate random pairs of samples and their
            # labels to create virtual training examples. This regularizes in
            # embedding space without touching the audio pipeline.
            mixup_alpha = 0.2
            if mixup_alpha > 0:
                lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().to(self.device)
                perm = torch.randperm(features.size(0), device=self.device)
                features = lam * features + (1 - lam) * features[perm]
                labels = lam * labels + (1 - lam) * labels[perm]

            # Label smoothing: 0→ε, 1→1-ε to prevent overconfident predictions
            if label_smoothing > 0:
                labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

            # Forward
            predictions = self.model(features).squeeze(-1)

            # Focal loss replaces BCE + manual hard-example mining.
            # gamma=2.0 automatically down-weights well-classified samples,
            # eliminating the need for the 0.1/0.9 threshold heuristics.
            loss_per_sample = focal_loss(predictions, labels, gamma=2.0)

            # Negative weighting
            neg_weight = _negative_weight_schedule(step, steps, max_negative_weight)
            weights = torch.where(labels < 0.5, neg_weight, 1.0)

            loss = (loss_per_sample * weights).mean()

            # LR schedule
            lr = _cosine_warmup_schedule(step, steps, warmup_steps, hold_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}", neg_w=f"{neg_weight:.0f}")

            # Validation in final quarter
            if step >= steps * 3 // 4 and step % validation_interval == 0:
                metrics = self._validate()
                pbar.write(
                    f"  Validation @ step {step}: "
                    f"FPPH={metrics['fpph']:.2f}, "
                    f"Recall={metrics['recall']:.3f}, Acc={metrics['accuracy']:.3f}"
                )
                self._log_metrics(step, phase, metrics)
                self._save_checkpoint(step, phase, metrics)
                self.model.train()

    def train(self) -> nn.Module:
        """Run full 3-phase training. Returns the final averaged model."""
        self._train_start = time.time()
        dataloader = self._build_dataloader()
        steps = self.config.steps
        max_neg_w = self.config.max_negative_weight

        # Phase 1: Full training
        self._train_phase(
            phase=1,
            steps=steps,
            base_lr=self.config.learning_rate,
            max_negative_weight=max_neg_w,
            dataloader=dataloader,
        )

        # Phase 2: Refinement
        metrics = self._validate()
        self._log_metrics(steps, 1, metrics)
        if metrics["fpph"] > self.config.target_fp_per_hour:
            max_neg_w *= 2
            logger.info(f"FP rate too high, doubling max_negative_weight to {max_neg_w}")

        self._train_phase(
            phase=2,
            steps=steps // 10,
            base_lr=self.config.learning_rate * 0.1,
            max_negative_weight=max_neg_w,
            dataloader=dataloader,
        )

        # Phase 3: Fine-tuning
        metrics = self._validate()
        self._log_metrics(steps + steps // 10, 2, metrics)
        if metrics["fpph"] > self.config.target_fp_per_hour:
            max_neg_w *= 2
            logger.info(f"FP rate still high, doubling max_negative_weight to {max_neg_w}")

        self._train_phase(
            phase=3,
            steps=steps // 10,
            base_lr=self.config.learning_rate * 0.01,
            max_negative_weight=max_neg_w,
            dataloader=dataloader,
        )

        # Final validation after phase 3
        metrics = self._validate()
        total_steps = steps + steps // 10 + steps // 10
        self._log_metrics(total_steps, 3, metrics)

        # Select and average best checkpoints
        final_model = self._average_best_checkpoints()

        # Log final averaged model metrics
        final_metrics = self._validate()
        self._log_metrics(total_steps, 0, {**final_metrics, "note": "final_averaged"})  # type: ignore[arg-type]

        # Optimize detection threshold on validation set
        optimal = self._find_optimal_threshold()
        self._log_metrics(total_steps, 0, {**optimal, "note": "optimal_threshold"})  # type: ignore[arg-type]
        logger.info(
            f"Optimal threshold: {optimal['threshold']:.2f} "
            f"(FPPH={optimal['fpph']:.2f}, Recall={optimal['recall']:.3f})"
        )
        logger.info(f"Metrics saved to {self._metrics_path}")

        return final_model

    def _find_optimal_threshold(self) -> dict[str, float]:
        """Find best detection threshold on validation data."""
        pos_features, neg_features = self._load_validation_data()
        if pos_features.shape[0] == 0:
            return {"fpph": 0.0, "recall": 0.0, "accuracy": 0.0, "threshold": 0.5}
        pos_preds = self._predict(pos_features)
        neg_preds = self._predict(neg_features) if neg_features.shape[0] > 0 else np.array([])

        clip_duration = self.config.augmentation.clip_duration
        validation_hours = neg_features.shape[0] * clip_duration / 3600.0
        return find_best_threshold(
            pos_preds, neg_preds,
            validation_hours=validation_hours,
            target_fpph=self.config.target_fp_per_hour,
        )

    def _average_best_checkpoints(self) -> nn.Module:
        """Average weights of top checkpoints.

        Select models in 90th percentile accuracy/recall and 10th percentile FP rate.
        """
        if not self.checkpoints:
            logger.warning("No checkpoints saved, returning current model")
            return self.model

        # Extract metrics
        fpph_values = [c["metrics"]["fpph"] for c in self.checkpoints]
        recall_values = [c["metrics"]["recall"] for c in self.checkpoints]
        acc_values = [c["metrics"]["accuracy"] for c in self.checkpoints]

        # Filter: low FP (10th percentile) + high recall/accuracy (90th percentile)
        fp_threshold = np.percentile(fpph_values, 10) if len(fpph_values) > 1 else float("inf")
        recall_threshold = np.percentile(recall_values, 90) if len(recall_values) > 1 else 0.0
        acc_threshold = np.percentile(acc_values, 90) if len(acc_values) > 1 else 0.0

        selected = [
            c
            for c in self.checkpoints
            if c["metrics"]["fpph"] <= fp_threshold
            and c["metrics"]["recall"] >= recall_threshold
            and c["metrics"]["accuracy"] >= acc_threshold
        ]

        if not selected:
            # Fallback: use checkpoint with best recall
            selected = [max(self.checkpoints, key=lambda c: c["metrics"]["recall"])]

        logger.info(f"Averaging {len(selected)} checkpoints")

        # Average state dicts
        avg_state = {}
        for key in selected[0]["state_dict"].keys():
            tensors = [c["state_dict"][key] for c in selected]
            avg_state[key] = torch.stack(tensors).float().mean(dim=0)

        self.model.load_state_dict(avg_state)
        return self.model

    def save(self, path: Path) -> None:
        """Save trained model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")


def run_train(config: WakeWordConfig) -> Path:
    """Run training and return path to saved model."""
    trainer = WakeWordTrainer(config)
    trainer.train()

    model_path = config.model_output_dir / f"{config.model_name}.pt"
    trainer.save(model_path)
    return model_path
