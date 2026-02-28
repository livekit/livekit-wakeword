"""3-phase adaptive training loop for wake word classifiers."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn

from ..config import WakeWordConfig
from ..data.dataset import create_dataloader
from ..models.pipeline import WakeWordClassifier
from ..utils import get_device
from .metrics import evaluate_model

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
        return evaluate_model(pos_preds, neg_preds, threshold=0.5)

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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        criterion = nn.BCELoss(reduction="none")

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

            # Forward
            predictions = self.model(features).squeeze(-1)
            loss_per_sample = criterion(predictions, labels)

            # Hard example mining
            with torch.no_grad():
                hard_mask = torch.ones_like(labels, dtype=torch.bool)
                neg_mask = labels < 0.5
                pos_mask = labels >= 0.5
                hard_mask[neg_mask] = predictions[neg_mask] >= 0.001
                hard_mask[pos_mask] = predictions[pos_mask] < 0.999

            # Negative weighting
            neg_weight = _negative_weight_schedule(step, steps, max_negative_weight)
            weights = torch.where(labels < 0.5, neg_weight, 1.0)

            if hard_mask.any():
                masked_loss = loss_per_sample[hard_mask] * weights[hard_mask]
                loss = masked_loss.mean()
            else:
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
                self._save_checkpoint(step, phase, metrics)
                self.model.train()

    def train(self) -> nn.Module:
        """Run full 3-phase training. Returns the final averaged model."""
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

        # Select and average best checkpoints
        final_model = self._average_best_checkpoints()
        return final_model

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
