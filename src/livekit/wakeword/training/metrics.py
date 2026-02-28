"""Training and evaluation metrics for wake word detection."""

from __future__ import annotations

import numpy as np


def false_positives_per_hour(
    predictions: np.ndarray,
    threshold: float,
    total_hours: float,
) -> float:
    """Compute false positives per hour (FPPH).

    Args:
        predictions: Model scores for negative samples
        threshold: Detection threshold
        total_hours: Total hours of audio represented

    Returns:
        False positives per hour
    """
    if total_hours <= 0:
        return float("inf")
    fp_count = np.sum(predictions >= threshold)
    return float(fp_count / total_hours)


def recall_at_threshold(
    predictions: np.ndarray,
    threshold: float,
) -> float:
    """Compute recall (true positive rate) at a given threshold.

    Args:
        predictions: Model scores for positive samples
        threshold: Detection threshold

    Returns:
        Recall (0-1)
    """
    if len(predictions) == 0:
        return 0.0
    return float(np.mean(predictions >= threshold))


def accuracy(
    positive_preds: np.ndarray,
    negative_preds: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute balanced accuracy.

    Args:
        positive_preds: Model scores for positive samples
        negative_preds: Model scores for negative samples
        threshold: Detection threshold

    Returns:
        Balanced accuracy (0-1): average of true positive rate and true negative rate
    """
    if len(positive_preds) == 0 and len(negative_preds) == 0:
        return 0.0
    tpr = float(np.mean(positive_preds >= threshold)) if len(positive_preds) > 0 else 0.0
    tnr = float(np.mean(negative_preds < threshold)) if len(negative_preds) > 0 else 0.0
    return (tpr + tnr) / 2.0


def evaluate_model(
    positive_preds: np.ndarray,
    negative_preds: np.ndarray,
    threshold: float = 0.5,
    validation_hours: float = 11.0,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Returns dict with keys: fpph, recall, accuracy, threshold
    """
    return {
        "fpph": false_positives_per_hour(negative_preds, threshold, validation_hours),
        "recall": recall_at_threshold(positive_preds, threshold),
        "accuracy": accuracy(positive_preds, negative_preds, threshold),
        "threshold": threshold,
    }
