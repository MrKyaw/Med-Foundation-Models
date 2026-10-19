"""Classification evaluation metrics."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_auc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Area Under the ROC Curve (AUC).

    Args:
        predictions: Predicted probabilities (B, C) or (B,)
        targets: Ground truth labels (B,) or (B, C) for multi-label
        num_classes: Number of classes (inferred if not specified)

    Returns:
        Dictionary with AUC scores
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError("scikit-learn is required for AUC computation")

    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)

    if num_classes is None:
        num_classes = predictions.shape[1]

    auc_scores = []

    for c in range(num_classes):
        pred_c = predictions[:, c].cpu().numpy()
        tgt_c = targets[:, c].cpu().numpy()

        # Skip if only one class present
        if len(np.unique(tgt_c)) < 2:
            continue

        auc = roc_auc_score(tgt_c, pred_c)
        auc_scores.append(auc)

    return {
        "auc_mean": np.mean(auc_scores) if auc_scores else 0.0,
        "auc_per_class": auc_scores,
    }


def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        predictions: Predicted probabilities (B, C) or (B,)
        targets: Ground truth labels (B,) or (B, C)
        num_classes: Number of classes
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )
    except ImportError:
        raise ImportError("scikit-learn is required for classification metrics")

    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)

    if num_classes is None:
        num_classes = predictions.shape[1]

    # Convert to binary predictions
    preds_binary = (predictions > threshold).int().cpu().numpy()
    targets_binary = targets.int().cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(targets_binary, preds_binary)
    precision = precision_score(targets_binary, preds_binary, average="macro", zero_division=0)
    recall = recall_score(targets_binary, preds_binary, average="macro", zero_division=0)
    f1 = f1_score(targets_binary, preds_binary, average="macro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(targets_binary.flatten(), preds_binary.flatten())

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
    }


def compute_sensitivity_specificity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute sensitivity and specificity.

    Args:
        predictions: Predicted probabilities (B,) or (B, 1)
        targets: Ground truth labels (B,) or (B, 1)
        threshold: Classification threshold

    Returns:
        Dictionary with sensitivity and specificity
    """
    if predictions.ndim == 2:
        predictions = predictions.squeeze(1)
    if targets.ndim == 2:
        targets = targets.squeeze(1)

    preds_binary = (predictions > threshold).int()
    targets_binary = targets.int()

    tp = ((preds_binary == 1) & (targets_binary == 1)).sum().item()
    tn = ((preds_binary == 0) & (targets_binary == 0)).sum().item()
    fp = ((preds_binary == 1) & (targets_binary == 0)).sum().item()
    fn = ((preds_binary == 0) & (targets_binary == 1)).sum().item()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_optimal_threshold(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric: str = "youden",
) -> float:
    """Find optimal classification threshold.

    Args:
        predictions: Predicted probabilities (B,)
        targets: Ground truth labels (B,)
        metric: 'youden' for Youden's J statistic, 'f1' for F1 score

    Returns:
        Optimal threshold
    """
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        raise ImportError("scikit-learn is required for threshold optimization")

    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    fpr, tpr, thresholds = roc_curve(targets_np, preds_np)

    if metric == "youden":
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif metric == "f1":
        f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-8)
        optimal_idx = np.argmax(f1_scores)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return thresholds[optimal_idx]
