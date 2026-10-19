"""Clinical validation metrics."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_clinical_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute clinical validation metrics.

    Args:
        predictions: Predicted probabilities (B,)
        targets: Ground truth labels (B,)
        thresholds: List of thresholds to evaluate

    Returns:
        Dictionary of clinical metrics
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    results = {}

    for thresh in thresholds:
        preds_binary = (preds_np >= thresh).astype(int)

        tp = ((preds_binary == 1) & (targets_np == 1)).sum()
        tn = ((preds_binary == 0) & (targets_np == 0)).sum()
        fp = ((preds_binary == 1) & (targets_np == 0)).sum()
        fn = ((preds_binary == 0) & (targets_np == 1)).sum()

        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        ppv = tp / (tp + fp + 1e-8)  # Positive Predictive Value
        npv = tn / (tn + fn + 1e-8)  # Negative Predictive Value

        results[f"sensitivity_at_{thresh}"] = float(sensitivity)
        results[f"specificity_at_{thresh}"] = float(specificity)
        results[f"ppv_at_{thresh}"] = float(ppv)
        results[f"npv_at_{thresh}"] = float(npv)

    return results


def compute_net_benefit(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute net benefit for decision curve analysis.

    Args:
        predictions: Predicted probabilities (B,)
        targets: Ground truth labels (B,)
        thresholds: List of threshold probabilities

    Returns:
        Dictionary with net benefit at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01).tolist()

    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    n = len(targets_np)
    net_benefits = []

    for thresh in thresholds:
        preds_binary = (preds_np >= thresh).astype(int)

        tp = ((preds_binary == 1) & (targets_np == 1)).sum()
        fp = ((preds_binary == 1) & (targets_np == 0)).sum()

        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh + 1e-8))
        net_benefits.append(net_benefit)

    return {
        "thresholds": thresholds,
        "net_benefit": net_benefits,
    }


def compute_calibration_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute calibration metrics.

    Args:
        predictions: Predicted probabilities (B,)
        targets: Ground truth labels (B,)
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (preds_np >= bin_boundaries[i]) & (preds_np < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_confidence = preds_np[mask].mean()
            avg_accuracy = targets_np[mask].mean()
            ece += (mask.sum() / len(preds_np)) * abs(avg_confidence - avg_accuracy)

    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(n_bins):
        mask = (preds_np >= bin_boundaries[i]) & (preds_np < bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_confidence = preds_np[mask].mean()
            avg_accuracy = targets_np[mask].mean()
            mce = max(mce, abs(avg_confidence - avg_accuracy))

    # Brier Score
    brier = np.mean((preds_np - targets_np) ** 2)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
    }
