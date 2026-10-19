"""Biomarker evaluation metrics."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_biomarker_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    biomarker_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute comprehensive biomarker regression metrics.

    Args:
        predictions: Predicted biomarkers (B, D)
        targets: Ground truth biomarkers (B, D)
        biomarker_names: List of biomarker names

    Returns:
        Dictionary of metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # MAE per biomarker
    mae_per_biomarker = np.abs(predictions - targets).mean(axis=0)

    # MSE per biomarker
    mse_per_biomarker = ((predictions - targets) ** 2).mean(axis=0)

    # R² per biomarker
    ss_res = ((predictions - targets) ** 2).sum(axis=0)
    ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_biomarker = 1 - (ss_res / (ss_tot + 1e-8))

    # Pearson correlation per biomarker
    correlations = []
    for i in range(predictions.shape[1]):
        pred_c = predictions[:, i]
        tgt_c = targets[:, i]
        if np.std(pred_c) > 0 and np.std(tgt_c) > 0:
            corr = np.corrcoef(pred_c, tgt_c)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)

    results = {
        "mae_mean": float(mae_per_biomarker.mean()),
        "mse_mean": float(mse_per_biomarker.mean()),
        "rmse_mean": float(np.sqrt(mse_per_biomarker.mean())),
        "r2_mean": float(r2_per_biomarker.mean()),
        "pearson_mean": float(np.mean(correlations)),
        "mae_per_biomarker": mae_per_biomarker.tolist(),
        "r2_per_biomarker": r2_per_biomarker.tolist(),
        "pearson_per_biomarker": correlations,
    }

    if biomarker_names:
        results["mae_by_name"] = {
            name: float(mae) for name, mae in zip(biomarker_names, mae_per_biomarker)
        }
        results["r2_by_name"] = {
            name: float(r2) for name, r2 in zip(biomarker_names, r2_per_biomarker)
        }

    return results


def compute_biomarker_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    tolerance: float = 0.1,
) -> Dict[str, float]:
    """Compute biomarker accuracy within tolerance.

    Args:
        predictions: Predicted biomarkers (B, D)
        targets: Ground truth biomarkers (B, D)
        tolerance: Relative tolerance (e.g., 0.1 = 10%)

    Returns:
        Dictionary with accuracy metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Relative error
    relative_error = np.abs(predictions - targets) / (np.abs(targets) + 1e-8)

    # Within tolerance
    within_tol = (relative_error <= tolerance).mean(axis=0)

    return {
        "accuracy_within_10pct": float((relative_error <= 0.1).mean()),
        "accuracy_within_20pct": float((relative_error <= 0.2).mean()),
        "accuracy_within_30pct": float((relative_error <= 0.3).mean()),
        "accuracy_per_biomarker": within_tol.tolist(),
    }


def compute_biomarker_bias(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute systematic bias in biomarker predictions.

    Args:
        predictions: Predicted biomarkers (B, D)
        targets: Ground truth biomarkers (B, D)

    Returns:
        Dictionary with bias metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Mean bias
    bias = (predictions - targets).mean(axis=0)

    # Percentage bias
    pct_bias = bias / (targets.mean(axis=0) + 1e-8) * 100

    # Bland-Altman metrics
    mean_values = (predictions + targets) / 2
    diff_values = predictions - targets

    return {
        "mean_bias": float(bias.mean()),
        "mean_pct_bias": float(pct_bias.mean()),
        "bias_per_biomarker": bias.tolist(),
        "pct_bias_per_biomarker": pct_bias.tolist(),
        "bland_altman_mean": float(diff_values.mean()),
        "bland_altman_std": float(diff_values.std()),
        "bland_altman_loa_lower": float(diff_values.mean() - 1.96 * diff_values.std()),
        "bland_altman_loa_upper": float(diff_values.mean() + 1.96 * diff_values.std()),
    }
