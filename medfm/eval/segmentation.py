"""Segmentation evaluation metrics."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_idx: Optional[int] = None,
    smooth: float = 1e-8,
) -> torch.Tensor:
    """Compute Dice score.

    Args:
        prediction: Predicted segmentation (B, C, H, W) or (B, H, W)
        target: Ground truth segmentation (B, H, W)
        class_idx: If specified, compute Dice for this class only
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice score (scalar or per-class)
    """
    if prediction.ndim == 4:
        prediction = prediction.argmax(dim=1)

    if class_idx is not None:
        pred = (prediction == class_idx).float()
        tgt = (target == class_idx).float()
        intersection = (pred * tgt).sum()
        return (2 * intersection + smooth) / (pred.sum() + tgt.sum() + smooth)

    # Compute Dice for all classes
    num_classes = int(torch.max(target).item()) + 1
    dice_scores = []

    for c in range(1, num_classes):  # Skip background
        pred = (prediction == c).float()
        tgt = (target == c).float()
        intersection = (pred * tgt).sum()
        dice = (2 * intersection + smooth) / (pred.sum() + tgt.sum() + smooth)
        dice_scores.append(dice)

    return torch.stack(dice_scores).mean()


def compute_segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 30,
    include_background: bool = False,
) -> Dict[str, float]:
    """Compute comprehensive segmentation metrics.

    Args:
        prediction: Predicted segmentation (B, C, H, W) or (B, H, W)
        target: Ground truth segmentation (B, H, W)
        num_classes: Number of classes
        include_background: Whether to include background class

    Returns:
        Dictionary of metrics
    """
    if prediction.ndim == 4:
        prediction = prediction.argmax(dim=1)

    start_class = 0 if include_background else 1

    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    specificity_scores = []

    for c in range(start_class, num_classes):
        pred = (prediction == c).float()
        tgt = (target == c).float()

        intersection = (pred * tgt).sum()
        union = pred.sum() + tgt.sum() - intersection

        # Dice
        dice = (2 * intersection + 1e-8) / (pred.sum() + tgt.sum() + 1e-8)
        dice_scores.append(dice.item())

        # IoU (Jaccard)
        iou = (intersection + 1e-8) / (union + 1e-8)
        iou_scores.append(iou.item())

        # Sensitivity (Recall)
        sensitivity = (intersection + 1e-8) / (tgt.sum() + 1e-8)
        sensitivity_scores.append(sensitivity.item())

        # Specificity
        tn = ((1 - pred) * (1 - tgt)).sum()
        fp = (pred * (1 - tgt)).sum()
        specificity = (tn + 1e-8) / (tn + fp + 1e-8)
        specificity_scores.append(specificity.item())

    return {
        "dice_mean": np.mean(dice_scores),
        "dice_std": np.std(dice_scores),
        "iou_mean": np.mean(iou_scores),
        "iou_std": np.std(iou_scores),
        "sensitivity_mean": np.mean(sensitivity_scores),
        "specificity_mean": np.mean(specificity_scores),
        "dice_per_class": dice_scores,
        "iou_per_class": iou_scores,
    }


def hausdorff_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spacing: Optional[Tuple[float, float, float]] = None,
) -> float:
    """Compute Hausdorff distance.

    Args:
        prediction: Predicted segmentation (B, H, W) or (B, D, H, W)
        target: Ground truth segmentation (B, H, W) or (B, D, H, W)
        spacing: Voxel spacing in physical units

    Returns:
        Hausdorff distance
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("scipy is required for Hausdorff distance computation")

    if prediction.ndim == 4:
        prediction = prediction.argmax(dim=1)

    hd_distances = []

    for pred, tgt in zip(prediction, target):
        pred_np = pred.cpu().numpy()
        tgt_np = tgt.cpu().numpy()

        # Get boundary points
        pred_boundary = pred_np - distance_transform_edt(pred_np)
        tgt_boundary = tgt_np - distance_transform_edt(tgt_np)

        # Compute distances
        dt_pred = distance_transform_edt(1 - pred_boundary)
        dt_tgt = distance_transform_edt(1 - tgt_boundary)

        hd = max(dt_pred[tgt_boundary > 0].max(), dt_tgt[pred_boundary > 0].max())

        if spacing is not None:
            hd *= np.mean(spacing)

        hd_distances.append(hd)

    return np.mean(hd_distances)


def surface_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tolerance: float = 1.0,
) -> float:
    """Compute surface Dice score.

    Args:
        prediction: Predicted segmentation (B, H, W) or (B, D, H, W)
        target: Ground truth segmentation (B, H, W) or (B, D, H, W)
        tolerance: Surface tolerance in voxels

    Returns:
        Surface Dice score
    """
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError:
        raise ImportError("scipy is required for surface Dice computation")

    if prediction.ndim == 4:
        prediction = prediction.argmax(dim=1)

    surface_dice_scores = []

    for pred, tgt in zip(prediction, target):
        pred_np = pred.cpu().numpy()
        tgt_np = tgt.cpu().numpy()

        # Get surfaces
        pred_surface = pred_np - binary_erosion(pred_np)
        tgt_surface = tgt_np - binary_erosion(tgt_np)

        # Compute distances
        dt_pred = distance_transform_edt(1 - pred_surface)
        dt_tgt = distance_transform_edt(1 - tgt_surface)

        # Count surface points within tolerance
        pred_in_tol = (dt_tgt <= tolerance) & (pred_surface > 0)
        tgt_in_tol = (dt_pred <= tolerance) & (tgt_surface > 0)

        surface_dice = (pred_in_tol.sum() + tgt_in_tol.sum()) / (
            pred_surface.sum() + tgt_surface.sum() + 1e-8
        )
        surface_dice_scores.append(surface_dice)

    return np.mean(surface_dice_scores)
