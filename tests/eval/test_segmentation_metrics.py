"""Tests for segmentation evaluation metrics."""

import pytest
import torch
import numpy as np
from medfm.eval.segmentation import (
    dice_score,
    compute_segmentation_metrics,
    hausdorff_distance,
    surface_dice,
)


class TestDiceScore:
    def test_perfect_prediction(self):
        target = torch.randint(0, 3, (2, 64, 64))
        prediction = target.clone()
        score = dice_score(prediction, target)
        assert score.item() > 0.99

    def test_single_class(self):
        target = torch.ones(2, 64, 64)
        prediction = torch.ones(2, 64, 64)
        score = dice_score(prediction, target, class_idx=1)
        assert score.item() > 0.99

    def test_probabilistic_input(self):
        target = torch.randint(0, 3, (2, 64, 64))
        prediction = torch.randn(2, 3, 64, 64)
        score = dice_score(prediction, target)
        assert 0 <= score.item() <= 1


class TestComputeSegmentationMetrics:
    def test_output_keys(self):
        target = torch.randint(0, 3, (2, 64, 64))
        prediction = torch.randint(0, 3, (2, 64, 64))
        metrics = compute_segmentation_metrics(prediction, target, num_classes=3)
        expected_keys = [
            "dice_mean", "dice_std", "iou_mean", "iou_std",
            "sensitivity_mean", "specificity_mean",
            "dice_per_class", "iou_per_class",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_perfect_prediction_metrics(self):
        target = torch.randint(0, 3, (2, 64, 64))
        prediction = target.clone()
        metrics = compute_segmentation_metrics(prediction, target, num_classes=3)
        assert metrics["dice_mean"] > 0.99
        assert metrics["iou_mean"] > 0.99
        assert metrics["sensitivity_mean"] > 0.99
        assert metrics["specificity_mean"] > 0.99

    def test_with_background(self):
        target = torch.randint(0, 3, (2, 64, 64))
        prediction = torch.randint(0, 3, (2, 64, 64))
        metrics = compute_segmentation_metrics(
            prediction, target, num_classes=3, include_background=True
        )
        assert "dice_per_class" in metrics
        assert len(metrics["dice_per_class"]) == 3
