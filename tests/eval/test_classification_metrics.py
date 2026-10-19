"""Tests for classification evaluation metrics."""

import pytest
import torch
import numpy as np
from medfm.eval.classification import (
    compute_auc,
    compute_classification_metrics,
    compute_sensitivity_specificity,
    find_optimal_threshold,
)


class TestComputeAUC:
    def test_perfect_auc(self):
        predictions = torch.tensor([0.1, 0.2, 0.8, 0.9])
        targets = torch.tensor([0, 0, 1, 1])
        result = compute_auc(predictions, targets)
        assert result["auc_mean"] > 0.99

    def test_multiclass_auc(self):
        predictions = torch.rand(10, 3)
        targets = torch.randint(0, 2, (10, 3))
        result = compute_auc(predictions, targets, num_classes=3)
        assert "auc_mean" in result
        assert "auc_per_class" in result


class TestComputeClassificationMetrics:
    def test_output_keys(self):
        predictions = torch.rand(10, 3)
        targets = torch.randint(0, 2, (10, 3))
        metrics = compute_classification_metrics(predictions, targets, num_classes=3)
        expected_keys = ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"]
        for key in expected_keys:
            assert key in metrics

    def test_perfect_classification(self):
        targets = torch.randint(0, 2, (10, 3))
        predictions = targets.float()
        metrics = compute_classification_metrics(predictions, targets, num_classes=3)
        assert metrics["accuracy"] > 0.99
        assert metrics["f1_score"] > 0.99


class TestComputeSensitivitySpecificity:
    def test_perfect_prediction(self):
        predictions = torch.tensor([0.1, 0.2, 0.8, 0.9])
        targets = torch.tensor([0, 0, 1, 1])
        metrics = compute_sensitivity_specificity(predictions, targets)
        assert metrics["sensitivity"] > 0.99
        assert metrics["specificity"] > 0.99

    def test_output_keys(self):
        predictions = torch.rand(10)
        targets = torch.randint(0, 2, (10,))
        metrics = compute_sensitivity_specificity(predictions, targets)
        expected_keys = ["sensitivity", "specificity", "tp", "tn", "fp", "fn"]
        for key in expected_keys:
            assert key in metrics


class TestFindOptimalThreshold:
    def test_youden_threshold(self):
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        threshold = find_optimal_threshold(predictions, targets, metric="youden")
        assert 0 < threshold < 1

    def test_f1_threshold(self):
        predictions = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        threshold = find_optimal_threshold(predictions, targets, metric="f1")
        assert 0 < threshold < 1
