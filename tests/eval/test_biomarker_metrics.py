"""Tests for biomarker evaluation metrics."""

import pytest
import torch
import numpy as np
from medfm.eval.biomarker import (
    compute_biomarker_metrics,
    compute_biomarker_accuracy,
    compute_biomarker_bias,
)


class TestComputeBiomarkerMetrics:
    def test_perfect_prediction(self):
        targets = torch.randn(10, 5)
        predictions = targets.clone()
        metrics = compute_biomarker_metrics(predictions, targets)
        assert metrics["mae_mean"] < 1e-5
        assert metrics["r2_mean"] > 0.99

    def test_output_keys(self):
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        metrics = compute_biomarker_metrics(predictions, targets)
        expected_keys = [
            "mae_mean", "mse_mean", "rmse_mean", "r2_mean", "pearson_mean",
            "mae_per_biomarker", "r2_per_biomarker", "pearson_per_biomarker",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_with_biomarker_names(self):
        predictions = torch.randn(10, 3)
        targets = torch.randn(10, 3)
        names = ["liver_fat", "visceral_fat", "muscle_volume"]
        metrics = compute_biomarker_metrics(predictions, targets, biomarker_names=names)
        assert "mae_by_name" in metrics
        assert "r2_by_name" in metrics
        assert set(metrics["mae_by_name"].keys()) == set(names)


class TestComputeBiomarkerAccuracy:
    def test_perfect_accuracy(self):
        targets = torch.randn(10, 5)
        predictions = targets.clone()
        metrics = compute_biomarker_accuracy(predictions, targets)
        assert metrics["accuracy_within_10pct"] > 0.99
        assert metrics["accuracy_within_20pct"] > 0.99
        assert metrics["accuracy_within_30pct"] > 0.99

    def test_output_keys(self):
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        metrics = compute_biomarker_accuracy(predictions, targets)
        expected_keys = [
            "accuracy_within_10pct",
            "accuracy_within_20pct",
            "accuracy_within_30pct",
            "accuracy_per_biomarker",
        ]
        for key in expected_keys:
            assert key in metrics


class TestComputeBiomarkerBias:
    def test_no_bias(self):
        targets = torch.randn(10, 5)
        predictions = targets.clone()
        metrics = compute_biomarker_bias(predictions, targets)
        assert abs(metrics["mean_bias"]) < 1e-5

    def test_output_keys(self):
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        metrics = compute_biomarker_bias(predictions, targets)
        expected_keys = [
            "mean_bias", "mean_pct_bias",
            "bias_per_biomarker", "pct_bias_per_biomarker",
            "bland_altman_mean", "bland_altman_std",
            "bland_altman_loa_lower", "bland_altman_loa_upper",
        ]
        for key in expected_keys:
            assert key in metrics
