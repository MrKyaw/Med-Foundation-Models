"""Tests for Biomarker heads."""

import pytest
import torch
from medfm.models.biomarker import BiomarkerHead, MultiTaskBiomarkerHead, LongitudinalBiomarkerHead


class TestBiomarkerHead:
    def test_output_shape(self):
        head = BiomarkerHead(in_features=128, biomarker_dims=10, hidden_dim=64)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 10)

    def test_loss(self):
        head = BiomarkerHead(in_features=128, biomarker_dims=10, hidden_dim=64)
        predictions = torch.randn(4, 10)
        targets = torch.randn(4, 10)
        loss = head.loss(predictions, targets)
        assert loss.item() > 0


class TestMultiTaskBiomarkerHead:
    def test_output_keys(self):
        head = MultiTaskBiomarkerHead(
            in_features=128,
            body_composition_dims=5,
            organ_volume_dims=8,
            fat_fraction_dims=3,
            vascular_dims=4,
            hidden_dim=64,
        )
        x = torch.randn(2, 128)
        out = head(x)
        assert set(out.keys()) == {"body_composition", "organ_volumes", "fat_fractions", "vascular"}
        assert out["body_composition"].shape == (2, 5)
        assert out["organ_volumes"].shape == (2, 8)
        assert out["fat_fractions"].shape == (2, 3)
        assert out["vascular"].shape == (2, 4)

    def test_loss(self):
        head = MultiTaskBiomarkerHead(
            in_features=128,
            body_composition_dims=5,
            organ_volume_dims=8,
            fat_fraction_dims=3,
            vascular_dims=4,
            hidden_dim=64,
        )
        x = torch.randn(4, 128)
        predictions = head(x)
        targets = {key: torch.randn_like(value) for key, value in predictions.items()}
        loss = head.loss(predictions, targets)
        assert loss.item() > 0


class TestLongitudinalBiomarkerHead:
    def test_output_shape(self):
        head = LongitudinalBiomarkerHead(
            in_features=128,
            biomarker_dims=10,
            hidden_dim=64,
            num_time_points=5,
        )
        x = torch.randn(2, 5, 128)  # (B, T, D)
        out = head(x)
        assert out.shape == (2, 5, 10)  # (B, T, biomarker_dims)
