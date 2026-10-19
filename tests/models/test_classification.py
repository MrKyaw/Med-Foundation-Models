"""Tests for Classification heads."""

import pytest
import torch
from medfm.models.classification import ClassificationHead, MultiLabelClassificationHead, HierarchicalClassificationHead


class TestClassificationHead:
    def test_output_shape(self):
        head = ClassificationHead(in_features=128, num_classes=5, hidden_dim=64)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 5)

    def test_multi_layer(self):
        head = ClassificationHead(
            in_features=128,
            num_classes=5,
            hidden_dim=64,
            num_layers=3,
        )
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 5)


class TestMultiLabelClassificationHead:
    def test_output_shape(self):
        head = MultiLabelClassificationHead(in_features=128, num_classes=5, hidden_dim=64)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 5)

    def test_loss(self):
        head = MultiLabelClassificationHead(in_features=128, num_classes=5, hidden_dim=64)
        x = torch.randn(4, 128)
        logits = head(x)
        targets = torch.randint(0, 2, (4, 5))
        loss = head.loss(logits, targets)
        assert loss.item() > 0


class TestHierarchicalClassificationHead:
    def test_output_shapes(self):
        head = HierarchicalClassificationHead(
            in_features=128,
            num_organ_systems=5,
            num_diseases=10,
            hidden_dim=64,
        )
        x = torch.randn(2, 128)
        organ_logits, disease_logits = head(x)
        assert organ_logits.shape == (2, 5)
        assert disease_logits.shape == (2, 10)
