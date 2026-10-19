"""Tests for Multimodal Fusion."""

import pytest
import torch
from medfm.models.multimodal import MultimodalMRI, CrossAttention, GatedFusion


class TestCrossAttention:
    def test_output_shape(self):
        attn = CrossAttention(dim=128, num_heads=4)
        query = torch.randn(2, 16, 128)
        context = torch.randn(2, 8, 128)
        out = attn(query, context)
        assert out.shape == (2, 16, 128)


class TestGatedFusion:
    def test_output_shape(self):
        fusion = GatedFusion(dim=128)
        x1 = torch.randn(2, 16, 128)
        x2 = torch.randn(2, 16, 128)
        out = fusion(x1, x2)
        assert out.shape == (2, 16, 128)


class TestMultimodalMRI:
    @pytest.fixture
    def multimodal_small(self):
        return MultimodalMRI(
            img_size=64,
            patch_size=16,
            in_channels=1,
            embed_dim=128,
            depth=2,
            num_heads=4,
            clinical_dim=64,
            radiology_dim=128,
            lab_dim=32,
            num_classes=5,
            fusion_type="cross_attention",
        )

    def test_forward_images_only(self, multimodal_small):
        images = torch.randn(2, 1, 64, 64)
        out = multimodal_small(images)
        assert out.shape == (2, 5)

    def test_forward_with_clinical(self, multimodal_small):
        images = torch.randn(2, 1, 64, 64)
        clinical = torch.randn(2, 64)
        out = multimodal_small(images, clinical=clinical)
        assert out.shape == (2, 5)

    def test_forward_with_all_modalities(self, multimodal_small):
        images = torch.randn(2, 1, 64, 64)
        clinical = torch.randn(2, 64)
        radiology = torch.randn(2, 128)
        lab = torch.randn(2, 32)
        out = multimodal_small(images, clinical=clinical, radiology=radiology, lab=lab)
        assert out.shape == (2, 5)

    def test_gated_fusion(self):
        model = MultimodalMRI(
            img_size=64,
            patch_size=16,
            in_channels=1,
            embed_dim=128,
            depth=2,
            num_heads=4,
            clinical_dim=64,
            radiology_dim=128,
            lab_dim=32,
            num_classes=5,
            fusion_type="gated",
        )
        images = torch.randn(2, 1, 64, 64)
        out = model(images)
        assert out.shape == (2, 5)

    def test_concat_fusion(self):
        model = MultimodalMRI(
            img_size=64,
            patch_size=16,
            in_channels=1,
            embed_dim=128,
            depth=2,
            num_heads=4,
            clinical_dim=64,
            radiology_dim=128,
            lab_dim=32,
            num_classes=5,
            fusion_type="concat",
        )
        images = torch.randn(2, 1, 64, 64)
        out = model(images)
        assert out.shape == (2, 5)
