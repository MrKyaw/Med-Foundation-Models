"""Tests for Vision Transformer."""

import pytest
import torch
from medfm.models.vit import VisionTransformer, PatchEmbed, Attention, MLP, Block


class TestPatchEmbed:
    def test_output_shape(self, device):
        embed = PatchEmbed(img_size=64, patch_size=16, in_channels=1, embed_dim=128)
        x = torch.randn(2, 1, 64, 64)
        out = embed(x)
        assert out.shape == (2, 16, 128)

    def test_different_patch_sizes(self, device):
        for patch_size in [8, 16, 32]:
            embed = PatchEmbed(img_size=64, patch_size=patch_size, in_channels=1, embed_dim=64)
            x = torch.randn(1, 1, 64, 64)
            out = embed(x)
            expected_patches = (64 // patch_size) ** 2
            assert out.shape == (1, expected_patches, 64)

    def test_multichannel_input(self, device):
        embed = PatchEmbed(img_size=64, patch_size=16, in_channels=3, embed_dim=128)
        x = torch.randn(2, 3, 64, 64)
        out = embed(x)
        assert out.shape == (2, 16, 128)


class TestAttention:
    def test_output_shape(self, device):
        attn = Attention(dim=128, num_heads=4)
        x = torch.randn(2, 16, 128)
        out = attn(x)
        assert out.shape == (2, 16, 128)

    def test_different_num_heads(self, device):
        for num_heads in [1, 2, 4, 8]:
            attn = Attention(dim=128, num_heads=num_heads)
            x = torch.randn(1, 8, 128)
            out = attn(x)
            assert out.shape == (1, 8, 128)


class TestMLP:
    def test_output_shape(self, device):
        mlp = MLP(in_features=128, hidden_features=256, out_features=64)
        x = torch.randn(2, 16, 128)
        out = mlp(x)
        assert out.shape == (2, 16, 64)


class TestBlock:
    def test_output_shape(self, device):
        block = Block(dim=128, num_heads=4, mlp_ratio=4.0)
        x = torch.randn(2, 16, 128)
        out = block(x)
        assert out.shape == (2, 16, 128)


class TestVisionTransformer:
    @pytest.fixture
    def vit_small(self):
        return VisionTransformer(
            img_size=64,
            patch_size=16,
            in_channels=1,
            num_classes=10,
            embed_dim=128,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        )

    def test_forward_shape(self, vit_small, device):
        x = torch.randn(2, 1, 64, 64)
        out = vit_small(x)
        assert out.shape == (2, 10)

    def test_forward_features_shape(self, vit_small, device):
        x = torch.randn(2, 1, 64, 64)
        out = vit_small.forward_features(x)
        assert out.shape == (2, 128)

    def test_no_classification_head(self, device):
        vit = VisionTransformer(
            img_size=64,
            patch_size=16,
            in_channels=1,
            num_classes=0,
            embed_dim=128,
            depth=2,
            num_heads=4,
        )
        x = torch.randn(2, 1, 64, 64)
        out = vit(x)
        assert out.shape == (2, 128)

    def test_get_num_layers(self, vit_small):
        assert vit_small.get_num_layers() == 2

    def test_gradient_flow(self, vit_small, device):
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        out = vit_small(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
