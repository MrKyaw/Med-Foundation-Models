"""Tests for Masked Autoencoder."""

import pytest
import torch
from medfm.models.mae import MaskedAutoencoderViT


class TestMaskedAutoencoderViT:
    @pytest.fixture
    def mae_small(self):
        return MaskedAutoencoderViT(
            img_size=64,
            patch_size=16,
            in_channels=1,
            embed_dim=128,
            depth=2,
            num_heads=4,
            decoder_embed_dim=64,
            decoder_depth=1,
            decoder_num_heads=4,
            mask_ratio=0.75,
        )

    def test_patchify_shape(self, mae_small):
        imgs = torch.randn(2, 1, 64, 64)
        patches = mae_small.patchify(imgs)
        assert patches.shape == (2, 16, 256)  # 4x4 patches, 16*16=256

    def test_unpatchify_shape(self, mae_small):
        patches = torch.randn(2, 16, 256)
        imgs = mae_small.unpatchify(patches)
        assert imgs.shape == (2, 1, 64, 64)

    def test_patchify_unpatchify_roundtrip(self, mae_small):
        imgs = torch.randn(2, 1, 64, 64)
        patches = mae_small.patchify(imgs)
        reconstructed = mae_small.unpatchify(patches)
        assert torch.allclose(imgs, reconstructed)

    def test_random_masking_shapes(self, mae_small):
        x = torch.randn(2, 16, 128)  # 16 patches (no cls token)
        x_masked, mask, ids_restore = mae_small.random_masking(x, 0.75)
        assert x_masked.shape == (2, 4, 128)  # 25% of 16 = 4
        assert mask.shape == (2, 16)
        assert ids_restore.shape == (2, 16)

    def test_forward_loss(self, mae_small):
        imgs = torch.randn(2, 1, 64, 64)
        loss, pred, mask = mae_small(imgs, mask_ratio=0.75)
        assert loss.item() > 0
        assert pred.shape == (2, 16, 256)
        assert mask.shape == (2, 16)

    def test_different_mask_ratios(self, mae_small):
        imgs = torch.randn(2, 1, 64, 64)
        for mask_ratio in [0.25, 0.5, 0.75, 0.9]:
            loss, pred, mask = mae_small(imgs, mask_ratio=mask_ratio)
            assert loss.item() > 0

    def test_gradient_flow(self, mae_small):
        imgs = torch.randn(2, 1, 64, 64, requires_grad=True)
        loss, _, _ = mae_small(imgs)
        loss.backward()
        assert imgs.grad is not None
