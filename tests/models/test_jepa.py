"""Tests for JEPA."""

import pytest
import torch
from medfm.models.jepa import JEPAViT


class TestJEPAViT:
    @pytest.fixture
    def jepa_small(self):
        return JEPAViT(
            img_size=64,
            patch_size=16,
            in_channels=1,
            embed_dim=128,
            depth=2,
            num_heads=4,
            predictor_embed_dim=64,
            predictor_depth=1,
            predictor_num_heads=4,
            mask_ratio=0.75,
        )

    def test_generate_masks(self, jepa_small):
        context_ids, target_ids, ids_restore = jepa_small.generate_masks(2, torch.device("cpu"))
        assert context_ids.shape == (2, 4)  # 25% of 16 patches
        assert target_ids.shape == (2, 12)  # 75% of 16 patches
        assert ids_restore.shape == (2, 16)

    def test_forward_context(self, jepa_small):
        x = torch.randn(2, 1, 64, 64)
        context_ids, _, _ = jepa_small.generate_masks(2, torch.device("cpu"))
        out = jepa_small.forward_context(x, context_ids)
        assert out.shape == (2, 5, 128)  # 4 context + 1 cls

    def test_forward_target(self, jepa_small):
        x = torch.randn(2, 1, 64, 64)
        _, target_ids, _ = jepa_small.generate_masks(2, torch.device("cpu"))
        out = jepa_small.forward_target(x, target_ids)
        assert out.shape == (2, 13, 128)  # 12 target + 1 cls

    def test_forward_predictor(self, jepa_small):
        # Predictor takes all patch embeddings and target_ids
        all_patches = torch.randn(2, 16, 128)  # 16 patch tokens
        _, target_ids, _ = jepa_small.generate_masks(2, torch.device("cpu"))
        out = jepa_small.forward_predictor(all_patches, None, target_ids)
        assert out.shape == (2, 12, 64)

    def test_forward_loss(self, jepa_small):
        x = torch.randn(2, 1, 64, 64)
        loss = jepa_small(x)
        assert loss.item() > 0

    def test_target_encoder_no_grad(self, jepa_small):
        # Target encoder params should not require grad
        for param in jepa_small.target_blocks.parameters():
            assert not param.requires_grad
        for param in jepa_small.target_norm.parameters():
            assert not param.requires_grad

    def test_update_target_encoder(self, jepa_small):
        # Modify context encoder
        with torch.no_grad():
            for param in jepa_small.context_blocks.parameters():
                param.add_(0.1)

        jepa_small.update_target_encoder(momentum=0.0)
        # After momentum=0, target should equal context
        for param_k, param_q in zip(
            jepa_small.target_blocks.parameters(), jepa_small.context_blocks.parameters()
        ):
            assert torch.allclose(param_k, param_q)
