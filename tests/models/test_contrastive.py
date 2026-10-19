"""Tests for Contrastive Learning methods."""

import pytest
import torch
from medfm.models.contrastive import SimCLR, DINO, ProjectionHead


class TestProjectionHead:
    def test_output_shape(self):
        head = ProjectionHead(in_dim=128, hidden_dim=256, out_dim=64)
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 64)


class TestSimCLR:
    @pytest.fixture
    def simclr_small(self):
        return SimCLR(
            img_size=64,
            patch_size=16,
            in_channels=1,
            embed_dim=128,
            depth=2,
            num_heads=4,
            projection_dim=64,
            temperature=0.07,
        )

    def test_forward_shape(self, simclr_small):
        x = torch.randn(2, 1, 64, 64)
        out = simclr_small(x)
        assert out.shape == (2, 64)

    def test_nt_xent_loss(self, simclr_small):
        z_i = torch.randn(4, 64)
        z_j = torch.randn(4, 64)
        loss = simclr_small.nt_xent_loss(z_i, z_j)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_output_normalized(self, simclr_small):
        x = torch.randn(2, 1, 64, 64)
        out = simclr_small(x)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestDINO:
    @pytest.fixture
    def dino_small(self):
        return DINO(
            img_size=64,
            patch_size=16,
            in_channels=1,
            embed_dim=128,
            depth=2,
            num_heads=4,
            out_dim=256,
            teacher_temp=0.04,
            student_temp=0.1,
        )

    def test_forward_shape(self, dino_small):
        x = torch.randn(2, 1, 64, 64)
        out = dino_small(x)
        assert out.shape == (2, 256)

    def test_teacher_no_grad(self, dino_small):
        for param in dino_small.teacher_backbone.parameters():
            assert not param.requires_grad
        for param in dino_small.teacher_head.parameters():
            assert not param.requires_grad

    def test_dino_loss(self, dino_small):
        student_out = torch.randn(4, 256)
        teacher_out = torch.randn(4, 256)
        loss = dino_small.dino_loss(student_out, teacher_out)
        assert loss.item() > 0

    def test_update_teacher(self, dino_small):
        # Modify student
        with torch.no_grad():
            for param in dino_small.student_backbone.parameters():
                param.add_(0.1)

        old_teacher_params = [p.clone() for p in dino_small.teacher_backbone.parameters()]
        dino_small.update_teacher()

        # Teacher should have changed
        for old_p, new_p in zip(old_teacher_params, dino_small.teacher_backbone.parameters()):
            assert not torch.allclose(old_p, new_p)
