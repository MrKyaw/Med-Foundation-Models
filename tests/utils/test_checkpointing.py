"""Tests for checkpointing utilities."""

import pytest
import torch
import os
from pathlib import Path
from medfm.utils.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    save_model_weights,
    load_model_weights,
)


class TestSaveCheckpoint:
    @pytest.fixture
    def model(self):
        return torch.nn.Linear(10, 2)

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_save_creates_file(self, model, optimizer, tmp_path):
        save_checkpoint(
            model, optimizer, None, epoch=0, best_metric=0.9,
            config={"lr": 1e-3}, checkpoint_dir=str(tmp_path), filename="test.pth"
        )
        assert (tmp_path / "test.pth").exists()

    def test_save_best_model(self, model, optimizer, tmp_path):
        save_checkpoint(
            model, optimizer, None, epoch=0, best_metric=0.9,
            config={"lr": 1e-3}, checkpoint_dir=str(tmp_path),
            filename="test.pth", is_best=True
        )
        assert (tmp_path / "best_model.pth").exists()


class TestLoadCheckpoint:
    @pytest.fixture
    def model(self):
        return torch.nn.Linear(10, 2)

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_load_restores_state(self, model, optimizer, tmp_path):
        # Save checkpoint
        save_checkpoint(
            model, optimizer, None, epoch=5, best_metric=0.95,
            config={"lr": 1e-3}, checkpoint_dir=str(tmp_path), filename="test.pth"
        )

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()

        # Load checkpoint
        checkpoint = load_checkpoint(
            str(tmp_path / "test.pth"), model, device="cpu"
        )

        assert checkpoint["epoch"] == 5
        assert checkpoint["best_metric"] == 0.95

    def test_load_nonexistent_file(self, model):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent.pth", model)


class TestSaveModelWeights:
    @pytest.fixture
    def model(self):
        return torch.nn.Linear(10, 2)

    def test_save_pth(self, model, tmp_path):
        save_path = str(tmp_path / "model.pth")
        save_model_weights(model, save_path, format="pth")
        assert Path(save_path).exists()

    def test_save_pt(self, model, tmp_path):
        save_path = str(tmp_path / "model.pt")
        save_model_weights(model, save_path, format="pt")
        assert Path(save_path).exists()

    def test_invalid_format(self, model, tmp_path):
        with pytest.raises(ValueError):
            save_model_weights(model, str(tmp_path / "model.xyz"), format="xyz")


class TestLoadModelWeights:
    @pytest.fixture
    def model(self):
        return torch.nn.Linear(10, 2)

    def test_load_weights(self, model, tmp_path):
        # Save weights
        save_path = str(tmp_path / "model.pth")
        save_model_weights(model, save_path)

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()

        # Load weights
        load_model_weights(model, save_path)

        # Verify weights are restored
        for param in model.parameters():
            assert not torch.allclose(param, torch.zeros_like(param))

    def test_load_nonexistent(self, model):
        with pytest.raises(FileNotFoundError):
            load_model_weights(model, "nonexistent.pth")
