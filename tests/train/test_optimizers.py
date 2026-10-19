"""Tests for optimizer and scheduler configurations."""

import pytest
import torch
from medfm.train.optimizers import get_optimizer, get_scheduler, WarmupCosineScheduler


class TestGetOptimizer:
    @pytest.fixture
    def model(self):
        return torch.nn.Linear(10, 2)

    def test_adamw(self, model):
        config = {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4}
        optimizer = get_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_adam(self, model):
        config = {"name": "adam", "lr": 1e-3, "weight_decay": 1e-4}
        optimizer = get_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_sgd(self, model):
        config = {"name": "sgd", "lr": 1e-2, "weight_decay": 1e-4}
        optimizer = get_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_invalid_optimizer(self, model):
        with pytest.raises(ValueError):
            get_optimizer(model, {"name": "invalid"})


class TestGetScheduler:
    @pytest.fixture
    def model(self):
        return torch.nn.Linear(10, 2)

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_cosine_scheduler(self, optimizer):
        config = {"name": "cosine"}
        scheduler = get_scheduler(optimizer, config, total_steps=100)
        assert scheduler is not None

    def test_step_scheduler(self, optimizer):
        config = {"name": "step", "step_size": 30, "gamma": 0.1}
        scheduler = get_scheduler(optimizer, config, total_steps=100)
        assert scheduler is not None

    def test_constant_scheduler(self, optimizer):
        config = {"name": "constant"}
        scheduler = get_scheduler(optimizer, config, total_steps=100)
        assert scheduler is not None

    def test_warmup_cosine_scheduler(self, optimizer):
        config = {"name": "warmup_cosine", "warmup_steps": 10}
        scheduler = get_scheduler(optimizer, config, total_steps=100)
        assert scheduler is not None


class TestWarmupCosineScheduler:
    def test_warmup_phase(self):
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100)

        initial_lrs = [group["lr"] for group in optimizer.param_groups]
        for _ in range(5):
            scheduler.step()

        # During warmup, LR should be increasing
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        assert all(c > i * 0.4 for c, i in zip(current_lrs, initial_lrs))

    def test_cosine_phase(self):
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100)

        # Skip warmup
        for _ in range(10):
            scheduler.step()

        peak_lrs = [group["lr"] for group in optimizer.param_groups]

        # Continue through cosine phase
        for _ in range(50):
            scheduler.step()

        current_lrs = [group["lr"] for group in optimizer.param_groups]
        assert all(c < p for c, p in zip(current_lrs, peak_lrs))

    def test_state_dict(self):
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=10, total_steps=100)

        for _ in range(5):
            scheduler.step()

        state = scheduler.state_dict()
        assert state["step_num"] == 5
