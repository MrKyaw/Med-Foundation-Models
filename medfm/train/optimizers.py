"""Optimizer and scheduler configurations."""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR
from typing import Dict, Any, Optional


def get_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> Optimizer:
    """Create optimizer from config."""
    name = config.get("name", "adamw").lower()
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)

    if name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizer


def get_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    total_steps: int,
) -> LambdaLR:
    """Create learning rate scheduler from config."""
    name = config.get("name", "cosine").lower()
    warmup_steps = config.get("warmup_steps", 0)

    if name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    elif name == "step":
        step_size = config.get("step_size", 30)
        gamma = config.get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    elif name == "warmup_cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    return scheduler


class WarmupCosineScheduler:
    """Warmup + Cosine annealing scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        """Update learning rate."""
        self.step_num += 1

        if self.step_num < self.warmup_steps:
            # Warmup phase
            lr_scale = self.step_num / max(1, self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.step_num - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            lr_scale = max(lr_scale, self.min_lr / max(self.base_lrs))

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * lr_scale

    def get_last_lr(self):
        """Get last learning rate."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Get state dict."""
        return {"step_num": self.step_num}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.step_num = state_dict["step_num"]
