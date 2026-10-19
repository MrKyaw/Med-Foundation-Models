"""Checkpoint saving and loading utilities."""

import os
import torch
from typing import Dict, Any, Optional
from pathlib import Path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        best_metric: Best metric achieved so far
        config: Training configuration
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "config": config,
    }

    # Save checkpoint
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state
        scheduler: Scheduler to restore state
        device: Device to load checkpoint on

    Returns:
        Dictionary with checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Checkpoint loaded: {checkpoint_path}")

    return checkpoint


def save_model_weights(
    model: torch.nn.Module,
    save_path: str,
    format: str = "pth",
):
    """Save model weights only.

    Args:
        model: Model to save
        save_path: Path to save weights
        format: File format ('pth', 'pt', 'onnx')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if format in ["pth", "pt"]:
        torch.save(model.state_dict(), save_path)
    elif format == "onnx":
        # Export to ONNX
        dummy_input = torch.randn(1, 1, 224, 224)
        torch.onnx.export(model, dummy_input, save_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Model saved: {save_path}")


def load_model_weights(
    model: torch.nn.Module,
    weights_path: str,
    device: str = "cpu",
    strict: bool = True,
):
    """Load model weights.

    Args:
        model: Model to load weights into
        weights_path: Path to weights file
        device: Device to load weights on
        strict: Whether to strictly enforce state dict matching
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)

    # Handle DataParallel checkpoints
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Remove 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)
    print(f"Weights loaded: {weights_path}")


def resume_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str = "cpu",
) -> int:
    """Resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to restore
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore
        device: Device to load checkpoint on

    Returns:
        Epoch to resume from
    """
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", 0.0)

    print(f"Resuming from epoch {start_epoch}, best metric: {best_metric:.4f}")

    return start_epoch
