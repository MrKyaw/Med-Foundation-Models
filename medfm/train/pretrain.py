"""Self-supervised pre-training scripts."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any


def pretrain_mae(
    model: nn.Module,
    train_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Pre-train Masked Autoencoder."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1.5e-4),
        weight_decay=config.get("weight_decay", 0.05),
    )

    epochs = config.get("epochs", 800)
    mask_ratio = config.get("mask_ratio", 0.75)

    history = {"loss": [], "epoch": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            optimizer.zero_grad()
            loss, _, _ = model(images, mask_ratio)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["loss"].append(avg_loss)
        history["epoch"].append(epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    return history


def pretrain_jepa(
    model: nn.Module,
    train_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Pre-train JEPA."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1.5e-4),
        weight_decay=config.get("weight_decay", 0.05),
    )

    epochs = config.get("epochs", 600)
    momentum = config.get("momentum", 0.996)

    history = {"loss": [], "epoch": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            optimizer.zero_grad()
            loss = model(images)
            loss.backward()
            optimizer.step()

            # Update target encoder
            model.update_target_encoder(momentum)

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["loss"].append(avg_loss)
        history["epoch"].append(epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    return history


def pretrain_simclr(
    model: nn.Module,
    train_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Pre-train SimCLR."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-6),
    )

    epochs = config.get("epochs", 1000)
    temperature = config.get("temperature", 0.07)

    history = {"loss": [], "epoch": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            # Create two augmented views
            # In practice, use two different augmentations
            z_i = model(images)
            z_j = model(images)

            loss = model.nt_xent_loss(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["loss"].append(avg_loss)
        history["epoch"].append(epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    return history


def pretrain_dino(
    model: nn.Module,
    train_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Pre-train DINO."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 5e-4),
        weight_decay=config.get("weight_decay", 0.04),
    )

    epochs = config.get("epochs", 800)

    history = {"loss": [], "epoch": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_output = model.teacher["head"](
                    model.teacher["backbone"].forward_features(images)
                )

            # Student forward
            student_output = model(images)

            loss = model.dino_loss(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update teacher
            model.update_teacher()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["loss"].append(avg_loss)
        history["epoch"].append(epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    return history
