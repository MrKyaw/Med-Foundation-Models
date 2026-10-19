"""Fine-tuning scripts for downstream tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any


def finetune_segmentation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Fine-tune for segmentation."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    epochs = config.get("epochs", 100)
    num_classes = config.get("num_classes", 30)

    history = {"train_loss": [], "val_dice": []}

    best_dice = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images, target_size=labels.shape[-2:])
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        val_dice = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images, target_size=labels.shape[-2:])
                preds = outputs.argmax(dim=1)

                # Compute Dice score
                for c in range(1, num_classes):
                    pred_c = (preds == c).float()
                    label_c = (labels == c).float()
                    intersection = (pred_c * label_c).sum()
                    dice = (2 * intersection) / (pred_c.sum() + label_c.sum() + 1e-8)
                    val_dice += dice.item()

                num_val_batches += 1

        avg_dice = val_dice / (num_val_batches * (num_classes - 1))
        history["val_dice"].append(avg_dice)

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), "best_segmentation_model.pth")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Dice: {avg_dice:.4f}")

    return history


def finetune_classification(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Fine-tune for classification."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    epochs = config.get("epochs", 50)
    num_classes = config.get("num_classes", 10)

    history = {"train_loss": [], "val_acc": []}

    best_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        history["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_classification_model.pth")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

    return history


def finetune_detection(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Fine-tune for detection."""
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    epochs = config.get("epochs", 100)

    history = {"train_loss": [], "val_map": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            optimizer.zero_grad()
            cls_outputs, reg_outputs, obj_outputs = model(images)

            # Simplified detection loss (in practice, use proper matching)
            loss = sum([o.mean() for o in obj_outputs])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    return history
