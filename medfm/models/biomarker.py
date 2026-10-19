"""Biomarker extraction heads for quantitative medical analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class BiomarkerHead(nn.Module):
    """Biomarker extraction head for quantitative MRI analysis."""

    def __init__(
        self,
        in_features: int = 768,
        biomarker_dims: int = 50,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.biomarker_dims = biomarker_dims

        self.extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, biomarker_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE loss for biomarker regression."""
        return F.mse_loss(predictions, targets)


class MultiTaskBiomarkerHead(nn.Module):
    """Multi-task biomarker extraction with separate heads for different biomarker types."""

    def __init__(
        self,
        in_features: int = 768,
        body_composition_dims: int = 10,
        organ_volume_dims: int = 20,
        fat_fraction_dims: int = 10,
        vascular_dims: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        shared_dim = hidden_dim // 2

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.body_composition_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, body_composition_dims),
        )

        self.organ_volume_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, organ_volume_dims),
        )

        self.fat_fraction_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, fat_fraction_dims),
        )

        self.vascular_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
            nn.Linear(shared_dim // 2, vascular_dims),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared(x)

        return {
            "body_composition": self.body_composition_head(shared_features),
            "organ_volumes": self.organ_volume_head(shared_features),
            "fat_fractions": self.fat_fraction_head(shared_features),
            "vascular": self.vascular_head(shared_features),
        }

    def loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combined loss for all biomarker tasks."""
        total_loss = 0
        for key in predictions:
            if key in targets:
                total_loss += F.mse_loss(predictions[key], targets[key])
        return total_loss / len(predictions)


class LongitudinalBiomarkerHead(nn.Module):
    """Longitudinal biomarker extraction for tracking changes over time."""

    def __init__(
        self,
        in_features: int = 768,
        biomarker_dims: int = 50,
        hidden_dim: int = 512,
        num_time_points: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_time_points = num_time_points

        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Biomarker prediction for each time point
        self.biomarker_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, biomarker_dims),
            )
            for _ in range(num_time_points)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) where T is number of time points, D is feature dim
        Returns:
            biomarkers: (B, T, biomarker_dims)
        """
        B, T, D = x.shape

        # Encode temporal sequence
        temporal_features, _ = self.temporal_encoder(x)

        # Predict biomarkers for each time point
        outputs = []
        for i in range(T):
            biomarker = self.biomarker_predictors[i](temporal_features[:, i, :])
            outputs.append(biomarker)

        return torch.stack(outputs, dim=1)
