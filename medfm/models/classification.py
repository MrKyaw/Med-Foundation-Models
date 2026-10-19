"""Classification heads for disease classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ClassificationHead(nn.Module):
    """Classification head for disease classification."""

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = []
        in_dim = in_features
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else num_classes
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiLabelClassificationHead(nn.Module):
    """Multi-label classification head for multi-disease prediction."""

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss for multi-label classification."""
        return F.binary_cross_entropy_with_logits(logits, targets.float())


class HierarchicalClassificationHead(nn.Module):
    """Hierarchical classification for disease taxonomy."""

    def __init__(
        self,
        in_features: int = 768,
        num_organ_systems: int = 10,
        num_diseases: int = 50,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.organ_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_organ_systems),
        )

        self.disease_classifier = nn.Sequential(
            nn.Linear(in_features + num_organ_systems, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_diseases),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        organ_logits = self.organ_classifier(x)
        organ_probs = F.softmax(organ_logits, dim=1)

        disease_input = torch.cat([x, organ_probs], dim=1)
        disease_logits = self.disease_classifier(disease_input)

        return organ_logits, disease_logits
