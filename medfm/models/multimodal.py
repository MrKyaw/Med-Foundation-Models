"""Multimodal fusion models for MRI + clinical data integration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .vit import VisionTransformer


class CrossAttention(nn.Module):
    """Cross-attention module for multimodal fusion."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = query.shape
        M = context.shape[1]

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.norm(out + query)
        return out


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining modalities."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x1, x2], dim=-1)
        gate = self.gate(combined)
        fused = self.fusion(combined)
        return gate * x1 + (1 - gate) * fused


class MultimodalMRI(nn.Module):
    """Multimodal model fusing MRI imaging with clinical data."""

    def __init__(
        self,
        backbone: Optional[VisionTransformer] = None,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        clinical_dim: int = 256,
        radiology_dim: int = 768,
        lab_dim: int = 128,
        num_classes: int = 10,
        fusion_type: str = "cross_attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        if backbone is None:
            backbone = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                num_classes=0,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                drop_rate=dropout,
            )

        self.backbone = backbone
        self.fusion_type = fusion_type

        # Clinical data encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Radiology report encoder (assumes pre-computed embeddings)
        self.radiology_proj = nn.Sequential(
            nn.Linear(radiology_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Lab values encoder
        self.lab_encoder = nn.Sequential(
            nn.Linear(lab_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Fusion modules
        if fusion_type == "cross_attention":
            self.fusion = CrossAttention(embed_dim, num_heads, dropout)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(embed_dim)
        elif fusion_type == "concat":
            self.fusion_proj = nn.Linear(embed_dim * 4, embed_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        radiology: Optional[torch.Tensor] = None,
        lab: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Extract imaging features
        imaging_features = self.backbone.forward_features(images)

        # Encode clinical data
        features = [imaging_features]

        if clinical is not None:
            clinical_features = self.clinical_encoder(clinical)
            features.append(clinical_features)

        if radiology is not None:
            radiology_features = self.radiology_proj(radiology)
            features.append(radiology_features)

        if lab is not None:
            lab_features = self.lab_encoder(lab)
            features.append(lab_features)

        # Fuse features
        if self.fusion_type == "cross_attention":
            fused = features[0]
            for f in features[1:]:
                if f.ndim == 2:
                    f = f.unsqueeze(1)
                if fused.ndim == 2:
                    fused = fused.unsqueeze(1)
                fused = self.fusion(fused, f)
                if fused.ndim == 3:
                    fused = fused.squeeze(1)
        elif self.fusion_type == "gated":
            fused = features[0]
            for f in features[1:]:
                if f.ndim == 2:
                    f = f.unsqueeze(1)
                if fused.ndim == 2:
                    fused = fused.unsqueeze(1)
                fused = self.fusion(fused, f)
                if fused.ndim == 3:
                    fused = fused.squeeze(1)
        elif self.fusion_type == "concat":
            if len(features) == 1:
                fused = features[0]
            else:
                flat_features = []
                for f in features:
                    if f.ndim == 3:
                        flat_features.append(f.squeeze(1))
                    else:
                        flat_features.append(f)
                fused = torch.cat(flat_features, dim=-1)
                fused = self.fusion_proj(fused)
        else:
            fused = torch.stack(features, dim=1).mean(dim=1)

        # Classify
        output = self.classifier(fused)
        return output
