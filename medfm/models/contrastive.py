"""Contrastive learning methods: SimCLR and DINO for MRI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .vit import VisionTransformer


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


class SimCLR(nn.Module):
    """SimCLR for self-supervised MRI representation learning."""

    def __init__(
        self,
        backbone: Optional[VisionTransformer] = None,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        projection_dim: int = 128,
        temperature: float = 0.07,
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
            )
        self.backbone = backbone
        self.projector = ProjectionHead(embed_dim, 2048, projection_dim)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        z = self.projector(features)
        z = F.normalize(z, dim=1)
        return z

    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss for SimCLR with numerical stability."""
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        # Positive pairs
        pos_sim = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ], dim=0)

        # Compute loss with numerical stability using log-sum-exp
        # For each row, compute log(sum(exp(sim))) - sim_positive
        sim_max = sim_matrix.max(dim=1, keepdim=True)[0]
        sim_matrix_stable = sim_matrix - sim_max
        exp_sim = sim_matrix_stable.exp()
        exp_sim = exp_sim.masked_fill(mask, 0.0)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True)) + sim_max
        pos_sim_stable = pos_sim.unsqueeze(1)
        loss = (log_sum_exp - pos_sim_stable).squeeze(1)
        return loss.mean()


class DINOHead(nn.Module):
    """DINO projection head with centering and sharpening."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=1)
        x = self.last_layer(x)
        return x


class DINO(nn.Module):
    """DINO for self-supervised MRI representation learning."""

    def __init__(
        self,
        backbone: Optional[VisionTransformer] = None,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        out_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
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
            )

        self.student_backbone = backbone
        self.student_head = DINOHead(embed_dim, out_dim)

        self.teacher_backbone = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=0,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.teacher_head = DINOHead(embed_dim, out_dim)

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Initialize teacher as copy of student
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher."""
        for param_q, param_k in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            param_k.data = param_k.data * self.center_momentum + param_q.data * (1.0 - self.center_momentum)
        for param_q, param_k in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            param_k.data = param_k.data * self.center_momentum + param_q.data * (1.0 - self.center_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        student_features = self.student_backbone.forward_features(x)
        student_out = self.student_head(student_features)
        return student_out

    def dino_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """DINO cross-entropy loss."""
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(2)

        teacher_out = (teacher_output - self.center) / self.teacher_temp
        teacher_out = F.softmax(teacher_out, dim=1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        for i in range(len(teacher_out)):
            for j in range(len(student_out)):
                if i == j:
                    continue
                loss = torch.sum(-teacher_out[i] * F.log_softmax(student_out[j], dim=1), dim=1)
                total_loss += loss.mean()

        # Update center
        self.center = self.center * self.center_momentum + teacher_output.mean(0) * (1.0 - self.center_momentum)

        return total_loss / (len(student_out) * (len(student_out) - 1))
