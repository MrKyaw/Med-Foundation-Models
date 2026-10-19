"""Joint Embedding Predictive Architecture (JEPA) for MRI."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .vit import PatchEmbed, Block, _init_vit_weights


class JEPAViT(nn.Module):
    """I-JEPA style architecture for self-supervised MRI representation learning."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 8,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Context encoder
        self.context_patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.context_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.context_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.context_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.context_norm = nn.LayerNorm(embed_dim)

        # Target encoder (EMA of context encoder)
        self.target_patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.target_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.target_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.target_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.target_norm = nn.LayerNorm(embed_dim)

        # Predictor
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, predictor_embed_dim))
        self.predictor_blocks = nn.ModuleList([
            Block(predictor_embed_dim, predictor_num_heads, mlp_ratio) for _ in range(predictor_depth)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)

        self._init_weights()
        self._initialize_target_encoder()

    def _init_weights(self):
        nn.init.trunc_normal_(self.context_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.context_cls_token, std=0.02)
        nn.init.trunc_normal_(self.target_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.target_cls_token, std=0.02)
        nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)
        self.apply(_init_vit_weights)

    def _initialize_target_encoder(self):
        """Initialize target encoder as copy of context encoder."""
        self.target_patch_embed.load_state_dict(self.context_patch_embed.state_dict())
        self.target_cls_token.data.copy_(self.context_cls_token.data)
        self.target_pos_embed.data.copy_(self.context_pos_embed.data)
        self.target_blocks.load_state_dict(self.context_blocks.state_dict())
        self.target_norm.load_state_dict(self.context_norm.state_dict())
        for param in self.target_blocks.parameters():
            param.requires_grad = False
        for param in self.target_norm.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996):
        """EMA update of target encoder."""
        for param_k, param_q in zip(
            self.target_patch_embed.parameters(), self.context_patch_embed.parameters()
        ):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)
        self.target_cls_token.data = self.target_cls_token.data * momentum + self.context_cls_token.data * (1.0 - momentum)
        self.target_pos_embed.data = self.target_pos_embed.data * momentum + self.context_pos_embed.data * (1.0 - momentum)
        for param_k, param_q in zip(
            self.target_blocks.parameters(), self.context_blocks.parameters()
        ):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)
        for param_k, param_q in zip(
            self.target_norm.parameters(), self.context_norm.parameters()
        ):
            param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)

    def generate_masks(self, batch_size: int, device: torch.device):
        """Generate context and target masks."""
        num_patches = self.num_patches
        num_mask = int(num_patches * self.mask_ratio)
        num_keep = num_patches - num_mask

        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        context_ids = ids_shuffle[:, :num_keep]
        target_ids = ids_shuffle[:, num_keep:]

        return context_ids, target_ids, ids_restore

    def forward_context(self, x: torch.Tensor, context_ids: torch.Tensor):
        """Forward pass through context encoder."""
        B = x.shape[0]
        x = self.context_patch_embed(x)
        x = x + self.context_pos_embed[:, 1:, :]

        x = torch.gather(x, dim=1, index=context_ids.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        cls_token = self.context_cls_token + self.context_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.context_blocks:
            x = block(x)
        x = self.context_norm(x)
        return x

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor, target_ids: torch.Tensor):
        """Forward pass through target encoder."""
        B = x.shape[0]
        x = self.target_patch_embed(x)
        x = x + self.target_pos_embed[:, 1:, :]

        x = torch.gather(x, dim=1, index=target_ids.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        cls_token = self.target_cls_token + self.target_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.target_blocks:
            x = block(x)
        x = self.target_norm(x)
        return x

    def forward_predictor(self, x: torch.Tensor, context_ids: torch.Tensor, target_ids: torch.Tensor):
        """Forward pass through predictor."""
        # x contains all patches (after embedding)
        x = self.predictor_embed(x)

        # Select target positions
        x = torch.gather(x, dim=1, index=target_ids.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # Add positional embedding for target positions
        pos_embed = torch.gather(
            self.predictor_pos_embed.expand(x.shape[0], -1, -1),
            dim=1,
            index=target_ids.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = x + pos_embed

        for block in self.predictor_blocks:
            x = block(x)
        x = self.predictor_norm(x)
        return x

    def forward(self, imgs: torch.Tensor):
        """Forward pass with JEPA loss."""
        B = imgs.shape[0]
        context_ids, target_ids, _ = self.generate_masks(B, imgs.device)

        # Get all patch embeddings first
        all_patches = self.context_patch_embed(imgs)
        all_patches = all_patches + self.context_pos_embed[:, 1:, :]

        # Forward context (visible patches only)
        context_patches = torch.gather(all_patches, dim=1, index=context_ids.unsqueeze(-1).expand(-1, -1, all_patches.shape[2]))
        cls_token = self.context_cls_token + self.context_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        context_features = torch.cat((cls_tokens, context_patches), dim=1)
        for block in self.context_blocks:
            context_features = block(context_features)
        context_features = self.context_norm(context_features)

        # Forward target (all patches, no grad)
        with torch.no_grad():
            target_patches = self.target_patch_embed(imgs)
            target_patches = target_patches + self.target_pos_embed[:, 1:, :]
            target_selected = torch.gather(target_patches, dim=1, index=target_ids.unsqueeze(-1).expand(-1, -1, target_patches.shape[2]))
            target_cls = self.target_cls_token + self.target_pos_embed[:, :1, :]
            target_cls_tokens = target_cls.expand(B, -1, -1)
            target_features = torch.cat((target_cls_tokens, target_selected), dim=1)
            for block in self.target_blocks:
                target_features = block(target_features)
            target_features = self.target_norm(target_features)

        # Forward predictor using all patch embeddings
        pred_features = self.forward_predictor(all_patches, context_ids, target_ids)

        # Project pred_features to match target_features dimension
        if pred_features.shape[-1] != target_features.shape[-1]:
            proj = nn.Linear(pred_features.shape[-1], target_features.shape[-1]).to(pred_features.device)
            pred_features = proj(pred_features)

        # Ensure same number of tokens for loss computation
        min_tokens = min(pred_features.shape[1], target_features.shape[1])
        pred_features = pred_features[:, :min_tokens, :]
        target_features = target_features[:, :min_tokens, :]

        # Ensure same shape for loss computation
        min_tokens = min(pred_features.shape[1], target_features.shape[1])
        pred_features = pred_features[:, :min_tokens, :]
        target_features = target_features[:, :min_tokens, :]
        loss = ((pred_features - target_features) ** 2).mean()
        return loss
