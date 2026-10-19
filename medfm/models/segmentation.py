"""Segmentation heads for medical image analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Up(nn.Module):
    """Upsampling block."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegmentationHead(nn.Module):
    """Segmentation head for ViT backbone features."""

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 30,
        features: List[int] = [64, 128, 256, 512],
        bilinear: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Project ViT features to spatial map
        self.feature_proj = nn.Conv2d(in_channels, features[-1], kernel_size=1)

        # Store feature channels for skip connections
        self.features = features

        # Decoder - each Up block takes in_channels + skip_channels and produces out_channels
        self.up1 = Up(features[-1], features[-2], bilinear)
        self.up2 = Up(features[-2], features[-3], bilinear)
        self.up3 = Up(features[-3], features[-4], bilinear)

        # Final upsampling
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(features[-4], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        # Project features
        x = self.feature_proj(x)

        # Decode with proper skip connections
        # Each Up block upsamples x1 and concatenates with x2 (skip)
        # Project skip connections to match expected channels
        f = self.features
        skip1 = nn.Conv2d(x.shape[1], f[-2], kernel_size=1).to(x.device)(x)
        x1 = self.up1(x, skip1)  # 512 -> 256
        skip2 = nn.Conv2d(x1.shape[1], f[-3], kernel_size=1).to(x1.device)(x1)
        x2 = self.up2(x1, skip2)  # 256 -> 128
        skip3 = nn.Conv2d(x2.shape[1], f[-4], kernel_size=1).to(x2.device)(x2)
        x3 = self.up3(x2, skip3)  # 128 -> 64
        x4 = self.final_up(x3)
        x4 = self.final_conv(x4)

        # Resize to target size if specified
        if target_size is not None:
            x4 = F.interpolate(x4, size=target_size, mode="bilinear", align_corners=True)

        return x4


class UNetDecoder(nn.Module):
    """Full U-Net decoder for dense prediction tasks."""

    def __init__(
        self,
        encoder_channels: List[int] = [768, 768, 768, 768],
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_classes: int = 30,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]
        for out_ch in decoder_channels:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

        # Final classifier
        self.classifier = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], target_size: Optional[tuple] = None) -> torch.Tensor:
        x = features[-1]

        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            if i < len(features) - 1:
                skip = features[-(i + 2)]
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
                # Project skip to match channels if needed
                if skip.shape[1] != x.shape[1]:
                    skip_proj = nn.Conv2d(skip.shape[1], x.shape[1], kernel_size=1).to(x.device)
                    skip = skip_proj(skip)
                x = x + skip

        x = self.classifier(x)

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)

        return x
