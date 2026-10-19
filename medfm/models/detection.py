"""Detection heads for lesion and abnormality detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class ConvBlock(nn.Module):
    """Convolution block for detection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale detection."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(ch, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        results = []
        prev_feature = self.lateral_convs[-1](features[-1])
        results.append(self.output_convs[-1](prev_feature))

        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            upsampled = F.interpolate(prev_feature, size=lateral.shape[-2:], mode="nearest")
            prev_feature = lateral + upsampled
            results.insert(0, self.output_convs[i](prev_feature))

        return results


class DetectionHead(nn.Module):
    """Detection head for lesion detection."""

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 5,
        num_anchors: int = 9,
        fpn_channels: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # FPN
        self.fpn = FeaturePyramidNetwork([in_channels] * 4, fpn_channels)

        # Classification head
        self.cls_head = nn.Sequential(
            ConvBlock(fpn_channels, fpn_channels),
            ConvBlock(fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, num_anchors * num_classes, kernel_size=3, padding=1),
        )

        # Regression head
        self.reg_head = nn.Sequential(
            ConvBlock(fpn_channels, fpn_channels),
            ConvBlock(fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, num_anchors * 4, kernel_size=3, padding=1),
        )

        # Objectness head
        self.obj_head = nn.Sequential(
            ConvBlock(fpn_channels, fpn_channels),
            ConvBlock(fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, num_anchors, kernel_size=3, padding=1),
        )

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract FPN features
        fpn_features = self.fpn(features)

        cls_outputs = []
        reg_outputs = []
        obj_outputs = []

        for feat in fpn_features:
            cls_outputs.append(self.cls_head(feat))
            reg_outputs.append(self.reg_head(feat))
            obj_outputs.append(self.obj_head(feat))

        return cls_outputs, reg_outputs, obj_outputs


class AnchorGenerator(nn.Module):
    """Anchor generator for detection."""

    def __init__(
        self,
        sizes: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),),
        aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def forward(self, feature_maps: List[torch.Tensor], image_size: Tuple[int, int]) -> List[torch.Tensor]:
        anchors = []
        for i, feat in enumerate(feature_maps):
            h, w = feat.shape[-2:]
            stride_h = image_size[0] / h
            stride_w = image_size[1] / w

            # Generate base anchors
            base_anchors = []
            for size in self.sizes[min(i, len(self.sizes) - 1)]:
                for ratio in self.aspect_ratios[min(i, len(self.aspect_ratios) - 1)]:
                    w_anchor = size * (ratio ** 0.5)
                    h_anchor = size / (ratio ** 0.5)
                    base_anchors.append([-w_anchor / 2, -h_anchor / 2, w_anchor / 2, h_anchor / 2])

            # Create grid
            shifts_x = torch.arange(0, w, device=feat.device) * stride_w + stride_w / 2
            shifts_y = torch.arange(0, h, device=feat.device) * stride_h + stride_h / 2
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1).reshape(-1, 4)

            # Add shifts to base anchors
            all_anchors = torch.tensor(base_anchors, device=feat.device).unsqueeze(0) + shifts.unsqueeze(1)
            anchors.append(all_anchors.reshape(-1, 4))

        return anchors
