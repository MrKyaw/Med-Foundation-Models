"""Tests for Detection heads."""

import pytest
import torch
from medfm.models.detection import DetectionHead, FeaturePyramidNetwork, AnchorGenerator


class TestFeaturePyramidNetwork:
    def test_output_shapes(self):
        fpn = FeaturePyramidNetwork([128, 128, 128, 128], out_channels=64)
        features = [
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
        ]
        outputs = fpn(features)
        assert len(outputs) == 4
        for out in outputs:
            assert out.shape[1] == 64


class TestDetectionHead:
    def test_output_shapes(self):
        head = DetectionHead(
            in_channels=128,
            num_classes=5,
            num_anchors=3,
            fpn_channels=64,
        )
        features = [
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
        ]
        cls_outputs, reg_outputs, obj_outputs = head(features)
        assert len(cls_outputs) == 4
        assert len(reg_outputs) == 4
        assert len(obj_outputs) == 4
        for cls_out in cls_outputs:
            assert cls_out.shape[1] == 3 * 5  # num_anchors * num_classes
        for reg_out in reg_outputs:
            assert reg_out.shape[1] == 3 * 4  # num_anchors * 4
        for obj_out in obj_outputs:
            assert obj_out.shape[1] == 3  # num_anchors


class TestAnchorGenerator:
    def test_anchor_shapes(self):
        generator = AnchorGenerator(
            sizes=((32, 64),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        feature_maps = [
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
        ]
        anchors = generator(feature_maps, image_size=(64, 64))
        assert len(anchors) == 2
        # 16*16 grid * 2 sizes * 3 aspect ratios = 1536 anchors per feature map
        assert anchors[0].shape == (1536, 4)
        assert anchors[1].shape == (1536, 4)
