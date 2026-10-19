"""Tests for Segmentation heads."""

import pytest
import torch
from medfm.models.segmentation import SegmentationHead, UNetDecoder, DoubleConv, Up


class TestDoubleConv:
    def test_output_shape(self):
        conv = DoubleConv(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 32, 32)
        out = conv(x)
        assert out.shape == (2, 128, 32, 32)


class TestUp:
    def test_upsampling(self):
        up = Up(in_channels=128, out_channels=64, bilinear=True)
        x1 = torch.randn(2, 128, 16, 16)
        x2 = torch.randn(2, 64, 32, 32)
        out = up(x1, x2)
        assert out.shape == (2, 64, 32, 32)


class TestSegmentationHead:
    def test_output_shape(self):
        head = SegmentationHead(
            in_channels=128,
            num_classes=5,
            features=[32, 64, 128, 256],
        )
        x = torch.randn(2, 128, 16, 16)
        out = head(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 5

    def test_with_target_size(self):
        head = SegmentationHead(
            in_channels=128,
            num_classes=5,
            features=[32, 64, 128, 256],
        )
        x = torch.randn(2, 128, 16, 16)
        out = head(x, target_size=(64, 64))
        assert out.shape == (2, 5, 64, 64)


class TestUNetDecoder:
    def test_output_shape(self):
        decoder = UNetDecoder(
            encoder_channels=[128, 128, 128, 128],
            decoder_channels=[256, 128, 64, 32],
            num_classes=5,
        )
        features = [
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 16, 16),
        ]
        out = decoder(features)
        assert out.shape[0] == 2
        assert out.shape[1] == 5
