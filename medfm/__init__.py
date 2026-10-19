"""
Med-Foundation-Models: Foundation Models for Whole-Body MRI.

A research-grade framework for building foundation models on whole-body MRI,
combining self-supervised representation learning with multimodal clinical data integration.
"""

__version__ = "0.1.0"
__author__ = "MrKyaw"
__email__ = "mr.kyaw@example.com"

from .models import (
    VisionTransformer,
    MaskedAutoencoderViT,
    JEPAViT,
    SimCLR,
    DINO,
    MultimodalMRI,
    SegmentationHead,
    DetectionHead,
    ClassificationHead,
    BiomarkerHead,
)
from .data import MRIDataset, ClinicalDataset, MRITransform
from .eval import (
    dice_score,
    compute_auc,
    compute_biomarker_metrics,
    compute_segmentation_metrics,
)

__all__ = [
    "VisionTransformer",
    "MaskedAutoencoderViT",
    "JEPAViT",
    "SimCLR",
    "DINO",
    "MultimodalMRI",
    "SegmentationHead",
    "DetectionHead",
    "ClassificationHead",
    "BiomarkerHead",
    "MRIDataset",
    "ClinicalDataset",
    "MRITransform",
    "dice_score",
    "compute_auc",
    "compute_biomarker_metrics",
    "compute_segmentation_metrics",
]
