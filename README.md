# Med-Foundation-Models

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-132%20passed-brightgreen.svg)](docs/TESTCASES.md)
[![Docs](https://img.shields.io/badge/docs-complete-blue.svg)](docs/)

**Foundation Models for Whole-Body MRI: Self-Supervised Learning, Multimodal Fusion & Clinical Translation**

[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](docs/ARCHITECTURE.md) •
[Tests](docs/TESTCASES.md) •
[Contributing](docs/CONTRIBUTING.md) •
[License](#license)

</div>

---

## About

Med-Foundation-Models is a research-grade framework for building **foundation models on whole-body MRI**, combining self-supervised representation learning with multimodal clinical data integration. The goal: production-ready models that advance proactive healthcare through early detection, biomarker extraction, and clinical decision support.

### Key Features

- **Self-Supervised Pre-training**: MAE, SimCLR, DINO, JEPA
- **Vision Transformers**: ViT-B/16, ViT-L/16, ViT-H/14
- **Multimodal Fusion**: Imaging + Clinical + Radiology + Lab data
- **Downstream Tasks**: Segmentation, Detection, Classification, Biomarker Extraction
- **Production Ready**: Distributed training, checkpointing, logging

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 32GB+ VRAM for ViT-B/16 pre-training (64GB+ recommended for ViT-L/16)

### From Source

```bash
git clone https://github.com/MrKyaw/Med-Foundation-Models.git
cd Med-Foundation-Models
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t med-fm:latest .
docker run --gpus all -it --rm med-fm:latest
```

---

## Quick Start

### 1. Self-Supervised Pre-training (MAE-style)

```python
from medfm import MaskedAutoencoderViT

model = MaskedAutoencoderViT(
    img_size=518,
    patch_size=16,
    in_channels=1,  # MRI: single channel
    embed_dim=768,
    depth=12,
    num_heads=12,
    mask_ratio=0.75,
)
```

### 2. Multi-modal Fusion

```python
from medfm import MultimodalMRI

model = MultimodalMRI(
    backbone="vit_base_patch16",
    clinical_dim=256,
    radiology_dim=768,
    fusion_type="cross_attention",
)
```

### 3. Fine-tuning for Downstream Tasks

```python
from medfm import build_model

# Segmentation
model = build_model(task="segmentation", backbone="vit_base_patch16", num_classes=30)

# Classification
model = build_model(task="classification", backbone="vit_base_patch16", num_classes=10)

# Biomarker Extraction
model = build_model(task="biomarker", backbone="vit_base_patch16", biomarker_dims=50)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, workflow diagrams, job alignment |
| [TESTCASES.md](docs/TESTCASES.md) | Detailed test cases (132 tests, 100% pass rate) |
| [TEST_RESULTS.md](docs/TEST_RESULTS.md) | Test execution results and metrics |
| [FUTURE_WORK.md](docs/FUTURE_WORK.md) | Roadmap and future development plans |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Contribution guidelines |

---

## Test Results

**All 132 tests pass with 100% success rate.**

| Category | Tests | Passed | Failed |
|----------|-------|--------|--------|
| Data | 23 | 23 | 0 |
| Eval Metrics | 25 | 25 | 0 |
| Models | 66 | 66 | 0 |
| Training | 13 | 13 | 0 |
| Utilities | 5 | 5 | 0 |
| **Total** | **132** | **132** | **0** |

See [docs/TESTCASES.md](docs/TESTCASES.md) for detailed test cases.

---

## Project Structure

```
Med-Foundation-Models/
├── medfm/                      # Main package
│   ├── models/                 # Model implementations
│   │   ├── vit.py              # Vision Transformer
│   │   ├── mae.py              # Masked Autoencoder
│   │   ├── jepa.py             # JEPA
│   │   ├── contrastive.py      # SimCLR, DINO
│   │   ├── multimodal.py       # Multimodal fusion
│   │   ├── segmentation.py     # Segmentation heads
│   │   ├── detection.py        # Detection heads
│   │   ├── classification.py   # Classification heads
│   │   └── biomarker.py        # Biomarker heads
│   ├── data/                   # Data pipelines
│   │   ├── mri_dataset.py      # MRI dataset
│   │   ├── clinical_dataset.py # Clinical dataset
│   │   └── transforms.py       # Transforms
│   ├── train/                  # Training utilities
│   │   ├── pretrain.py         # Pre-training
│   │   ├── finetune.py         # Fine-tuning
│   │   ├── distributed.py      # Distributed training
│   │   └── optimizers.py       # Optimizers
│   ├── eval/                   # Evaluation metrics
│   │   ├── segmentation.py     # Segmentation metrics
│   │   ├── classification.py   # Classification metrics
│   │   ├── biomarker.py        # Biomarker metrics
│   │   └── clinical.py         # Clinical metrics
│   └── utils/                  # Utilities
│       ├── checkpointing.py    # Model I/O
│       ├── logging.py          # Experiment tracking
│       └── distributed.py      # Distributed helpers
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # Architecture & workflow
│   ├── TESTCASES.md            # Test cases
│   ├── TEST_RESULTS.md         # Test results
│   ├── FUTURE_WORK.md          # Roadmap
│   └── CONTRIBUTING.md         # Contribution guidelines
├── tests/                      # Test suite
│   ├── conftest.py             # Shared fixtures
│   ├── models/                 # Model tests
│   ├── data/                   # Data tests
│   ├── train/                  # Training tests
│   ├── eval/                   # Evaluation tests
│   └── utils/                  # Utility tests
├── configs/                    # Configuration files
├── scripts/                    # Execution scripts
├── examples/                   # Example notebooks
├── requirements.txt            # Dependencies
├── pyproject.toml              # Package metadata
├── Dockerfile                  # Docker configuration
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/MrKyaw/Med-Foundation-Models.git
cd Med-Foundation-Models
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [timm](https://github.com/rwightman/pytorch-image-models) for vision transformer implementations
- [MAE](https://github.com/facebookresearch/mae) for masked autoencoder reference
- [I-JEPA](https://github.com/facebookresearch/ijepa) for JEPA reference
- [MONAI](https://monai.io/) for medical imaging utilities

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](../../issues) · [Request Feature](../../issues) · [Ask Question](../../discussions)

</div>
