# Med-Foundation-Models Architecture & Workflow

## System Overview

Med-Foundation-Models implements an end-to-end foundation model pipeline for whole-body MRI, from raw data ingestion to clinically validated AI models. The system follows a **pre-train → fine-tune → deploy** paradigm with multimodal fusion capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        MED-FOUNDATION-MODELS ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Raw Data   │───▶│  Data       │───▶│  Self-Sup.  │───▶│ Foundation  │          │
│  │  Ingestion  │    │  Processing │    │  Pre-train  │    │ Model       │          │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘          │
│                                                                    │                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │                 │
│  │  Clinical   │    │  Downstream │    │  Clinical   │           │                 │
│  │  Validation │◀───│  Tasks      │◀───│  Fusion     │◀──────────┘                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Architecture

### Layer 1: Data Ingestion & Processing

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Whole-Body  │  │  Clinical    │  │  Radiology   │          │
│  │  MRI Scans   │  │  Data        │  │  Reports     │          │
│  │  (.nii, .dcm)│  │  (.csv)      │  │  (.txt)      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────┐           │
│  │              Data Transform Pipeline            │           │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │           │
│  │  │ Resize  │  │ Normalize│ │ Augment │         │           │
│  │  │ 518px   │  │ [0,1]   │ │ Flip/   │         │           │
│  │  └─────────┘  └─────────┘  │ Rotate  │         │           │
│  │                            └─────────┘         │           │
│  └────────────────────┬───────────────────────────┘           │
│                       │                                        │
│                       ▼                                        │
│              ┌────────────────┐                                │
│              │  DataLoader    │                                │
│              │  (Distributed) │                                │
│              └────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
- `medfm/data/mri_dataset.py` — MRIDataset class with NIfTI/DICOM/HDF5 support
- `medfm/data/transforms.py` — MRI-specific augmentations
- `medfm/data/clinical_dataset.py` — Structured clinical data loader

---

### Layer 2: Self-Supervised Pre-training

```
┌─────────────────────────────────────────────────────────────────┐
│              SELF-SUPERVISED PRE-TRAINING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Input: Unlabeled Whole-Body MRI (B, 1, H, W)        │      │
│  └──────────────────────────┬───────────────────────────┘      │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │     MAE     │    │   SimCLR    │    │    JEPA     │        │
│  │  (MIM)      │    │ (Contrast.) │    │ (Predict.)  │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌─────────────────────────────────────────────────────┐      │
│  │         Shared Vision Transformer Backbone          │      │
│  │  ┌─────┐ ┌─────┐ ┌─────┐        ┌─────┐            │      │
│  │  │Patch│→│Embed│→│Trans│→ ... → │Norm │            │      │
│  │  │Embed│ │Pos  │ │Block│        │Layer│            │      │
│  │  └─────┘ └─────┘ └─────┘        └─────┘            │      │
│  └─────────────────────────────────────────────────────┘      │
│                             │                                   │
│                             ▼                                   │
│              ┌────────────────────────────┐                    │
│              │  Pre-trained Foundation   │                    │
│              │  Model Weights            │                    │
│              │  (Checkpoints)            │                    │
│              └────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

**Pre-training Strategies:**

| Method | Mechanism | Loss | Use Case |
|--------|-----------|------|----------|
| **MAE** | Mask 75% patches, reconstruct | MSE on masked | General representation |
| **SimCLR** | Two augmented views, contrast | NT-Xent | Discriminative features |
| **DINO** | Self-distillation, no labels | Cross-entropy | Robust features |
| **JEPA** | Predict target from context | MSE in latent | Spatial reasoning |

---

### Layer 3: Multimodal Fusion

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │   MRI      │  │  Radiology │  │    Lab     │  │ Clinical │ │
│  │  Imaging   │  │  Reports   │  │   Values   │  │  Data    │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘ │
│        │               │               │              │        │
│        ▼               ▼               ▼              ▼        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  ViT     │   │  Text    │   │  MLP     │   │  MLP     │    │
│  │ Backbone │   │ Encoder  │   │ Encoder  │   │ Encoder  │    │
│  │ (768-d)  │   │ (768-d)  │   │ (256-d)  │   │ (256-d)  │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │              │              │              │           │
│       └──────────────┴──────────────┴──────────────┘           │
│                          │                                      │
│                          ▼                                      │
│              ┌───────────────────────┐                         │
│              │   Fusion Module       │                         │
│              │  (Cross-Attention /   │                         │
│              │   Gated / Concat)     │                         │
│              └───────────┬───────────┘                         │
│                          │                                      │
│                          ▼                                      │
│              ┌───────────────────────┐                         │
│              │   Joint Embedding     │                         │
│              │   (Clinical Output)   │                         │
│              └───────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

**Fusion Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Cross-Attention** | Imaging tokens attend to clinical tokens | Complex interactions |
| **Gated Fusion** | Learnable gating mechanism | Modality imbalance |
| **Concatenation** | Simple feature concatenation | Fast inference |

---

### Layer 4: Downstream Task Heads

```
┌─────────────────────────────────────────────────────────────────┐
│                   DOWNSTREAM TASK HEADS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              ┌─────────────────────────┐                       │
│              │  Pre-trained Backbone   │                       │
│              │  (ViT-B/16 or ViT-L/16) │                       │
│              └────────────┬────────────┘                       │
│                           │                                    │
│         ┌─────────────────┼─────────────────┐                 │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │ Segmentation│   │  Detection  │   │Classification│         │
│  │    Head     │   │    Head     │   │    Head      │         │
│  │  (U-Net)    │   │   (FPN)     │   │   (MLP)      │         │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘          │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │ Multi-organ │   │   Lesion    │   │   Disease    │          │
│  │ Segmentation│   │  Detection  │   │ Classification│         │
│  │  (30 classes)│   │  (5 classes)│   │  (10 classes)│         │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│         ┌─────────────────┐                                    │
│         ▼                 ▼                                    │
│  ┌─────────────┐   ┌─────────────┐                            │
│  │  Biomarker  │   │ Longitudinal│                            │
│  │ Extraction  │   │  Modeling   │                            │
│  │   Head      │   │    Head     │                            │
│  └──────┬──────┘   └──────┬──────┘                            │
│         │                 │                                    │
│         ▼                 ▼                                    │
│  ┌─────────────┐   ┌─────────────┐                            │
│  │ Quantitative│   │  Temporal   │                            │
│  │ Biomarkers  │   │ Progression │                            │
│  │  (50 dims)  │   │  Tracking   │                            │
│  └─────────────┘   └─────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

### Layer 5: Clinical Validation & Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                 CLINICAL VALIDATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Trained Model                                      │       │
│  └──────────────────────────┬──────────────────────────┘       │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Performance │    │   Clinical  │    │  Regulatory │        │
│  │ Benchmarks  │    │  Validation │    │  Compliance │        │
│  │  (DSC/AUC)  │    │(Sensitivity │    │  (FDA 510k) │        │
│  │             │    │/Specificity)│    │             │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         └──────────────────┼──────────────────┘                │
│                            │                                   │
│                            ▼                                   │
│              ┌────────────────────────────┐                   │
│              │   Production Deployment   │                   │
│              │  (Clinical Workflow)       │                   │
│              └────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           END-TO-END WORKFLOW                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  PHASE 1: DATA PREPARATION (Weeks 1-2)                                             │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │  • Ingest whole-body MRI scans (NIfTI/DICOM)                     │               │
│  │  • Collect clinical data (labs, demographics, reports)           │               │
│  │  • Apply preprocessing (normalization, resampling)               │               │
│  │  • Create train/val/test splits                                 │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│                                        │                                            │
│                                        ▼                                            │
│  PHASE 2: SELF-SUPERVISED PRE-TRAINING (Weeks 3-6)                                 │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │  • Train MAE on unlabeled MRI data                              │               │
│  │  • Fine-tune with contrastive learning (SimCLR/DINO)             │               │
│  │  • Optional: JEPA for latent space prediction                   │               │
│  │  • Save foundation model checkpoints                            │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│                                        │                                            │
│                                        ▼                                            │
│  PHASE 3: MULTIMODAL FUSION TRAINING (Weeks 7-8)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │  • Integrate imaging + clinical + radiology data                │               │
│  │  • Train fusion module (cross-attention)                        │               │
│  │  • Validate on held-out clinical cohort                         │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│                                        │                                            │
│                                        ▼                                            │
│  PHASE 4: DOWNSTREAM TASK FINE-TUNING (Weeks 9-12)                                 │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │  • Segmentation: Multi-organ, lesion, body composition          │               │
│  │  • Detection: Lesion detection with FPN                         │               │
│  │  • Classification: Disease classification (10 classes)           │               │
│  │  • Biomarker: Quantitative extraction (50 biomarkers)            │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│                                        │                                            │
│                                        ▼                                            │
│  PHASE 5: CLINICAL VALIDATION (Weeks 13-14)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │  • Performance benchmarks (DSC, AUC, sensitivity, specificity)   │               │
│  │  • Clinical review with radiologists                            │               │
│  │  • Calibration analysis (ECE, MCE)                              │               │
│  │  • Bias detection across demographics                           │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│                                        │                                            │
│                                        ▼                                            │
│  PHASE 6: PRODUCTION DEPLOYMENT (Weeks 15-16)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐               │
│  │  • Export to ONNX/TorchScript                                   │               │
│  │  • Integrate with clinical workflow                             │               │
│  │  • Deploy to inference cluster                                  │               │
│  │  • Set up monitoring and alerting                               │               │
│  └─────────────────────────────────────────────────────────────────┘               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Job Alignment Matrix

| Job Requirement | Implementation | Component |
|-----------------|----------------|-----------|
| Foundation models for whole-body MRI | ViT-B/16, ViT-L/16, ViT-H/14 backbones | `models/vit.py` |
| Self-supervised learning (MIM, contrastive, JEPA) | MAE, SimCLR, DINO, JEPA implementations | `models/mae.py`, `models/contrastive.py`, `models/jepa.py` |
| Multimodal fusion (imaging + clinical + radiology + lab) | Cross-attention, gated, concat fusion | `models/multimodal.py` |
| Medical image analysis | Segmentation, detection, classification heads | `models/segmentation.py`, `models/detection.py`, `models/classification.py` |
| Biomarker extraction | Multi-task biomarker head | `models/biomarker.py` |
| Longitudinal modeling | Temporal biomarker tracking | `models/biomarker.py` (LongitudinalBiomarkerHead) |
| Generative/agentic AI | LLM adaptation interface | `models/multimodal.py` (RadiologyReportDataset) |
| Production-ready models | Distributed training, checkpointing, ONNX export | `train/distributed.py`, `utils/checkpointing.py` |
| Clinical validation | Sensitivity, specificity, calibration metrics | `eval/clinical.py` |
| Experiment design | Config-driven training, hyperparameter management | `train/pretrain.py`, `train/finetune.py` |

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Deep Learning Framework | PyTorch 2.1+ |
| Vision Transformers | timm 0.9.12+ |
| Medical Imaging | MONAI, nibabel, SimpleITK |
| Experiment Tracking | TensorBoard, Weights & Biases |
| Distributed Training | PyTorch DDP, Accelerate |
| Data Processing | NumPy, Pandas, h5py |
| Evaluation | scikit-learn, scipy |
| Deployment | ONNX, TorchScript |

---

## Scaling Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      SCALING STRATEGY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Single GPU ──▶ Multi-GPU ──▶ Multi-Node ──▶ Cloud             │
│  (32GB)         (8x A100)     (64x A100)    (Auto-scaling)    │
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ 1 GPU   │───▶│ 8 GPUs  │───▶│ 64 GPUs │───▶│ Cloud   │     │
│  │ 256 bs  │    │ 2048 bs │    │ 16K bs  │    │ Auto    │     │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                                                                 │
│  • Gradient accumulation for large batch sizes                  │
│  • Mixed precision (FP16/BF16) training                         │
│  • Gradient checkpointing for memory efficiency                 │
│  • Data parallelism with DDP                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quality Assurance

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUALITY ASSURANCE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Unit      │    │  Integration│    │   Clinical  │         │
│  │   Tests     │    │    Tests    │    │   Review    │         │
│  │  (132)      │    │   (E2E)     │    │  (Sign-off) │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  Metrics:                                                       │
│  • Segmentation: DSC, HD95, Surface Dice                        │
│  • Classification: AUC-ROC, Sensitivity, Specificity            │
│  • Biomarker: MAE, R², Pearson correlation                      │
│  • Clinical: Net Benefit, Calibration Error                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reproducibility

```bash
# Clone repository
git clone https://github.com/MrKyaw/Med-Foundation-Models.git
cd Med-Foundation-Models

# Install dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Pre-training example
python scripts/pretrain.sh --config configs/pretrain/mae_vit_base.yaml

# Fine-tuning example
python scripts/finetune.sh --config configs/finetune/segmentation.yaml

# Evaluation
python scripts/evaluate.sh --model checkpoints/best_model.pth
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.
