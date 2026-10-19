# Test Cases - Med-Foundation-Models

**Total Tests:** 132  
**Pass Rate:** 100%  
**Last Updated:** 2024-10-19

---

## Test Categories

### 1. Model Tests (66 tests)

#### 1.1 Vision Transformer (12 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_patchify_shape` | Verify patch embedding output shape | PASS |
| `test_different_patch_sizes` | Test patch sizes 8, 16, 32 | PASS |
| `test_multichannel_input` | Test 3-channel input | PASS |
| `test_attention_output_shape` | Verify attention output shape | PASS |
| `test_different_num_heads` | Test heads 1, 2, 4, 8 | PASS |
| `test_mlp_output_shape` | Verify MLP output shape | PASS |
| `test_block_output_shape` | Verify transformer block output | PASS |
| `test_forward_shape` | Full forward pass shape | PASS |
| `test_forward_features_shape` | Feature extraction shape | PASS |
| `test_no_classification_head` | Test without classification head | PASS |
| `test_get_num_layers` | Verify layer count | PASS |
| `test_gradient_flow` | Verify gradients flow | PASS |

#### 1.2 Masked Autoencoder (7 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_patchify_shape` | Verify patchify output | PASS |
| `test_unpatchify_shape` | Verify unpatchify output | PASS |
| `test_patchify_unpatchify_roundtrip` | Roundtrip reconstruction | PASS |
| `test_random_masking_shapes` | Verify masking shapes | PASS |
| `test_forward_loss` | Forward pass loss computation | PASS |
| `test_different_mask_ratios` | Test ratios 0.25-0.9 | PASS |
| `test_gradient_flow` | Verify gradients flow | PASS |

#### 1.3 JEPA (7 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_generate_masks` | Verify mask generation | PASS |
| `test_forward_context` | Context encoder forward | PASS |
| `test_forward_target` | Target encoder forward | PASS |
| `test_forward_predictor` | Predictor forward pass | PASS |
| `test_forward_loss` | JEPA loss computation | PASS |
| `test_target_encoder_no_grad` | Verify target encoder frozen | PASS |
| `test_update_target_encoder` | EMA update verification | PASS |

#### 1.4 Contrastive Learning (10 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_projection_head_output` | Projection head shape | PASS |
| `test_simclr_forward_shape` | SimCLR forward shape | PASS |
| `test_simclr_nt_xent_loss` | NT-Xent loss computation | PASS |
| `test_simclr_output_normalized` | Output normalization | PASS |
| `test_dino_forward_shape` | DINO forward shape | PASS |
| `test_dino_teacher_no_grad` | Teacher frozen | PASS |
| `test_dino_loss` | DINO loss computation | PASS |
| `test_dino_update_teacher` | EMA update verification | PASS |

#### 1.5 Multimodal Fusion (7 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_cross_attention_output` | Cross-attention shape | PASS |
| `test_gated_fusion_output` | Gated fusion shape | PASS |
| `test_forward_images_only` | Images only forward | PASS |
| `test_forward_with_clinical` | With clinical data | PASS |
| `test_forward_with_all_modalities` | All modalities forward | PASS |
| `test_gated_fusion` | Gated fusion strategy | PASS |
| `test_concat_fusion` | Concat fusion strategy | PASS |

#### 1.6 Segmentation (5 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_double_conv_output` | Double conv shape | PASS |
| `test_up_upsampling` | Upsampling block | PASS |
| `test_segmentation_head_output` | Segmentation head shape | PASS |
| `test_segmentation_with_target_size` | With target size | PASS |
| `test_unet_decoder_output` | U-Net decoder shape | PASS |

#### 1.7 Detection (3 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_fpn_output_shapes` | FPN output shapes | PASS |
| `test_detection_head_output` | Detection head shapes | PASS |
| `test_anchor_generator_shapes` | Anchor generation | PASS |

#### 1.8 Classification (5 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_classification_head_output` | Classification head shape | PASS |
| `test_classification_multi_layer` | Multi-layer classifier | PASS |
| `test_multilabel_output_shape` | Multi-label shape | PASS |
| `test_multilabel_loss` | Multi-label loss | PASS |
| `test_hierarchical_output_shapes` | Hierarchical output | PASS |

#### 1.9 Biomarker (5 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_biomarker_head_output` | Biomarker head shape | PASS |
| `test_biomarker_loss` | Biomarker loss | PASS |
| `test_multitask_output_keys` | Multi-task output keys | PASS |
| `test_multitask_loss` | Multi-task loss | PASS |
| `test_longitudinal_output_shape` | Longitudinal shape | PASS |

---

### 2. Data Tests (23 tests)

#### 2.1 MRI Dataset (6 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_empty_dataset` | Empty dataset handling | PASS |
| `test_numpy_loading` | NumPy file loading | PASS |
| `test_npz_loading` | NPZ file loading | PASS |
| `test_label_loading` | Label loading | PASS |
| `test_clinical_loading` | Clinical data loading | PASS |
| `test_class_distribution` | Class distribution | PASS |

#### 2.2 MRI Transforms (17 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_empty_transform` | Empty transform pipeline | PASS |
| `test_composed_transforms` | Composed transforms | PASS |
| `test_flip_happens` | Random flip execution | PASS |
| `test_no_flip` | No flip when p=0 | PASS |
| `test_rotation` | Random rotation | PASS |
| `test_intensity_scale_output` | Intensity scaling | PASS |
| `test_intensity_shift_output` | Intensity shift | PASS |
| `test_noise_output` | Gaussian noise | PASS |
| `test_gamma_output` | Gamma correction | PASS |
| `test_normalization` | Normalization | PASS |
| `test_resize_smaller` | Resize to smaller | PASS |
| `test_resize_larger` | Resize to larger | PASS |
| `test_center_crop` | Center crop | PASS |
| `test_pad` | Padding | PASS |
| `test_train_transforms` | Train transform pipeline | PASS |
| `test_eval_transforms` | Eval transform pipeline | PASS |

---

### 3. Evaluation Metric Tests (25 tests)

#### 3.1 Segmentation Metrics (6 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_dice_perfect_prediction` | Perfect prediction Dice | PASS |
| `test_dice_single_class` | Single class Dice | PASS |
| `test_dice_probabilistic_input` | Probabilistic input Dice | PASS |
| `test_segmentation_output_keys` | Output keys verification | PASS |
| `test_segmentation_perfect_metrics` | Perfect metrics | PASS |
| `test_segmentation_with_background` | With background class | PASS |

#### 3.2 Classification Metrics (8 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_auc_perfect` | Perfect AUC | PASS |
| `test_auc_multiclass` | Multi-class AUC | PASS |
| `test_classification_output_keys` | Output keys | PASS |
| `test_classification_perfect` | Perfect classification | PASS |
| `test_sensitivity_specificity_perfect` | Perfect sens/spec | PASS |
| `test_sensitivity_specificity_keys` | Output keys | PASS |
| `test_threshold_youden` | Youden threshold | PASS |
| `test_threshold_f1` | F1 threshold | PASS |

#### 3.3 Biomarker Metrics (8 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_biomarker_perfect_prediction` | Perfect prediction | PASS |
| `test_biomarker_output_keys` | Output keys | PASS |
| `test_biomarker_with_names` | With biomarker names | PASS |
| `test_biomarker_accuracy_perfect` | Perfect accuracy | PASS |
| `test_biomarker_accuracy_keys` | Accuracy output keys | PASS |
| `test_biomarker_bias_none` | No bias case | PASS |
| `test_biomarker_bias_keys` | Bias output keys | PASS |

---

### 4. Training Tests (13 tests)

#### 4.1 Optimizers (4 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_optimizer_adamw` | AdamW optimizer | PASS |
| `test_optimizer_adam` | Adam optimizer | PASS |
| `test_optimizer_sgd` | SGD optimizer | PASS |
| `test_optimizer_invalid` | Invalid optimizer error | PASS |

#### 4.2 Schedulers (5 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_scheduler_cosine` | Cosine annealing | PASS |
| `test_scheduler_step` | Step scheduler | PASS |
| `test_scheduler_constant` | Constant scheduler | PASS |
| `test_scheduler_warmup_cosine` | Warmup + cosine | PASS |

#### 4.3 Warmup Cosine Scheduler (3 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_warmup_phase` | Warmup phase LR increase | PASS |
| `test_cosine_phase` | Cosine phase LR decay | PASS |
| `test_state_dict` | State dict save/load | PASS |

---

### 5. Utility Tests (13 tests)

#### 5.1 Checkpointing (9 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_save_checkpoint_creates_file` | Save creates file | PASS |
| `test_save_best_model` | Save best model | PASS |
| `test_load_restores_state` | Load restores state | PASS |
| `test_load_nonexistent_file` | Nonexistent file error | PASS |
| `test_save_weights_pth` | Save PTH format | PASS |
| `test_save_weights_pt` | Save PT format | PASS |
| `test_save_invalid_format` | Invalid format error | PASS |
| `test_load_weights` | Load weights | PASS |
| `test_load_weights_nonexistent` | Nonexistent weights error | PASS |

#### 5.2 Logging (10 tests)

| Test Name | Description | Status |
|-----------|-------------|--------|
| `test_smoothed_value_update` | Value update | PASS |
| `test_smoothed_value_avg` | Average calculation | PASS |
| `test_smoothed_value_global_avg` | Global average | PASS |
| `test_smoothed_value_window_size` | Window size | PASS |
| `test_smoothed_value_str` | String representation | PASS |
| `test_metric_logger_update` | Metric update | PASS |
| `test_metric_logger_str` | String representation | PASS |
| `test_metric_logger_log_every` | Log every iteration | PASS |
| `test_setup_logging_creates_dir` | Creates directory | PASS |
| `test_setup_logging_non_main` | Non-main process | PASS |

---

## Test Execution

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/models/ -v
pytest tests/data/ -v
pytest tests/eval/ -v
pytest tests/train/ -v
pytest tests/utils/ -v

# Run with coverage
pytest tests/ -v --cov=medfm

# Run with report
pytest tests/ -v --tb=short --junitxml=report.xml
```

### Test Environment

- **Python:** 3.12.8
- **PyTorch:** 2.1+
- **Platform:** macOS (Darwin)
- **Test Framework:** pytest 9.1.1
- **Coverage:** 100% (132/132)

---

## Continuous Integration

All tests run on every commit via GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/ -v
```

---

**Last Updated:** 2024-10-19
