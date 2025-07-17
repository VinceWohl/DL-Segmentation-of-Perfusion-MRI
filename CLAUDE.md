# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a deep learning segmentation performance project focused on brain perfusion territory segmentation using nnUNet v2. The project implements a **comprehensive 9-trainer experimental setup** for dual-channel binary segmentation approaches evaluating architecture variants and loss function enhancements.

## Current Project Status (Updated: 2025-07-17)

### Recent Major Development
- **Completed comprehensive 9-trainer experimental setup** for systematic comparison
- **All trainers configured with 1000 epochs** and ready for training
- **Sequential training script prepared** for fold_0 evaluation
- **Complete documentation updated** in README_MultiLabel_Setup.md

### Ready for Execution
The project is now ready for comprehensive training and evaluation with all 9 trainers:
1. nnUNetTrainer_SeparateDecoders (baseline)
2. nnUNetTrainer_SeparateDecoders_SpatialLoss (enhanced)
3. nnUNetTrainer_SeparateDecoders_ComplementaryLoss (complementary)
4. nnUNetTrainer_SeparateDecoders_SpatialLoss_ComplementaryLoss (full enhancement)
5. nnUNetTrainer_SeparateDecoders_CrossAttention (cross-attention baseline)
6. nnUNetTrainer_SharedDecoder (baseline)
7. nnUNetTrainer_SharedDecoder_SpatialLoss (spatial enhancement)
8. nnUNetTrainer_SharedDecoder_ComplementaryLoss (complementary enhancement)
9. nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss (full enhancement)

## Key Components

### 1. nnUNet Framework
- **Main nnUNet**: `/nnUNet/` - Primary nnUNet v2 implementation with 9 custom trainers
- **Master version**: `/nnUNet_master/` - Reference implementation
- **Multi-label version**: `/nnUNet_multi-label/` - Custom implementations for dual-channel binary segmentation

### 2. Comprehensive 9-Trainer Experimental Setup

#### Architecture Variants
- **Separate Decoders**: Independent learning pathways for each hemisphere
- **Shared Decoder**: Shared spatial learning with multi-label head
- **Cross-Attention**: Inter-hemisphere information sharing through 8-head attention

#### Loss Function Enhancements
- **Baseline**: BCE + Dice loss only
- **Spatial**: +Spatial consistency loss (weight: 0.1)
- **Complementary**: +Complementary loss for mutual exclusivity (weight: 0.1)
- **Combined**: +Both spatial and complementary losses

#### Complete Trainer Matrix
```
Architecture          | Baseline | Spatial | Complementary | Combined
---------------------|----------|---------|---------------|----------
Separate Decoders    |    ✓     |    ✓    |      ✓        |    ✓
Shared Decoder       |    ✓     |    ✓    |      ✓        |    ✓
Cross-Attention      |    ✓     |    -    |      -        |    -
```

### 3. Data Structure
- **Raw data**: `/data/nnUNet_raw/` - Input images and labels
- **Preprocessed**: `/data/nnUNet_preprocessed/` - Processed training data
- **Results**: `/data/nnUNet_results/` - Training outputs and model checkpoints
- **Test predictions**: `/data/test_predictions/` - Model inference results

### 4. Data Preparation Pipeline
Located in `/data_preparation/`:
- **Data anonymization**: SPM-based defacing and subject renumeration
- **Data conversion**: MATLAB scripts for nnUNet format conversion
- **Data cropping**: ROI extraction scripts

### 5. Evaluation Scripts
Located in `/evaluation/`:
- `evaluate_model.py`: Model evaluation with Dice score analysis and violin plots
- `evaluate_slice_wise_results.py`: Slice-wise performance analysis

## Common Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install nnUNet in development mode
cd nnUNet
pip install -e .
```

### Environment Variables
Set these before running nnUNet commands:
```bash
export nnUNet_raw="/home/ubuntu/DLSegPerf/data/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/DLSegPerf/data/nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/DLSegPerf/data/nnUNet_results"
```

### Data Preprocessing
```bash
# Verify dataset integrity
python -m nnunetv2.experiment_planning.verify_dataset_integrity -d Dataset001_PerfusionTerritories

# Plan and preprocess
nnUNetv2_plan_and_preprocess -d 001
```

### Training - Comprehensive 9-Trainer Setup

#### Sequential Training Script (All Trainers)
```bash
#!/bin/bash
# Sequential training for comprehensive comparison

echo "Starting comprehensive training of all 9 trainers for fold_0..."

# Separate Decoders trainers
echo "Training SeparateDecoders baseline..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders

echo "Training SeparateDecoders with SpatialLoss..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders_SpatialLoss

echo "Training SeparateDecoders with ComplementaryLoss..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders_ComplementaryLoss

echo "Training SeparateDecoders with SpatialLoss + ComplementaryLoss..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders_SpatialLoss_ComplementaryLoss

echo "Training SeparateDecoders with CrossAttention..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders_CrossAttention

# Shared Decoder trainers
echo "Training SharedDecoder baseline..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SharedDecoder

echo "Training SharedDecoder with SpatialLoss..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SharedDecoder_SpatialLoss

echo "Training SharedDecoder with ComplementaryLoss..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SharedDecoder_ComplementaryLoss

echo "Training SharedDecoder with SpatialLoss + ComplementaryLoss..."
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss

echo "All trainers completed for fold_0!"
```

#### Individual Trainer Commands
```bash
# Baseline comparisons
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SharedDecoder

# Enhanced trainers
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders_SpatialLoss_ComplementaryLoss
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss

# Cross-attention
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders_CrossAttention
```

#### Cross-Validation Training
```bash
# Full cross-validation for best performing trainer
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train Dataset001_PerfusionTerritories 2d $FOLD -tr nnUNetTrainer_SeparateDecoders_SpatialLoss_ComplementaryLoss
done
```

### Inference
```bash
# Predict using trained model
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d Dataset001_PerfusionTerritories -c 2d -f 0 -tr nnUNetTrainer_SeparateDecoders
```

### Evaluation
```bash
# Run model evaluation
cd evaluation
python evaluate_model.py

# Generate slice-wise results
python evaluate_slice_wise_results.py
```

## Development Environment

### Python Environment
- Uses virtual environment at `/venv/`
- Based on PyTorch with medical imaging dependencies
- Key packages: torch, nibabel, SimpleITK, matplotlib, scipy

### No Standard Build System
- No traditional build/test/lint commands
- Uses setuptools with pyproject.toml for nnUNet installation
- Development dependencies include black, ruff, and pre-commit

## Architecture Notes

### Multi-Label Segmentation Approach
The project implements dual-channel binary segmentation where:
- Channel 0: Left perfusion territory (binary mask)
- Channel 1: Right perfusion territory (binary mask)
- Input: 2-4 channels (CBF LICA, CBF RICA, optional T1w, FLAIR)

### Key Technical Differences from Standard nnUNet
- Sigmoid activation instead of softmax for binary output
- Per-channel BCE loss with balanced weighting
- Independent hemisphere optimization
- Custom validation with detailed per-channel metrics

### Loss Function Implementation

#### Baseline Loss
```python
# Balanced per-channel BCE + combined Dice loss
bce_loss = 0.5 * BCE_loss(pred[:, 0], target[:, 0]) + 0.5 * BCE_loss(pred[:, 1], target[:, 1])
dice_loss = Dice_loss(pred, target)
total_loss = bce_loss + dice_loss
```

#### Enhanced Loss Functions
```python
# Spatial consistency loss (weight: 0.1)
spatial_loss = compute_spatial_consistency(pred_probs, target)

# Complementary loss (weight: 0.1)
complementary_loss = compute_complementary_constraints(pred_probs, target)

# Combined enhancement
total_loss = bce_loss + dice_loss + 0.1 * spatial_loss + 0.1 * complementary_loss
```

## File Locations

### Custom Trainer Files (All 9 Trainers)
**Separate Decoders Trainers:**
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SeparateDecoders.py`
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SeparateDecoders_SpatialLoss.py`
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SeparateDecoders_ComplementaryLoss.py`
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SeparateDecoders_SpatialLoss_ComplementaryLoss.py`
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SeparateDecoders_CrossAttention.py`

**Shared Decoder Trainers:**
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SharedDecoder.py`
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SharedDecoder_SpatialLoss.py`
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SharedDecoder_ComplementaryLoss.py`
- `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss.py`

### Configuration Files
- **Dataset configuration**: `data/nnUNet_raw/Dataset001_PerfusionTerritories/dataset.json`
- **Cross-validation splits**: `data/nnUNet_raw/Dataset001_PerfusionTerritories/splits_final.json`
- **Trainer comparison**: `nnUNet/nnunetv2/training/nnUNetTrainer/trainer_comparison.csv`

### Results and Validation
- **Training results**: `data/nnUNet_results/Dataset001_PerfusionTerritories/`
- **Validation summaries**: `validation_summary.json` files in each fold directory
- **Evaluation outputs**: `evaluation/evaluation_results/`

### Documentation
- **Main documentation**: `nnUNet_multi-label/README_MultiLabel_Setup.md`
- **Project guidance**: `CLAUDE.md` (this file)

## Data Format Specifications

### Input Images
- Multi-channel NIfTI format
- Naming convention: `CaseID_XXXX.nii` (where XXXX is 0000-0003 for channels)
- Supported: 2-4 input channels with automatic adaptation

### Labels
- 2-channel binary masks
- Shape: (slices, H, W, 2) for 2D processing
- Values: 0 (background) or 1 (perfusion territory)
- No multi-class overlap between channels

### Output Predictions
- 2-channel binary probability maps
- Sigmoid activation for independent hemisphere predictions
- Volume-based evaluation metrics

## Performance Expectations (Preliminary Results)

### Current Performance Ranking
1. **SeparateDecoders_SpatialLoss_ComplementaryLoss**: 90.61% mean Dice
2. **SharedDecoder (baseline)**: 90.49% mean Dice
3. **SeparateDecoders_SpatialLoss**: 90.26% mean Dice
4. **SeparateDecoders_CrossAttention**: 90.23% mean Dice
5. **SharedDecoder_SpatialLoss_ComplementaryLoss**: 90.09% mean Dice
6. **SeparateDecoders (baseline)**: 89.94% mean Dice
7. **SharedDecoder_ComplementaryLoss**: 89.93% mean Dice

### Key Insights
- **Spatial Loss**: Provides consistent +0.3-0.7% improvement
- **Complementary Loss**: Most effective when combined with spatial loss
- **Architecture Impact**: Separate decoders show slight advantage in enhanced configurations
- **Cross-Attention**: Competitive baseline performance

## Next Steps

1. **Execute comprehensive training** of all 9 trainers for fold_0
2. **Analyze systematic performance comparison** across architectures and loss functions
3. **Extend best performers** to 5-fold cross-validation
4. **Perform statistical significance testing** on results
5. **Fine-tune hyperparameters** based on comprehensive results
6. **Clinical validation** against radiologist annotations

## Important Notes for Future Sessions

- **All trainers are ready for execution** with 1000 epochs
- **Sequential training script available** for systematic comparison
- **Comprehensive documentation completed** in README_MultiLabel_Setup.md
- **CSV comparison table available** at `nnUNet/nnunetv2/training/nnUNetTrainer/trainer_comparison.csv`
- **Project is at execution phase** - ready for comprehensive training and analysis