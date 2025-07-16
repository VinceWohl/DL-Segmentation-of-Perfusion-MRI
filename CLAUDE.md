# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a deep learning segmentation performance project focused on brain perfusion territory segmentation using nnUNet v2. The project implements custom multi-label binary segmentation approaches for left and right perfusion territories.

## Key Components

### 1. nnUNet Framework
- **Main nnUNet**: `/nnUNet/` - Primary nnUNet v2 implementation
- **Master version**: `/nnUNet_master/` - Reference implementation
- **Multi-label version**: `/nnUNet_multi-label/` - Custom implementations for dual-channel binary segmentation

### 2. Custom Trainer Implementations
The project includes custom nnUNet trainers for multi-label segmentation:
- `nnUNetTrainer_SeparateDecoders`: Shared encoder with separate decoders for each hemisphere
- `nnUNetTrainer_SeparateDecoders_CrossAttention`: Adds cross-attention mechanism
- `nnUNetTrainer_SeparateDecoders_SpatialLoss`: Incorporates spatial loss functions
- `nnUNetTrainer_SharedDecoder`: Shared decoder with multi-label head

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
export nnUNet_raw="/path/to/DLSegPerf/data/nnUNet_raw"
export nnUNet_preprocessed="/path/to/DLSegPerf/data/nnUNet_preprocessed"
export nnUNet_results="/path/to/DLSegPerf/data/nnUNet_results"
```

### Data Preprocessing
```bash
# Verify dataset integrity
python -m nnunetv2.experiment_planning.verify_dataset_integrity -d Dataset001_PerfusionTerritories

# Plan and preprocess
nnUNetv2_plan_and_preprocess -d 001
```

### Training
```bash
# Train with custom trainer (separate decoders)
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SeparateDecoders

# Train with shared decoder
nnUNetv2_train Dataset001_PerfusionTerritories 2d 0 -tr nnUNetTrainer_SharedDecoder

# Cross-validation training (all folds)
for FOLD in 0 1 2 3 4; do
    nnUNetv2_train Dataset001_PerfusionTerritories 2d $FOLD -tr nnUNetTrainer_SeparateDecoders
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
- Volume-based Dice computation
- Custom validation with detailed per-channel metrics

### Loss Function Implementation
```python
# Balanced per-channel BCE + combined Dice loss
bce_loss = 0.5 * BCE_loss(pred[:, 0], target[:, 0]) + 0.5 * BCE_loss(pred[:, 1], target[:, 1])
dice_loss = Dice_loss(pred, target)
total_loss = bce_loss + dice_loss
```

## File Locations

### Custom Trainer Files
- Separate decoders: `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SeparateDecoders.py`
- Shared decoder: `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SharedDecoder.py`
- Variants with additional features in the same directory

### Configuration Files
- Dataset configuration: `data/nnUNet_raw/Dataset001_PerfusionTerritories/dataset.json`
- Cross-validation splits: `data/nnUNet_raw/Dataset001_PerfusionTerritories/splits_final.json`

### Results and Validation
- Training results: `data/nnUNet_results/Dataset001_PerfusionTerritories/`
- Validation summaries: `validation_summary.json` files in each fold directory
- Evaluation outputs: `evaluation/evaluation_results/`

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