#!/usr/bin/env python3
"""
Slice-wise DSC Analysis for nnUNet Perfusion Territory Segmentation Results

This script creates box-whisker plots showing the Dice Similarity Coefficient (DSC)
for each slice across the z-dimension (14 slices per volume) for both left (LICA)
and right (RICA) hemisphere perfusion territories.

Author: Generated for DLSegPerf project
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_dice_coefficient(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Calculate Dice Similarity Coefficient between prediction and ground truth.
    
    Args:
        pred: Prediction binary mask
        gt: Ground truth binary mask  
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient as float
    """
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def calculate_slice_wise_dsc(pred_path: str, gt_path: str) -> List[float]:
    """
    Calculate slice-wise DSC for a single prediction-ground truth pair.
    
    Args:
        pred_path: Path to prediction NIfTI file
        gt_path: Path to ground truth NIfTI file
        
    Returns:
        List of DSC values for each slice
    """
    try:
        # Load prediction and ground truth
        pred_img = nib.load(pred_path)
        gt_img = nib.load(gt_path)
        
        pred_data = pred_img.get_fdata()
        gt_data = gt_img.get_fdata()
        
        # Ensure binary masks (threshold at 0.5 for predictions)
        pred_data = (pred_data > 0.5).astype(np.uint8)
        gt_data = (gt_data > 0.5).astype(np.uint8)
        
        # Calculate DSC for each slice (assuming z is last dimension)
        num_slices = pred_data.shape[-1]
        slice_dsc = []
        
        for slice_idx in range(num_slices):
            pred_slice = pred_data[:, :, slice_idx]
            gt_slice = gt_data[:, :, slice_idx]
            
            dsc = calculate_dice_coefficient(pred_slice, gt_slice)
            slice_dsc.append(dsc)
            
        return slice_dsc
        
    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        return []

def collect_all_predictions() -> Dict[str, List[Tuple[str, str]]]:
    """
    Collect all prediction-ground truth pairs from all result directories.
    
    Returns:
        Dictionary with hemisphere as key and list of (pred_path, gt_path) tuples as values
    """
    base_dir = "data/TrainingsResults-PerfTerr"
    gt_dir = f"{base_dir}/nnUNet_raw_250827-PerfTerr-25/Dataset001_PerfusionTerritories/labelsTr"
    
    # Result directories
    result_dirs = [
        "nnUNet_results_Single-class_CBF",
        "nnUNet_results_Single-class_CBF_FLAIR", 
        "nnUNet_results_Single-class_CBF_T1w",
        "nnUNet_results_Single-class_CBF_T1w_FLAIR"
    ]
    
    pairs = {"L": [], "R": []}
    
    for result_dir in result_dirs:
        result_path = f"{base_dir}/{result_dir}/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d"
        
        # Check all folds
        for fold in range(5):
            fold_path = f"{result_path}/fold_{fold}/validation"
            
            if not os.path.exists(fold_path):
                continue
                
            # Find all prediction files
            pred_files = glob.glob(f"{fold_path}/*.nii")
            
            for pred_file in pred_files:
                filename = os.path.basename(pred_file)
                
                # Determine hemisphere
                if filename.endswith('-L.nii'):
                    hemisphere = 'L'
                elif filename.endswith('-R.nii'):
                    hemisphere = 'R'
                else:
                    continue
                
                # Find corresponding ground truth
                gt_file = os.path.join(gt_dir, filename)
                
                if os.path.exists(gt_file):
                    pairs[hemisphere].append((pred_file, gt_file))
                else:
                    print(f"Ground truth not found for: {filename}")
    
    print(f"Found {len(pairs['L'])} left hemisphere pairs and {len(pairs['R'])} right hemisphere pairs")
    return pairs

def create_slice_wise_plots():
    """
    Create and save slice-wise DSC box-whisker plots for both hemispheres.
    """
    print("Collecting prediction-ground truth pairs...")
    all_pairs = collect_all_predictions()
    
    if not all_pairs['L'] and not all_pairs['R']:
        print("No prediction-ground truth pairs found!")
        return
    
    # Process each hemisphere
    for hemisphere in ['L', 'R']:
        hemisphere_name = "Left (LICA)" if hemisphere == 'L' else "Right (RICA)"
        print(f"\nProcessing {hemisphere_name} hemisphere...")
        
        pairs = all_pairs[hemisphere]
        if not pairs:
            print(f"No pairs found for {hemisphere_name} hemisphere")
            continue
        
        # Collect slice-wise DSC data
        all_slice_dsc = []
        sample_names = []
        
        for pred_path, gt_path in pairs:
            slice_dsc = calculate_slice_wise_dsc(pred_path, gt_path)
            
            if slice_dsc:
                all_slice_dsc.append(slice_dsc)
                sample_name = os.path.basename(pred_path).replace('.nii', '')
                sample_names.append(sample_name)
        
        if not all_slice_dsc:
            print(f"No valid DSC data for {hemisphere_name} hemisphere")
            continue
        
        # Convert to DataFrame for easier plotting
        num_slices = len(all_slice_dsc[0]) if all_slice_dsc else 14
        plot_data = []
        
        for sample_idx, slice_dsc in enumerate(all_slice_dsc):
            for slice_idx, dsc in enumerate(slice_dsc):
                plot_data.append({
                    'Slice': slice_idx + 1,  # 1-indexed for plotting
                    'DSC': dsc,
                    'Sample': sample_names[sample_idx]
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Create box plot with notches
        box_plot = plt.boxplot([df[df['Slice'] == slice]['DSC'].values 
                               for slice in range(1, num_slices + 1)],
                              positions=range(1, num_slices + 1),
                              notch=True,
                              patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2),
                              whiskerprops=dict(color='black', linewidth=1.5),
                              capprops=dict(color='black', linewidth=1.5))
        
        # Customize plot
        plt.xlabel('Slice ID', fontsize=14, fontweight='bold')
        plt.ylabel('DSC (Dice Similarity Coefficient)', fontsize=14, fontweight='bold')
        plt.title(f'Slice-wise DSC Distribution - {hemisphere_name} Hemisphere\n'
                 f'Cross-validation Results (n={len(all_slice_dsc)} samples)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis ticks
        plt.xticks(range(1, num_slices + 1))
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        plt.ylim(0, 1.02)
        
        # Add statistics text
        overall_mean = df['DSC'].mean()
        overall_std = df['DSC'].std()
        
        plt.text(0.02, 0.98, f'Overall Mean DSC: {overall_mean:.3f} +/- {overall_std:.3f}\n'
                            f'Total samples: {len(all_slice_dsc)}\n'
                            f'Slices per volume: {num_slices}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("model_evaluation/inter-slice_plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"slice_wise_dsc_{hemisphere}_hemisphere.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        print(f"Saved plot: {output_file}")
        plt.close()
        
        # Print summary statistics
        print(f"\n{hemisphere_name} Hemisphere Summary:")
        print(f"- Samples processed: {len(all_slice_dsc)}")
        print(f"- Slices per volume: {num_slices}")
        print(f"- Overall mean DSC: {overall_mean:.4f} +/- {overall_std:.4f}")
        print(f"- DSC range: {df['DSC'].min():.4f} - {df['DSC'].max():.4f}")
        
        # Slice-wise statistics
        print("\nSlice-wise DSC statistics:")
        for slice_idx in range(1, num_slices + 1):
            slice_data = df[df['Slice'] == slice_idx]['DSC']
            print(f"  Slice {slice_idx:2d}: {slice_data.mean():.4f} +/- {slice_data.std():.4f} "
                  f"(min: {slice_data.min():.4f}, max: {slice_data.max():.4f})")

def main():
    """Main function to execute the slice-wise DSC analysis."""
    print("=" * 70)
    print("Slice-wise DSC Analysis for nnUNet Perfusion Territory Segmentation")
    print("=" * 70)
    
    # Check if required directories exist
    base_dir = "data/TrainingsResults-PerfTerr"
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} not found!")
        return
    
    gt_dir = f"{base_dir}/nnUNet_raw_250827-PerfTerr-25/Dataset001_PerfusionTerritories/labelsTr"
    if not os.path.exists(gt_dir):
        print(f"Error: Ground truth directory {gt_dir} not found!")
        return
    
    # Create plots
    create_slice_wise_plots()
    
    print("\n" + "=" * 70)
    print("Analysis completed! Check model_evaluation/inter-slice_plots/ for results.")
    print("=" * 70)

if __name__ == "__main__":
    main()