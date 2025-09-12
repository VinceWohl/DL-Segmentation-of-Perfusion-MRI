#!/usr/bin/env python3
"""
Slice-wise Hausdorff Distance (95th percentile) Analysis for nnUNet Perfusion Territory Segmentation Results

This script creates box-whisker plots showing the 95th percentile Hausdorff Distance (HD95)
for each slice across the z-dimension (14 slices per volume) for both left (LICA)
and right (RICA) hemisphere perfusion territories.

Why not normal DSC (as a size-sensitive metric)?
Boundary effects: In smaller regions, boundary errors have a proportionally larger impact
Pixel-level errors: A few misclassified pixels affect small regions much more than large regions
That's why HD95:
Measures pure boundary accuracy independent of region size
Focuses on contour quality rather than volumetric overlap
Is truly size-independent without artificial normalization

But still:
Statistical variance: Smaller regions are more susceptible to random variations

HD95 is a boundary-focused metric that is size-independent and measures contour accuracy.
Filters out slices with no ground truth segmentation (infinite HD95).

Author: Generated for DLSegPerf project
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion
import warnings
warnings.filterwarnings('ignore')

def get_contour_points(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract contour (boundary) points from binary mask.
    
    Args:
        binary_mask: 2D binary mask
        
    Returns:
        Array of contour points as (N, 2) coordinates
    """
    if np.sum(binary_mask) == 0:
        return np.array([])
    
    # Find boundary by subtracting eroded mask from original
    eroded = binary_erosion(binary_mask)
    contour = binary_mask.astype(bool) & ~eroded
    
    # Get coordinates of boundary points
    contour_points = np.column_stack(np.where(contour))
    
    return contour_points

def calculate_hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate 95th percentile Hausdorff Distance between prediction and ground truth contours.
    
    Args:
        pred: Prediction binary mask (2D)
        gt: Ground truth binary mask (2D)
        
    Returns:
        HD95 distance in pixels (float)
    """
    # Check if either mask is empty
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.inf  # Return infinity for empty masks
    
    # Get contour points (boundary pixels)
    pred_contour = get_contour_points(pred)
    gt_contour = get_contour_points(gt)
    
    if len(pred_contour) == 0 or len(gt_contour) == 0:
        return np.inf
    
    # Calculate distances from pred to gt
    distances_pred_to_gt = cdist(pred_contour, gt_contour, metric='euclidean')
    min_distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)
    
    # Calculate distances from gt to pred
    distances_gt_to_pred = cdist(gt_contour, pred_contour, metric='euclidean')
    min_distances_gt_to_pred = np.min(distances_gt_to_pred, axis=1)
    
    # Combine all distances
    all_distances = np.concatenate([min_distances_pred_to_gt, min_distances_gt_to_pred])
    
    # Calculate 95th percentile
    hd95 = np.percentile(all_distances, 95)
    
    return hd95

def calculate_slice_wise_hd95(pred_path: str, gt_path: str) -> Dict:
    """
    Calculate slice-wise HD95 for a single prediction-ground truth pair.
    Filters out slices with no segmented region (empty ground truth).
    
    Args:
        pred_path: Path to prediction NIfTI file
        gt_path: Path to ground truth NIfTI file
        
    Returns:
        Dictionary with slice indices and HD95 values for valid slices only
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
        
        # Calculate HD95 for each slice
        num_slices = pred_data.shape[-1]
        slice_data = {'slice_ids': [], 'hd95_values': []}
        
        for slice_idx in range(num_slices):
            pred_slice = pred_data[:, :, slice_idx]
            gt_slice = gt_data[:, :, slice_idx]
            
            # Skip slices with no ground truth segmentation
            if np.sum(gt_slice) == 0:
                continue
                
            hd95_value = calculate_hausdorff_distance_95(pred_slice, gt_slice)
            
            # Skip slices with infinite HD95 (no prediction)
            if np.isinf(hd95_value):
                continue
                
            slice_data['slice_ids'].append(slice_idx + 1)  # 1-indexed
            slice_data['hd95_values'].append(hd95_value)
            
        return slice_data
        
    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        return {'slice_ids': [], 'hd95_values': []}

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
    Create and save slice-wise HD95 plots for both hemispheres.
    """
    print("Collecting prediction-ground truth pairs...")
    all_pairs = collect_all_predictions()
    
    if not all_pairs['L'] and not all_pairs['R']:
        print("No prediction-ground truth pairs found!")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each hemisphere
    for hemisphere in ['L', 'R']:
        hemisphere_name = "Left (LICA)" if hemisphere == 'L' else "Right (RICA)"
        print(f"\nProcessing {hemisphere_name} hemisphere...")
        
        pairs = all_pairs[hemisphere]
        if not pairs:
            print(f"No pairs found for {hemisphere_name} hemisphere")
            continue
        
        # Collect slice-wise HD95 data
        plot_data = []
        sample_count = 0
        
        for pred_path, gt_path in pairs:
            slice_data = calculate_slice_wise_hd95(pred_path, gt_path)
            
            if slice_data['hd95_values']:
                sample_count += 1
                sample_name = os.path.basename(pred_path).replace('.nii', '')
                
                for slice_id, hd95_val in zip(slice_data['slice_ids'], slice_data['hd95_values']):
                    plot_data.append({
                        'Slice': slice_id,
                        'HD95': hd95_val,
                        'Sample': sample_name
                    })
        
        if not plot_data:
            print(f"No valid HD95 data for {hemisphere_name} hemisphere")
            continue
        
        df = pd.DataFrame(plot_data)
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Get unique slices and create box plot
        unique_slices = sorted(df['Slice'].unique())
        box_data = [df[df['Slice'] == slice_id]['HD95'].values for slice_id in unique_slices]
        
        box_plot = plt.boxplot(box_data,
                              positions=unique_slices,
                              notch=True,
                              patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2),
                              whiskerprops=dict(color='black', linewidth=1.5),
                              capprops=dict(color='black', linewidth=1.5))
        
        # Customize plot
        plt.xlabel('Slice ID', fontsize=14, fontweight='bold')
        plt.ylabel('HD95 (pixels)', fontsize=14, fontweight='bold')
        plt.title(f'Slice-wise Hausdorff Distance (95th percentile) - {hemisphere_name} Hemisphere\n'
                 f'Cross-validation Results', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.xlim(0.5, 14.5)
        plt.ylim(0, 20)
        
        plt.xticks(range(1, 15))
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Calculate statistics
        overall_mean = df['HD95'].mean()
        overall_std = df['HD95'].std()
        total_valid_slices = len(df)
        
        # Create clean legend text
        legend_text = f'Mean HD95: {overall_mean:.2f} ± {overall_std:.2f} pixels\n'
        legend_text += f'Total valid slices: {total_valid_slices}\n'
        legend_text += f'Samples processed: {sample_count}\n'
        legend_text += f'Slices per volume: {len(unique_slices)}'
        
        # Position legend below plot
        plt.text(0.5, -0.12, legend_text,
                 transform=plt.gca().transAxes,
                 horizontalalignment='center',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=11)
        
        plt.tight_layout()
        
        # Save plot with simple naming
        output_dir = Path("model_evaluation/inter-slice_plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"slice_wise_hausdorff_{hemisphere}_hemisphere_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        print(f"Saved plot: {output_file}")
        plt.close()
        
        # Print summary statistics
        print(f"\n{hemisphere_name} Hemisphere Summary:")
        print(f"- Samples processed: {sample_count}")
        print(f"- Valid slices: {total_valid_slices}")
        print(f"- Overall mean HD95: {overall_mean:.2f} ± {overall_std:.2f} pixels")
        print(f"- HD95 range: {df['HD95'].min():.2f} - {df['HD95'].max():.2f} pixels")

def main():
    """Main function to execute the slice-wise HD95 analysis."""
    print("=" * 80)
    print("Slice-wise Hausdorff Distance (95th percentile) Analysis for nnUNet Perfusion Territory Segmentation")
    print("=" * 80)
    
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
    
    print("\n" + "=" * 80)
    print("Analysis completed! Check model_evaluation/inter-slice_plots/ for results.")
    print("=" * 80)

if __name__ == "__main__":
    main()