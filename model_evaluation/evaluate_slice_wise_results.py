#!/usr/bin/env python3
"""
Slice-wise evaluation script for nnUNet multi-label perfusion territory segmentation.

This script compares validation predictions from each fold against ground truth labels,
computing slice-wise Dice scores while excluding slices with only zeros in ground truth.
Evaluation is performed separately for left and right hemispheres.
"""

import os
import sys
import json
import numpy as np
import SimpleITK as sitk

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def load_nii_image(file_path):
    """Load NIfTI image and return as numpy array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)

def compute_dice_score(pred, gt):
    """Compute Dice score for binary masks."""
    intersection = np.sum(pred * gt)
    total = np.sum(pred) + np.sum(gt)
    
    if total == 0:
        return 1.0  # Both empty
    else:
        return 2.0 * intersection / total

def extract_channels(image_array):
    """Extract left and right channels from multi-channel image."""
    if image_array.ndim == 4:
        if image_array.shape[-1] == 2:
            # Format: (slices, H, W, channels)
            left_channel = image_array[..., 0]
            right_channel = image_array[..., 1]
        elif image_array.shape[0] == 2:
            # Format: (channels, slices, H, W)
            left_channel = image_array[0, ...]
            right_channel = image_array[1, ...]
        else:
            raise ValueError(f"Unexpected 4D image shape: {image_array.shape}")
    else:
        raise ValueError(f"Expected 4D image, got {image_array.ndim}D")
    
    return left_channel, right_channel

def evaluate_fold(fold_dir, gt_dir, fold_number):
    """Evaluate a single fold against ground truth."""
    print(f"\n=== Evaluating Fold {fold_number} ===")
    
    validation_dir = os.path.join(fold_dir, 'validation')
    if not os.path.exists(validation_dir):
        print(f"Warning: Validation directory not found: {validation_dir}")
        return None
    
    # Get all prediction files
    pred_files = [f for f in os.listdir(validation_dir) if f.endswith('.nii')]
    print(f"Found {len(pred_files)} prediction files")
    
    results = {
        'fold': fold_number,
        'cases': [],
        'left_hemisphere_dice': [],
        'right_hemisphere_dice': [],
        'slice_info': []
    }
    
    for pred_file in tqdm(pred_files, desc=f"Processing fold {fold_number}"):
        case_id = pred_file.replace('.nii', '')
        
        # Load prediction
        pred_path = os.path.join(validation_dir, pred_file)
        gt_path = os.path.join(gt_dir, f"{case_id}.nii")
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {case_id}")
            continue
        
        try:
            # Load images
            pred_array = load_nii_image(pred_path)
            gt_array = load_nii_image(gt_path)
            
            # Extract channels
            pred_left, pred_right = extract_channels(pred_array)
            gt_left, gt_right = extract_channels(gt_array)
            
            # Convert to binary
            pred_left = (pred_left > 0.5).astype(np.uint8)
            pred_right = (pred_right > 0.5).astype(np.uint8)
            gt_left = (gt_left > 0).astype(np.uint8)
            gt_right = (gt_right > 0).astype(np.uint8)
            
            # Process each slice
            case_left_dices = []
            case_right_dices = []
            
            for slice_idx in range(pred_left.shape[0]):
                # Get slice data
                pred_left_slice = pred_left[slice_idx]
                pred_right_slice = pred_right[slice_idx]
                gt_left_slice = gt_left[slice_idx]
                gt_right_slice = gt_right[slice_idx]
                
                # Skip slices with no ground truth
                if np.sum(gt_left_slice) == 0 and np.sum(gt_right_slice) == 0:
                    continue
                
                # Compute dice for left hemisphere (only if GT has content)
                if np.sum(gt_left_slice) > 0:
                    left_dice = compute_dice_score(pred_left_slice, gt_left_slice)
                    case_left_dices.append(left_dice)
                    results['left_hemisphere_dice'].append(left_dice)
                
                # Compute dice for right hemisphere (only if GT has content)
                if np.sum(gt_right_slice) > 0:
                    right_dice = compute_dice_score(pred_right_slice, gt_right_slice)
                    case_right_dices.append(right_dice)
                    results['right_hemisphere_dice'].append(right_dice)
                
                # Store slice info
                results['slice_info'].append({
                    'case_id': case_id,
                    'slice_idx': slice_idx,
                    'fold': fold_number,
                    'left_dice': left_dice if np.sum(gt_left_slice) > 0 else None,
                    'right_dice': right_dice if np.sum(gt_right_slice) > 0 else None,
                    'gt_left_pixels': int(np.sum(gt_left_slice)),
                    'gt_right_pixels': int(np.sum(gt_right_slice))
                })
            
            # Store case-level results
            results['cases'].append({
                'case_id': case_id,
                'left_mean_dice': np.mean(case_left_dices) if case_left_dices else None,
                'right_mean_dice': np.mean(case_right_dices) if case_right_dices else None,
                'num_left_slices': len(case_left_dices),
                'num_right_slices': len(case_right_dices)
            })
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            continue
    
    print(f"Fold {fold_number} results:")
    print(f"  Left hemisphere: {len(results['left_hemisphere_dice'])} slices, "
          f"mean Dice: {np.mean(results['left_hemisphere_dice']):.4f}")
    print(f"  Right hemisphere: {len(results['right_hemisphere_dice'])} slices, "
          f"mean Dice: {np.mean(results['right_hemisphere_dice']):.4f}")
    
    return results

def create_box_plots(all_results, output_dir):
    """Create box plots for slice-wise Dice scores."""
    print("\n=== Creating Box Plots ===")
    
    # Prepare data for plotting
    plot_data = []
    
    # Collect all data for overall statistics
    all_left_scores = []
    all_right_scores = []
    
    for fold_results in all_results:
        fold_num = fold_results['fold']
        
        # Add left hemisphere data
        for dice in fold_results['left_hemisphere_dice']:
            plot_data.append({
                'Fold': f'Fold {fold_num}',
                'Hemisphere': 'Left',
                'Dice Score': dice
            })
            all_left_scores.append(dice)
        
        # Add right hemisphere data
        for dice in fold_results['right_hemisphere_dice']:
            plot_data.append({
                'Fold': f'Fold {fold_num}',
                'Hemisphere': 'Right',
                'Dice Score': dice
            })
            all_right_scores.append(dice)
    
    # Add overall data for each hemisphere
    for dice in all_left_scores:
        plot_data.append({
            'Fold': 'Overall',
            'Hemisphere': 'Left',
            'Dice Score': dice
        })
    
    for dice in all_right_scores:
        plot_data.append({
            'Fold': 'Overall',
            'Hemisphere': 'Right',
            'Dice Score': dice
        })
    
    df = pd.DataFrame(plot_data)
    
    # Calculate statistics for annotations
    fold_stats = {}
    for fold_num in range(5):
        fold_left = df[(df['Fold'] == f'Fold {fold_num}') & (df['Hemisphere'] == 'Left')]['Dice Score']
        fold_right = df[(df['Fold'] == f'Fold {fold_num}') & (df['Hemisphere'] == 'Right')]['Dice Score']
        
        fold_stats[f'Fold {fold_num}'] = {
            'Left': {'mean': fold_left.mean(), 'std': fold_left.std()},
            'Right': {'mean': fold_right.mean(), 'std': fold_right.std()}
        }
    
    # Overall statistics
    overall_left = df[(df['Fold'] == 'Overall') & (df['Hemisphere'] == 'Left')]['Dice Score']
    overall_right = df[(df['Fold'] == 'Overall') & (df['Hemisphere'] == 'Right')]['Dice Score']
    
    fold_stats['Overall'] = {
        'Left': {'mean': overall_left.mean(), 'std': overall_left.std()},
        'Right': {'mean': overall_right.mean(), 'std': overall_right.std()}
    }
    
    # Create separate plots for each hemisphere
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left hemisphere plot
    left_data = df[df['Hemisphere'] == 'Left']
    box_plot_left = sns.boxplot(data=left_data, x='Fold', y='Dice Score', ax=axes[0], 
                               color='lightblue', showfliers=True, flierprops=dict(marker='o', markersize=4))
    axes[0].set_title('Left Hemisphere - Slice-wise Dice Scores', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Dice Score', fontsize=12)
    axes[0].set_xlabel('Fold', fontsize=12)
    
    # Add statistics annotations for left hemisphere
    fold_names = ['Fold 0', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Overall']
    for i, fold in enumerate(fold_names):
        mean_val = fold_stats[fold]['Left']['mean']
        std_val = fold_stats[fold]['Left']['std']
        # Position boxes higher and with more spacing
        y_pos = 0.02 + (i % 2) * 0.08  # Alternate heights to avoid overlap
        axes[0].text(i, y_pos, f'{mean_val:.4f}\n± {std_val:.4f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # Right hemisphere plot
    right_data = df[df['Hemisphere'] == 'Right']
    box_plot_right = sns.boxplot(data=right_data, x='Fold', y='Dice Score', ax=axes[1], 
                                color='lightcoral', showfliers=True, flierprops=dict(marker='o', markersize=4))
    axes[1].set_title('Right Hemisphere - Slice-wise Dice Scores', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_xlabel('Fold', fontsize=12)
    
    # Add statistics annotations for right hemisphere
    for i, fold in enumerate(fold_names):
        mean_val = fold_stats[fold]['Right']['mean']
        std_val = fold_stats[fold]['Right']['std']
        # Position boxes higher and with more spacing
        y_pos = 0.02 + (i % 2) * 0.08  # Alternate heights to avoid overlap
        axes[1].text(i, y_pos, f'{mean_val:.4f}\n± {std_val:.4f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slice_wise_dice_box_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined plot with overall statistics
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create the box plot
    box_plot_combined = sns.boxplot(data=df, x='Fold', y='Dice Score', hue='Hemisphere', ax=ax, 
                                   palette=['lightblue', 'lightcoral'], showfliers=True, 
                                   flierprops=dict(marker='o', markersize=4))
    ax.set_title('Slice-wise Dice Scores by Fold and Hemisphere', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_xlabel('Fold', fontsize=12)
    
    # Move legend to the right side outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add statistics annotations for combined plot - all at the same height
    for i, fold in enumerate(fold_names):
        # Use same y position for all annotations
        y_pos = 0.02
        
        # Left hemisphere annotation (slightly left of center)
        left_mean = fold_stats[fold]['Left']['mean']
        left_std = fold_stats[fold]['Left']['std']
        ax.text(i - 0.2, y_pos, f'{left_mean:.4f}\n± {left_std:.4f}', 
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.15", facecolor="lightblue", alpha=0.9))
        
        # Right hemisphere annotation (slightly right of center)
        right_mean = fold_stats[fold]['Right']['mean']
        right_std = fold_stats[fold]['Right']['std']
        ax.text(i + 0.2, y_pos, f'{right_mean:.4f}\n± {right_std:.4f}', 
                ha='center', va='bottom', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.15", facecolor="lightcoral", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_slice_wise_dice_box_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Box plots saved to {output_dir}")

def generate_summary_statistics(all_results, output_dir):
    """Generate summary statistics across all folds."""
    print("\n=== Generating Summary Statistics ===")
    
    # Collect all scores
    all_left_scores = []
    all_right_scores = []
    fold_summaries = []
    
    for fold_results in all_results:
        fold_num = fold_results['fold']
        left_scores = fold_results['left_hemisphere_dice']
        right_scores = fold_results['right_hemisphere_dice']
        
        all_left_scores.extend(left_scores)
        all_right_scores.extend(right_scores)
        
        fold_summaries.append({
            'fold': fold_num,
            'left_mean': np.mean(left_scores),
            'left_std': np.std(left_scores),
            'left_median': np.median(left_scores),
            'left_min': np.min(left_scores),
            'left_max': np.max(left_scores),
            'left_n_slices': len(left_scores),
            'right_mean': np.mean(right_scores),
            'right_std': np.std(right_scores),
            'right_median': np.median(right_scores),
            'right_min': np.min(right_scores),
            'right_max': np.max(right_scores),
            'right_n_slices': len(right_scores)
        })
    
    # Overall statistics
    overall_stats = {
        'evaluation_type': 'slice_wise_dice_scores',
        'total_folds': len(all_results),
        'left_hemisphere': {
            'total_slices': len(all_left_scores),
            'mean_dice': float(np.mean(all_left_scores)),
            'std_dice': float(np.std(all_left_scores)),
            'median_dice': float(np.median(all_left_scores)),
            'min_dice': float(np.min(all_left_scores)),
            'max_dice': float(np.max(all_left_scores)),
            'q25_dice': float(np.percentile(all_left_scores, 25)),
            'q75_dice': float(np.percentile(all_left_scores, 75))
        },
        'right_hemisphere': {
            'total_slices': len(all_right_scores),
            'mean_dice': float(np.mean(all_right_scores)),
            'std_dice': float(np.std(all_right_scores)),
            'median_dice': float(np.median(all_right_scores)),
            'min_dice': float(np.min(all_right_scores)),
            'max_dice': float(np.max(all_right_scores)),
            'q25_dice': float(np.percentile(all_right_scores, 25)),
            'q75_dice': float(np.percentile(all_right_scores, 75))
        },
        'per_fold_summary': fold_summaries
    }
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, 'slice_wise_evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"Summary statistics saved to {summary_file}")
    
    # Print summary
    print("\n=== SLICE-WISE EVALUATION SUMMARY ===")
    print(f"Total folds evaluated: {len(all_results)}")
    print(f"Left hemisphere: {len(all_left_scores)} slices")
    print(f"  Mean Dice: {np.mean(all_left_scores):.4f} ± {np.std(all_left_scores):.4f}")
    print(f"  Median Dice: {np.median(all_left_scores):.4f}")
    print(f"  Range: [{np.min(all_left_scores):.4f}, {np.max(all_left_scores):.4f}]")
    print(f"Right hemisphere: {len(all_right_scores)} slices")
    print(f"  Mean Dice: {np.mean(all_right_scores):.4f} ± {np.std(all_right_scores):.4f}")
    print(f"  Median Dice: {np.median(all_right_scores):.4f}")
    print(f"  Range: [{np.min(all_right_scores):.4f}, {np.max(all_right_scores):.4f}]")
    print("=" * 50)

def main():
    """Main evaluation function."""
    # Define paths
    base_dir = "/home/vincent/projects/claude_project_folder/nnUNet_multi-label"
    results_dir = os.path.join(base_dir, "nnUNet_results")
    gt_dir = os.path.join(base_dir, "nnUNet_raw/Dataset001_PerfusionTerritories/labelsTr")
    output_dir = os.path.join(base_dir, "evaluation_results")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find dataset results directory
    dataset_dirs = [d for d in os.listdir(results_dir) if d.startswith("Dataset001_PerfusionTerritories")]
    
    if not dataset_dirs:
        print("Error: No dataset results found")
        return
    
    dataset_dir = os.path.join(results_dir, dataset_dirs[0])
    trainer_dir = os.path.join(dataset_dir, "nnUNetTrainer_SharedDecoder__nnUNetPlans__2d")
    
    if not os.path.exists(trainer_dir):
        print(f"Error: Trainer directory not found: {trainer_dir}")
        return
    
    print(f"Evaluating results from: {trainer_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all fold directories
    fold_dirs = [d for d in os.listdir(trainer_dir) if d.startswith("fold_")]
    fold_dirs.sort()
    
    if not fold_dirs:
        print("Error: No fold directories found")
        return
    
    print(f"Found {len(fold_dirs)} folds: {fold_dirs}")
    
    # Evaluate all folds
    all_results = []
    
    for fold_dir in fold_dirs:
        fold_number = int(fold_dir.split('_')[1])
        fold_path = os.path.join(trainer_dir, fold_dir)
        
        fold_results = evaluate_fold(fold_path, gt_dir, fold_number)
        if fold_results:
            all_results.append(fold_results)
    
    if not all_results:
        print("Error: No fold results obtained")
        return
    
    # Save detailed results
    detailed_results_file = os.path.join(output_dir, 'detailed_slice_wise_results.json')
    with open(detailed_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Detailed results saved to {detailed_results_file}")
    
    # Generate summary statistics
    generate_summary_statistics(all_results, output_dir)
    
    # Create box plots
    create_box_plots(all_results, output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()