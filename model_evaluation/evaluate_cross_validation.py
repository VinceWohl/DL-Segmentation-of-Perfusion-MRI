#!/usr/bin/env python3
"""
Cross-Validation nnUNet Results Evaluation Script

This script evaluates nnUNet cross-validation results by comparing predictions 
with ground truth masks using comprehensive metrics:

- Dice Score: Volume overlap coefficient  
- Dice Score (Slice-wise): Mean slice-wise Dice coefficient
- Hausdorff Distance 95th percentile (HD95): Surface distance metric
- Sensitivity/Specificity: Classification performance metrics
- Volume metrics: GT volume, predicted volume, volume difference
- Basic cardinalities: True Positives, False Positives, False Negatives, True Negatives

Input folder structure:
- data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/
  - fold_0/validation/
  - fold_1/validation/  
  - fold_2/validation/
  - fold_3/validation/
  - fold_4/validation/

Ground truth reference:
- data/nnUNet_preprocessed/Dataset001_PerfusionTerritories/gt_segmentations/

Output:
- Excel file with detailed results and summary statistics
- Visualization plots for metrics distribution

Author: Generated for nnUNet cross-validation evaluation
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy.spatial.distance import cdist

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class CrossValidationEvaluator:
    def __init__(self, results_root, output_dir):
        self.results_root = Path(results_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        self.gt_dir = Path("/home/ubuntu/DLSegPerf/data/nnUNet_preprocessed/Dataset001_PerfusionTerritories/gt_segmentations")
        
        # Results storage
        self.results = []
        self.fold_summaries = []
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []
    
    def load_nifti_file(self, file_path):
        """Load a NIfTI file and return the data array and voxel spacing"""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            # Convert to binary mask (ensure 0/1 values)
            mask = (data > 0).astype(np.uint8)
            # Get voxel spacing in mm
            spacing = nii.header.get_zooms()[:3]  # x, y, z spacing
            return mask, spacing
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def calculate_dice_score(self, gt_mask, pred_mask):
        """Calculate Dice similarity coefficient (volume-wise)"""
        intersection = np.sum(gt_mask * pred_mask)
        union = np.sum(gt_mask) + np.sum(pred_mask)
        
        if union == 0:
            return 1.0 if np.sum(gt_mask) == np.sum(pred_mask) else 0.0
        
        dice = (2.0 * intersection) / union
        return dice
    
    def calculate_dice_score_slicewise(self, gt_mask, pred_mask):
        """Calculate slice-wise Dice similarity coefficient (mean across slices)"""
        if gt_mask.shape != pred_mask.shape:
            return float('nan')
        
        nz = gt_mask.shape[2]
        slice_dice_scores = []
        
        for z in range(nz):
            gt_slice = gt_mask[:, :, z]
            pred_slice = pred_mask[:, :, z]
            
            intersection = np.sum(gt_slice * pred_slice)
            union = np.sum(gt_slice) + np.sum(pred_slice)
            
            if union == 0:
                slice_dice = 1.0 if np.sum(gt_slice) == np.sum(pred_slice) else 0.0
            else:
                slice_dice = (2.0 * intersection) / union
            
            slice_dice_scores.append(slice_dice)
        
        return np.mean(slice_dice_scores)
    
    def calculate_cardinalities(self, gt_mask, pred_mask):
        """Calculate True Positives, False Positives, False Negatives, True Negatives"""
        tp = np.sum((gt_mask == 1) & (pred_mask == 1))
        fp = np.sum((gt_mask == 0) & (pred_mask == 1))
        fn = np.sum((gt_mask == 1) & (pred_mask == 0))
        tn = np.sum((gt_mask == 0) & (pred_mask == 0))
        
        return int(tp), int(fp), int(fn), int(tn)
    
    def get_surface_points(self, mask, spacing):
        """Extract surface points from binary mask with physical coordinates"""
        # Get the boundary/edge of the mask
        structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        eroded = ndimage.binary_erosion(mask, structure)
        boundary = mask & ~eroded
        
        # Get coordinates of boundary points
        coords = np.array(np.where(boundary)).T
        
        # Convert to physical coordinates using spacing
        if len(coords) > 0:
            physical_coords = coords * np.array(spacing)
            return physical_coords
        else:
            return coords
    
    def calculate_hausdorff_distance_95(self, gt_mask, pred_mask, spacing):
        """Calculate 95th percentile Hausdorff distance in millimeters"""
        try:
            # Get surface points for both masks
            gt_surface = self.get_surface_points(gt_mask, spacing)
            pred_surface = self.get_surface_points(pred_mask, spacing)
            
            # Handle edge cases
            if len(gt_surface) == 0 and len(pred_surface) == 0:
                return 0.0  # Both masks are empty
            elif len(gt_surface) == 0 or len(pred_surface) == 0:
                return float('inf')  # One mask is empty, other is not
            
            # Calculate distances from GT surface to predicted surface
            distances_gt_to_pred = cdist(gt_surface, pred_surface).min(axis=1)
            
            # Calculate distances from predicted surface to GT surface  
            distances_pred_to_gt = cdist(pred_surface, gt_surface).min(axis=1)
            
            # Combine all distances
            all_distances = np.concatenate([distances_gt_to_pred, distances_pred_to_gt])
            
            # Calculate 95th percentile
            hd95 = np.percentile(all_distances, 95)
            return float(hd95)
            
        except Exception as e:
            print(f"Warning: HD95 calculation failed: {e}")
            return float('nan')
    
    def calculate_sensitivity_specificity(self, tp, fp, fn, tn):
        """Calculate sensitivity and specificity"""
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sensitivity, specificity
    
    def evaluate_single_case(self, pred_file, gt_file, fold, base_name):
        """Evaluate a single case prediction against ground truth"""
        print(f"  Evaluating {base_name} (Fold {fold})...")
        
        # Load masks
        pred_mask, pred_spacing = self.load_nifti_file(pred_file)
        gt_mask, gt_spacing = self.load_nifti_file(gt_file)
        
        if gt_mask is None or pred_mask is None:
            print(f"    ERROR: Failed to load masks")
            return None
        
        # Use GT spacing (should be same for both)
        spacing = gt_spacing
        
        # Ensure shapes match
        if gt_mask.shape != pred_mask.shape:
            print(f"    ERROR: Shape mismatch - GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            return None
        
        # Calculate metrics
        dice_score = self.calculate_dice_score(gt_mask, pred_mask)
        dice_score_slicewise = self.calculate_dice_score_slicewise(gt_mask, pred_mask)
        tp, fp, fn, tn = self.calculate_cardinalities(gt_mask, pred_mask)
        sensitivity, specificity = self.calculate_sensitivity_specificity(tp, fp, fn, tn)
        hd95 = self.calculate_hausdorff_distance_95(gt_mask, pred_mask, spacing)
        
        # Calculate additional metrics
        voxel_volume_mm3 = np.prod(spacing)  # mm³ per voxel
        gt_volume_voxels = np.sum(gt_mask)
        pred_volume_voxels = np.sum(pred_mask)
        gt_volume_mm3 = gt_volume_voxels * voxel_volume_mm3
        pred_volume_mm3 = pred_volume_voxels * voxel_volume_mm3
        volume_difference_mm3 = pred_volume_mm3 - gt_volume_mm3
        
        print(f"    Dice: {dice_score:.4f}, Dice(slice): {dice_score_slicewise:.4f}, HD95: {hd95:.2f}")
        
        return {
            'Fold': fold,
            'Base_Name': base_name,
            'Dice_Score': dice_score,
            'Dice_Score_Slicewise': dice_score_slicewise,
            'HD95_mm': hd95,
            'True_Positives': tp,
            'False_Positives': fp,
            'False_Negatives': fn,
            'True_Negatives': tn,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'GT_Volume_mm3': gt_volume_mm3,
            'Pred_Volume_mm3': pred_volume_mm3,
            'Volume_Difference_mm3': volume_difference_mm3
        }
    
    def find_fold_cases(self):
        """Find all validation cases across all folds"""
        fold_cases = {}
        
        for fold_num in range(5):  # Standard 5-fold CV
            fold_dir = self.results_root / f"fold_{fold_num}" / "validation"
            if not fold_dir.exists():
                print(f"Warning: Fold {fold_num} directory not found: {fold_dir}")
                continue
            
            # Find all .nii files in validation directory
            pred_files = list(fold_dir.glob("*.nii"))
            fold_cases[fold_num] = []
            
            for pred_file in pred_files:
                base_name = pred_file.stem
                gt_file = self.gt_dir / f"{base_name}.nii"
                
                if gt_file.exists():
                    fold_cases[fold_num].append((pred_file, gt_file, base_name))
                else:
                    print(f"Warning: No matching GT file for {base_name}")
        
        return fold_cases
    
    def load_fold_summary(self, fold_num):
        """Load validation summary from fold's summary.json"""
        summary_file = self.results_root / f"fold_{fold_num}" / "validation" / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                return {
                    'fold': fold_num,
                    'mean_dice': summary['foreground_mean']['Dice'],
                    'mean_iou': summary['foreground_mean']['IoU'],
                    'mean_fn': summary['foreground_mean']['FN'],
                    'mean_fp': summary['foreground_mean']['FP'],
                    'mean_tp': summary['foreground_mean']['TP'],
                    'mean_tn': summary['foreground_mean']['TN'],
                    'cases_count': len(summary['metric_per_case'])
                }
            except Exception as e:
                print(f"Warning: Could not load summary for fold {fold_num}: {e}")
        return None
    
    def run_evaluation(self):
        """Run the complete cross-validation evaluation"""
        print("Cross-Validation nnUNet Results Evaluation")
        print("=" * 60)
        print(f"Results root: {self.results_root}")
        print(f"Ground truth: {self.gt_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        
        if not PANDAS_AVAILABLE:
            print("ERROR: pandas package is required. Please install with: pip install pandas openpyxl")
            return False
        
        # Check if required directories exist
        if not self.results_root.exists():
            print(f"ERROR: Results directory not found: {self.results_root}")
            return False
        
        if not self.gt_dir.exists():
            print(f"ERROR: Ground truth directory not found: {self.gt_dir}")
            return False
        
        # Load fold summaries
        print("Loading fold summaries...")
        for fold_num in range(5):
            fold_summary = self.load_fold_summary(fold_num)
            if fold_summary:
                self.fold_summaries.append(fold_summary)
                print(f"  Fold {fold_num}: {fold_summary['mean_dice']:.4f} Dice, {fold_summary['cases_count']} cases")
            else:
                print(f"  Fold {fold_num}: Summary not available")
        
        print()
        
        # Find all validation cases
        fold_cases = self.find_fold_cases()
        total_cases = sum(len(cases) for cases in fold_cases.values())
        
        if total_cases == 0:
            print("ERROR: No validation cases found")
            return False
        
        print(f"Found {total_cases} validation cases across {len(fold_cases)} folds")
        
        # Process each fold
        for fold_num, cases in fold_cases.items():
            print(f"\nProcessing Fold {fold_num} ({len(cases)} cases):")
            
            for pred_file, gt_file, base_name in cases:
                self.processed_count += 1
                result = self.evaluate_single_case(pred_file, gt_file, fold_num, base_name)
                
                if result is not None:
                    self.results.append(result)
                    self.successful_count += 1
                else:
                    self.failed_files.append(f"{base_name} (fold {fold_num})")
        
        # Save results to Excel
        print("\nSaving results to Excel...")
        self.save_results_to_excel()
        
        # Create visualization
        print("Creating metrics visualization...")
        self.create_metrics_visualization()
        
        # Print summary
        self.print_evaluation_summary()
        
        return True
    
    def save_results_to_excel(self):
        """Save evaluation results to Excel file with multiple sheets"""
        if not self.results:
            print("No results to save")
            return
        
        try:
            # Create timestamp for output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = self.output_dir / f"cross_validation_evaluation_results_{timestamp}.xlsx"
            
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            # Parse subject information from base name
            df[['Subject', 'Visit', 'Hemisphere']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
            df['Subject'] = 'sub-p' + df['Subject'].str.zfill(3)
            df['Visit'] = 'v' + df['Visit']
            
            # Sort the DataFrame: Fold -> Subject -> Visit -> Hemisphere
            df = df.sort_values(['Fold', 'Subject', 'Visit', 'Hemisphere']).reset_index(drop=True)
            
            # Reorder columns for better readability
            column_order = [
                'Fold', 'Subject', 'Visit', 'Hemisphere', 'Base_Name',
                'Dice_Score', 'Dice_Score_Slicewise', 'HD95_mm', 'Sensitivity', 'Specificity',
                'True_Positives', 'False_Positives', 'False_Negatives', 'True_Negatives',
                'GT_Volume_mm3', 'Pred_Volume_mm3', 'Volume_Difference_mm3'
            ]
            df = df[column_order]
            
            # Create per-fold summary statistics
            fold_summary_stats = []
            for fold_num in range(5):
                fold_data = df[df['Fold'] == fold_num]
                if len(fold_data) > 0:
                    fold_summary_stats.append({
                        'Fold': fold_num,
                        'Cases': len(fold_data),
                        'Mean_Dice': fold_data['Dice_Score'].mean(),
                        'Std_Dice': fold_data['Dice_Score'].std(),
                        'Mean_Dice_Slicewise': fold_data['Dice_Score_Slicewise'].mean(),
                        'Std_Dice_Slicewise': fold_data['Dice_Score_Slicewise'].std(),
                        'Mean_HD95': fold_data['HD95_mm'].replace([np.inf, -np.inf], np.nan).mean(),
                        'Std_HD95': fold_data['HD95_mm'].replace([np.inf, -np.inf], np.nan).std(),
                        'Mean_Sensitivity': fold_data['Sensitivity'].mean(),
                        'Mean_Specificity': fold_data['Specificity'].mean()
                    })
            
            fold_summary_df = pd.DataFrame(fold_summary_stats)
            
            # Create overall summary statistics
            numeric_columns = ['Dice_Score', 'Dice_Score_Slicewise', 'HD95_mm', 'Sensitivity', 'Specificity',
                              'True_Positives', 'False_Positives', 'False_Negatives', 'True_Negatives',
                              'GT_Volume_mm3', 'Pred_Volume_mm3', 'Volume_Difference_mm3']
            
            overall_summary_stats = []
            for col in numeric_columns:
                if col in df.columns:
                    values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) > 0:
                        overall_summary_stats.append({
                            'Metric': col,
                            'Mean': values.mean(),
                            'Std': values.std(),
                            'Min': values.min(),
                            'Max': values.max(),
                            'Median': values.median(),
                            'Count': len(values)
                        })
            
            overall_summary_df = pd.DataFrame(overall_summary_stats)
            
            # Add fold summaries from nnUNet if available
            if self.fold_summaries:
                nnunet_fold_df = pd.DataFrame(self.fold_summaries)
                nnunet_fold_df.columns = ['Fold', 'nnUNet_Mean_Dice', 'nnUNet_Mean_IoU', 
                                         'nnUNet_Mean_FN', 'nnUNet_Mean_FP', 'nnUNet_Mean_TP', 
                                         'nnUNet_Mean_TN', 'nnUNet_Cases_Count']
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Detailed results
                df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Per-fold summary
                fold_summary_df.to_excel(writer, sheet_name='Per_Fold_Summary', index=False)
                
                # Overall summary statistics
                overall_summary_df.to_excel(writer, sheet_name='Overall_Summary', index=False)
                
                # nnUNet fold summaries if available
                if self.fold_summaries:
                    nnunet_fold_df.to_excel(writer, sheet_name='nnUNet_Fold_Summary', index=False)
                
                # Auto-adjust column widths for all sheets
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
            
            print(f"Results saved to Excel: {excel_file}")
            
        except Exception as e:
            print(f"Error saving Excel file: {e}")
    
    def create_metrics_visualization(self):
        """Create comprehensive visualization of evaluation metrics"""
        if not self.results:
            print("No results available for visualization")
            return False
        
        if not VISUALIZATION_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Skipping visualization.")
            return False
        
        # Prepare data for plotting
        df = pd.DataFrame(self.results)
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 2x2 subplot grid
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Cross-Validation nnUNet Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # Metrics data
        dice_vol_values = df['Dice_Score'].values
        dice_slice_values = df['Dice_Score_Slicewise'].values
        hd95_values = df['HD95_mm'].replace([np.inf, -np.inf], np.nan).dropna().values
        sensitivity_values = df['Sensitivity'].values
        
        metrics_data = [
            ('Dice Score\n(Volume)', dice_vol_values),
            ('Dice Score\n(Slice-wise)', dice_slice_values),
            ('HD95\n(mm)', hd95_values),
            ('Sensitivity', sensitivity_values)
        ]
        
        # Create subplots
        for i, (metric_name, values) in enumerate(metrics_data):
            ax = plt.subplot(2, 2, i + 1)
            
            if len(values) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_name.replace('\n', ' '))
                continue
            
            # Create violin plot
            violin_parts = ax.violinplot([values], positions=[1], widths=0.6, 
                                       showmeans=False, showmedians=False)
            
            # Customize violin plot colors
            for pc in violin_parts['bodies']:
                pc.set_facecolor('lightsteelblue')
                pc.set_alpha(0.8)
                pc.set_edgecolor('navy')
                pc.set_linewidth(1.5)
            
            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Add mean as horizontal line
            ax.axhline(y=mean_val, color='red', linestyle='-', linewidth=2, 
                      label=f'Mean: {mean_val:.4f}', zorder=4)
            
            # Add ±1 std lines
            ax.axhline(y=mean_val + std_val, color='orange', linestyle='--', 
                      linewidth=1.5, alpha=0.8, label=f'±1 Std: {std_val:.4f}', zorder=3)
            ax.axhline(y=mean_val - std_val, color='orange', linestyle='--', 
                      linewidth=1.5, alpha=0.8, zorder=3)
            
            # Find and plot outliers
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            if len(outliers) > 0:
                x_positions = np.random.normal(1, 0.02, len(outliers))
                ax.scatter(x_positions, outliers, color='darkred', s=40, alpha=0.9, 
                          zorder=6, edgecolors='black', linewidth=0.5, 
                          label=f'Outliers: {len(outliers)}')
            
            ax.set_title(f'{metric_name.replace(chr(10), " ")}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11)
            ax.set_xticks([1])
            ax.set_xticklabels([''])
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
        
        # Create output filename
        output_file = self.output_dir / "cross_validation_metrics_visualization.png"
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create per-fold comparison plot
        self.create_fold_comparison_plot()
        
        print(f"Metrics visualization saved: {output_file}")
        return True
    
    def create_fold_comparison_plot(self):
        """Create per-fold performance comparison plot"""
        if not self.results:
            return False
        
        df = pd.DataFrame(self.results)
        
        # Create fold comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cross-Validation Performance by Fold', fontsize=16, fontweight='bold')
        
        # Define metrics to plot
        metrics = [
            ('Dice_Score', 'Dice Score (Volume)', axes[0, 0]),
            ('Dice_Score_Slicewise', 'Dice Score (Slice-wise)', axes[0, 1]),
            ('HD95_mm', 'HD95 (mm)', axes[1, 0]),
            ('Sensitivity', 'Sensitivity', axes[1, 1])
        ]
        
        for metric_col, metric_title, ax in metrics:
            fold_data = []
            fold_labels = []
            
            for fold_num in range(5):
                fold_values = df[df['Fold'] == fold_num][metric_col]
                if metric_col == 'HD95_mm':
                    fold_values = fold_values.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(fold_values) > 0:
                    fold_data.append(fold_values.values)
                    fold_labels.append(f'Fold {fold_num}')
            
            if fold_data:
                # Create box plots
                box_parts = ax.boxplot(fold_data, tick_labels=fold_labels, patch_artist=True)
                
                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(fold_data)))
                for patch, color in zip(box_parts['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add overall mean line
                all_values = np.concatenate(fold_data)
                overall_mean = np.mean(all_values)
                ax.axhline(y=overall_mean, color='red', linestyle='--', 
                          linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.4f}')
                
                ax.set_title(metric_title, fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle=':')
                ax.legend(loc='best', fontsize=9)
        
        # Save fold comparison plot
        output_file = self.output_dir / "cross_validation_fold_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Fold comparison plot saved: {output_file}")
    
    def print_evaluation_summary(self):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION EVALUATION SUMMARY:")
        print("=" * 60)
        print(f"Total evaluations: {self.processed_count}")
        print(f"Successful evaluations: {self.successful_count}")
        print(f"Failed evaluations: {len(self.failed_files)}")
        if self.failed_files:
            print(f"Failed files: {', '.join(self.failed_files)}")
        print(f"Success rate: {self.successful_count/self.processed_count*100:.1f}%")
        
        if self.results:
            df = pd.DataFrame(self.results)
            
            # Calculate overall statistics
            dice_scores = df['Dice_Score'].values
            dice_scores_slice = df['Dice_Score_Slicewise'].values
            hd95_values = df['HD95_mm'].replace([np.inf, -np.inf], np.nan).dropna().values
            
            print(f"\nOVERALL PERFORMANCE SUMMARY:")
            print(f"Dice Score (Volume)   - Mean: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
            print(f"Dice Score (Slice)    - Mean: {np.mean(dice_scores_slice):.4f} ± {np.std(dice_scores_slice):.4f}")
            if len(hd95_values) > 0:
                print(f"HD95 (mm)             - Mean: {np.mean(hd95_values):.2f} ± {np.std(hd95_values):.2f}")
                print(f"HD95 Range            - Min: {np.min(hd95_values):.2f}, Max: {np.max(hd95_values):.2f}")
            
            # Per-fold statistics
            print(f"\nPER-FOLD PERFORMANCE:")
            for fold_num in range(5):
                fold_data = df[df['Fold'] == fold_num]
                if len(fold_data) > 0:
                    fold_dice = fold_data['Dice_Score'].mean()
                    print(f"Fold {fold_num}: {fold_dice:.4f} Dice ({len(fold_data)} cases)")
        
        print("=" * 60)


def main():
    """Main function to run evaluation"""
    # Default paths
    default_results_root = Path("/home/ubuntu/DLSegPerf/data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d")
    default_output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/evaluation_results")
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        results_root = Path(sys.argv[1])
    else:
        results_root = default_results_root
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = default_output_dir
    
    print("Cross-Validation nnUNet Evaluation Script")
    print("=" * 60)
    print(f"Results root: {results_root}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Initialize evaluator
    evaluator = CrossValidationEvaluator(results_root, output_dir)
    
    # Run evaluation
    success = evaluator.run_evaluation()
    
    if success:
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved in: {output_dir}")
    else:
        print("Evaluation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)