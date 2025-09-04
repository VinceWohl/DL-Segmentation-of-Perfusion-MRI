#!/usr/bin/env python3
"""
Cross-Validation nnUNet Results Evaluation Script - Multi-Class Version

This script evaluates nnUNet cross-validation results for multi-class segmentation by comparing predictions 
with ground truth masks using comprehensive metrics:

- Dice Score: Volume overlap coefficient  
- Dice Score (Slice-wise): Mean slice-wise Dice coefficient
- Hausdorff Distance 95th percentile (HD95): Surface distance metric
- Sensitivity/Specificity: Classification performance metrics
- Volume metrics: GT volume, predicted volume, volume difference
- Basic cardinalities: True Positives, False Positives, False Negatives, True Negatives

Multi-class segmentation classes:
- Class 0: Background
- Class 1: Perfusion Left
- Class 2: Perfusion Right  
- Class 3: Perfusion Overlap

Hemisphere evaluation:
- Left Hemisphere: Class 1 + Class 3 (perfusion_left + perfusion_overlap)
- Right Hemisphere: Class 2 + Class 3 (perfusion_right + perfusion_overlap)

Input folder structure:
- data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/
  - fold_0/validation/
  - fold_1/validation/  
  - fold_2/validation/
  - fold_3/validation/
  - fold_4/validation/

Ground truth reference:
- data/nnUNet_raw/Dataset001_PerfusionTerritories/labelsTr/

Output:
- Excel file with detailed results and summary statistics including:
  - Per-class metrics
  - Per-hemisphere metrics (combined classes)
  - Visualization plots for metrics distribution

Author: Generated for nnUNet multi-class cross-validation evaluation
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


class MultiClassCrossValidationEvaluator:
    def __init__(self, results_root, output_dir):
        self.results_root = Path(results_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        self.gt_dir = Path("/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories/labelsTr")
        
        # Multi-class configuration
        self.class_names = {
            0: 'Background',
            1: 'Perfusion_Left',
            2: 'Perfusion_Right', 
            3: 'Perfusion_Overlap'
        }
        
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
            # Ensure integer labels for multi-class
            data = data.astype(np.uint8)
            # Get voxel spacing in mm
            spacing = nii.header.get_zooms()[:3]  # x, y, z spacing
            return data, spacing
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_class_mask(self, multi_class_mask, class_id):
        """Extract binary mask for specific class"""
        return (multi_class_mask == class_id).astype(np.uint8)
    
    def combine_hemisphere_mask(self, multi_class_mask, hemisphere='left'):
        """Combine classes to create hemisphere mask"""
        if hemisphere.lower() == 'left':
            # Left hemisphere: Class 1 (perfusion_left) + Class 3 (perfusion_overlap)
            mask = ((multi_class_mask == 1) | (multi_class_mask == 3)).astype(np.uint8)
        elif hemisphere.lower() == 'right':
            # Right hemisphere: Class 2 (perfusion_right) + Class 3 (perfusion_overlap)
            mask = ((multi_class_mask == 2) | (multi_class_mask == 3)).astype(np.uint8)
        else:
            raise ValueError(f"Unknown hemisphere: {hemisphere}")
        return mask
    
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
        
        # Load multi-class masks
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
        
        # Initialize results for this case
        case_results = {
            'Fold': fold,
            'Base_Name': base_name,
        }
        
        # Calculate voxel volume
        voxel_volume_mm3 = np.prod(spacing)
        
        # Evaluate each class separately
        unique_classes = np.unique(np.concatenate([gt_mask.flatten(), pred_mask.flatten()]))
        unique_classes = unique_classes[unique_classes > 0]  # Skip background
        
        for class_id in unique_classes:
            if class_id in self.class_names:
                class_name = self.class_names[class_id]
                
                # Extract binary masks for this class
                gt_class_mask = self.extract_class_mask(gt_mask, class_id)
                pred_class_mask = self.extract_class_mask(pred_mask, class_id)
                
                # Calculate metrics for this class
                dice_score = self.calculate_dice_score(gt_class_mask, pred_class_mask)
                dice_score_slicewise = self.calculate_dice_score_slicewise(gt_class_mask, pred_class_mask)
                tp, fp, fn, tn = self.calculate_cardinalities(gt_class_mask, pred_class_mask)
                sensitivity, specificity = self.calculate_sensitivity_specificity(tp, fp, fn, tn)
                hd95 = self.calculate_hausdorff_distance_95(gt_class_mask, pred_class_mask, spacing)
                
                # Volume calculations
                gt_volume_voxels = np.sum(gt_class_mask)
                pred_volume_voxels = np.sum(pred_class_mask)
                gt_volume_mm3 = gt_volume_voxels * voxel_volume_mm3
                pred_volume_mm3 = pred_volume_voxels * voxel_volume_mm3
                volume_difference_mm3 = pred_volume_mm3 - gt_volume_mm3
                
                # Store class-specific results
                case_results.update({
                    f'{class_name}_Dice_Score': dice_score,
                    f'{class_name}_Dice_Score_Slicewise': dice_score_slicewise,
                    f'{class_name}_HD95_mm': hd95,
                    f'{class_name}_True_Positives': tp,
                    f'{class_name}_False_Positives': fp,
                    f'{class_name}_False_Negatives': fn,
                    f'{class_name}_True_Negatives': tn,
                    f'{class_name}_Sensitivity': sensitivity,
                    f'{class_name}_Specificity': specificity,
                    f'{class_name}_GT_Volume_mm3': gt_volume_mm3,
                    f'{class_name}_Pred_Volume_mm3': pred_volume_mm3,
                    f'{class_name}_Volume_Difference_mm3': volume_difference_mm3
                })
        
        # Calculate hemisphere metrics
        for hemisphere in ['left', 'right']:
            # Create combined hemisphere masks
            gt_hemisphere_mask = self.combine_hemisphere_mask(gt_mask, hemisphere)
            pred_hemisphere_mask = self.combine_hemisphere_mask(pred_mask, hemisphere)
            
            # Calculate hemisphere metrics
            dice_score = self.calculate_dice_score(gt_hemisphere_mask, pred_hemisphere_mask)
            dice_score_slicewise = self.calculate_dice_score_slicewise(gt_hemisphere_mask, pred_hemisphere_mask)
            tp, fp, fn, tn = self.calculate_cardinalities(gt_hemisphere_mask, pred_hemisphere_mask)
            sensitivity, specificity = self.calculate_sensitivity_specificity(tp, fp, fn, tn)
            hd95 = self.calculate_hausdorff_distance_95(gt_hemisphere_mask, pred_hemisphere_mask, spacing)
            
            # Volume calculations
            gt_volume_voxels = np.sum(gt_hemisphere_mask)
            pred_volume_voxels = np.sum(pred_hemisphere_mask)
            gt_volume_mm3 = gt_volume_voxels * voxel_volume_mm3
            pred_volume_mm3 = pred_volume_voxels * voxel_volume_mm3
            volume_difference_mm3 = pred_volume_mm3 - gt_volume_mm3
            
            hemisphere_name = f'Hemisphere_{hemisphere.title()}'
            
            # Store hemisphere-specific results
            case_results.update({
                f'{hemisphere_name}_Dice_Score': dice_score,
                f'{hemisphere_name}_Dice_Score_Slicewise': dice_score_slicewise,
                f'{hemisphere_name}_HD95_mm': hd95,
                f'{hemisphere_name}_True_Positives': tp,
                f'{hemisphere_name}_False_Positives': fp,
                f'{hemisphere_name}_False_Negatives': fn,
                f'{hemisphere_name}_True_Negatives': tn,
                f'{hemisphere_name}_Sensitivity': sensitivity,
                f'{hemisphere_name}_Specificity': specificity,
                f'{hemisphere_name}_GT_Volume_mm3': gt_volume_mm3,
                f'{hemisphere_name}_Pred_Volume_mm3': pred_volume_mm3,
                f'{hemisphere_name}_Volume_Difference_mm3': volume_difference_mm3
            })
        
        # Print summary
        dice_left = case_results.get('Hemisphere_Left_Dice_Score', 0)
        dice_right = case_results.get('Hemisphere_Right_Dice_Score', 0)
        print(f"    Left: {dice_left:.4f}, Right: {dice_right:.4f}")
        
        return case_results
    
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
                
                # Extract multi-class metrics from nnUNet summary
                fold_summary = {
                    'fold': fold_num,
                    'cases_count': len(summary['metric_per_case'])
                }
                
                # Add per-class metrics from nnUNet
                if 'mean' in summary:
                    for class_id, metrics in summary['mean'].items():
                        if class_id in ['1', '2', '3']:  # Multi-class IDs
                            class_name = self.class_names[int(class_id)]
                            fold_summary.update({
                                f'{class_name}_mean_dice': metrics['Dice'],
                                f'{class_name}_mean_iou': metrics['IoU'],
                                f'{class_name}_mean_tp': metrics['TP'],
                                f'{class_name}_mean_fp': metrics['FP'],
                                f'{class_name}_mean_fn': metrics['FN'],
                                f'{class_name}_mean_tn': metrics['TN']
                            })
                
                return fold_summary
            except Exception as e:
                print(f"Warning: Could not load summary for fold {fold_num}: {e}")
        return None
    
    def run_evaluation(self):
        """Run the complete cross-validation evaluation"""
        print("Multi-Class Cross-Validation nnUNet Results Evaluation")
        print("=" * 70)
        print(f"Results root: {self.results_root}")
        print(f"Ground truth: {self.gt_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Classes: {list(self.class_names.values())}")
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
                print(f"  Fold {fold_num}: {fold_summary['cases_count']} cases")
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
            excel_file = self.output_dir / f"multi_class_cross_validation_evaluation_results_{timestamp}.xlsx"
            
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            # Parse subject information from base name
            df[['Subject', 'Visit']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)')
            df['Subject'] = 'sub-p' + df['Subject'].str.zfill(3)
            df['Visit'] = 'v' + df['Visit']
            
            # Sort the DataFrame: Fold -> Subject -> Visit
            df = df.sort_values(['Fold', 'Subject', 'Visit']).reset_index(drop=True)
            
            # Prepare separate DataFrames for different analyses
            
            # 1. Per-class detailed results
            class_columns = []
            hemisphere_columns = []
            base_columns = ['Fold', 'Subject', 'Visit', 'Base_Name']
            
            for col in df.columns:
                if any(class_name in col for class_name in ['Perfusion_Left', 'Perfusion_Right', 'Perfusion_Overlap']):
                    class_columns.append(col)
                elif 'Hemisphere_' in col:
                    hemisphere_columns.append(col)
            
            # Create per-class DataFrame
            class_df = df[base_columns + class_columns].copy()
            
            # Create per-hemisphere DataFrame  
            hemisphere_df = df[base_columns + hemisphere_columns].copy()
            
            # 2. Per-class summary statistics
            class_summary_stats = []
            for class_name in ['Perfusion_Left', 'Perfusion_Right', 'Perfusion_Overlap']:
                dice_col = f'{class_name}_Dice_Score'
                if dice_col in df.columns:
                    values = df[dice_col].dropna()
                    if len(values) > 0:
                        class_summary_stats.append({
                            'Class': class_name,
                            'Cases': len(values),
                            'Mean_Dice': values.mean(),
                            'Std_Dice': values.std(),
                            'Min_Dice': values.min(),
                            'Max_Dice': values.max(),
                            'Median_Dice': values.median()
                        })
            
            class_summary_df = pd.DataFrame(class_summary_stats)
            
            # 3. Per-hemisphere summary statistics
            hemisphere_summary_stats = []
            for hemisphere in ['Left', 'Right']:
                dice_col = f'Hemisphere_{hemisphere}_Dice_Score'
                if dice_col in df.columns:
                    values = df[dice_col].dropna()
                    if len(values) > 0:
                        hemisphere_summary_stats.append({
                            'Hemisphere': hemisphere,
                            'Cases': len(values),
                            'Mean_Dice': values.mean(),
                            'Std_Dice': values.std(),
                            'Min_Dice': values.min(),
                            'Max_Dice': values.max(),
                            'Median_Dice': values.median(),
                            'Mean_HD95': df[f'Hemisphere_{hemisphere}_HD95_mm'].replace([np.inf, -np.inf], np.nan).mean(),
                            'Mean_Sensitivity': df[f'Hemisphere_{hemisphere}_Sensitivity'].mean(),
                            'Mean_Specificity': df[f'Hemisphere_{hemisphere}_Specificity'].mean()
                        })
            
            hemisphere_summary_df = pd.DataFrame(hemisphere_summary_stats)
            
            # 4. Per-fold summary statistics
            fold_summary_stats = []
            for fold_num in range(5):
                fold_data = df[df['Fold'] == fold_num]
                if len(fold_data) > 0:
                    fold_stats = {
                        'Fold': fold_num,
                        'Cases': len(fold_data)
                    }
                    
                    # Add hemisphere statistics for this fold
                    for hemisphere in ['Left', 'Right']:
                        dice_col = f'Hemisphere_{hemisphere}_Dice_Score'
                        if dice_col in fold_data.columns:
                            fold_stats[f'{hemisphere}_Mean_Dice'] = fold_data[dice_col].mean()
                            fold_stats[f'{hemisphere}_Std_Dice'] = fold_data[dice_col].std()
                    
                    fold_summary_stats.append(fold_stats)
            
            fold_summary_df = pd.DataFrame(fold_summary_stats)
            
            # 5. Overall summary statistics
            # Get all numeric columns for overall statistics
            numeric_columns = []
            for col in df.columns:
                if any(metric in col for metric in ['Dice_Score', 'HD95_mm', 'Sensitivity', 'Specificity', 
                                                   'True_Positives', 'False_Positives', 'False_Negatives', 'True_Negatives',
                                                   'GT_Volume_mm3', 'Pred_Volume_mm3', 'Volume_Difference_mm3']):
                    numeric_columns.append(col)
            
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
            
            # 6. nnUNet fold summaries if available
            nnunet_fold_df = None
            if self.fold_summaries:
                nnunet_fold_df = pd.DataFrame(self.fold_summaries)
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Per-class detailed results
                class_df.to_excel(writer, sheet_name='Per_Class_Details', index=False)
                
                # Per-hemisphere detailed results
                hemisphere_df.to_excel(writer, sheet_name='Per_Hemisphere_Details', index=False)
                
                # Per-class summary
                class_summary_df.to_excel(writer, sheet_name='Per_Class_Summary', index=False)
                
                # Per-hemisphere summary  
                hemisphere_df_summary = hemisphere_summary_df.copy()
                hemisphere_df_summary.to_excel(writer, sheet_name='Per_Hemisphere_Summary', index=False)
                
                # Per-fold summary
                fold_summary_df.to_excel(writer, sheet_name='Per_Fold_Summary', index=False)
                
                # Overall summary statistics
                overall_summary_df.to_excel(writer, sheet_name='Overall_Summary', index=False)
                
                # nnUNet fold summaries if available
                if nnunet_fold_df is not None:
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
        
        # Create figure with hemisphere metrics
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Multi-Class Cross-Validation nnUNet Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # Hemisphere Dice scores
        left_dice = df['Hemisphere_Left_Dice_Score'].dropna()
        right_dice = df['Hemisphere_Right_Dice_Score'].dropna()
        
        # Create subplots
        ax1 = plt.subplot(2, 2, 1)
        if len(left_dice) > 0 and len(right_dice) > 0:
            ax1.boxplot([left_dice, right_dice], labels=['Left Hemisphere', 'Right Hemisphere'])
            ax1.set_title('Hemisphere Dice Scores')
            ax1.set_ylabel('Dice Score')
            ax1.grid(True, alpha=0.3)
        
        # Per-class Dice scores
        ax2 = plt.subplot(2, 2, 2)
        class_dice_data = []
        class_labels = []
        for class_name in ['Perfusion_Left', 'Perfusion_Right', 'Perfusion_Overlap']:
            dice_col = f'{class_name}_Dice_Score'
            if dice_col in df.columns:
                values = df[dice_col].dropna()
                if len(values) > 0:
                    class_dice_data.append(values)
                    class_labels.append(class_name.replace('Perfusion_', ''))
        
        if class_dice_data:
            ax2.boxplot(class_dice_data, labels=class_labels)
            ax2.set_title('Per-Class Dice Scores')
            ax2.set_ylabel('Dice Score')
            ax2.grid(True, alpha=0.3)
        
        # Hemisphere HD95
        ax3 = plt.subplot(2, 2, 3)
        left_hd95 = df['Hemisphere_Left_HD95_mm'].replace([np.inf, -np.inf], np.nan).dropna()
        right_hd95 = df['Hemisphere_Right_HD95_mm'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(left_hd95) > 0 and len(right_hd95) > 0:
            ax3.boxplot([left_hd95, right_hd95], labels=['Left Hemisphere', 'Right Hemisphere'])
            ax3.set_title('Hemisphere HD95 (mm)')
            ax3.set_ylabel('HD95 (mm)')
            ax3.grid(True, alpha=0.3)
        
        # Fold-wise performance
        ax4 = plt.subplot(2, 2, 4)
        fold_left_means = []
        fold_right_means = []
        fold_labels = []
        
        for fold_num in range(5):
            fold_data = df[df['Fold'] == fold_num]
            if len(fold_data) > 0:
                left_mean = fold_data['Hemisphere_Left_Dice_Score'].mean()
                right_mean = fold_data['Hemisphere_Right_Dice_Score'].mean()
                fold_left_means.append(left_mean)
                fold_right_means.append(right_mean)
                fold_labels.append(f'Fold {fold_num}')
        
        if fold_left_means and fold_right_means:
            x = np.arange(len(fold_labels))
            width = 0.35
            ax4.bar(x - width/2, fold_left_means, width, label='Left Hemisphere', alpha=0.8)
            ax4.bar(x + width/2, fold_right_means, width, label='Right Hemisphere', alpha=0.8)
            ax4.set_xlabel('Fold')
            ax4.set_ylabel('Mean Dice Score')
            ax4.set_title('Fold-wise Hemisphere Performance')
            ax4.set_xticks(x)
            ax4.set_xticklabels(fold_labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Save plot
        output_file = self.output_dir / "multi_class_cross_validation_metrics_visualization.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Metrics visualization saved: {output_file}")
        return True
    
    def print_evaluation_summary(self):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 70)
        print("MULTI-CLASS CROSS-VALIDATION EVALUATION SUMMARY:")
        print("=" * 70)
        print(f"Total evaluations: {self.processed_count}")
        print(f"Successful evaluations: {self.successful_count}")
        print(f"Failed evaluations: {len(self.failed_files)}")
        if self.failed_files:
            print(f"Failed files: {', '.join(self.failed_files)}")
        print(f"Success rate: {self.successful_count/self.processed_count*100:.1f}%")
        
        if self.results:
            df = pd.DataFrame(self.results)
            
            # Hemisphere performance summary
            left_dice = df['Hemisphere_Left_Dice_Score'].dropna()
            right_dice = df['Hemisphere_Right_Dice_Score'].dropna()
            
            print(f"\nHEMISPHERE PERFORMANCE SUMMARY:")
            if len(left_dice) > 0:
                print(f"Left Hemisphere       - Mean: {left_dice.mean():.4f} +/- {left_dice.std():.4f}")
            if len(right_dice) > 0:
                print(f"Right Hemisphere      - Mean: {right_dice.mean():.4f} +/- {right_dice.std():.4f}")
            
            # Per-class performance summary
            print(f"\nPER-CLASS PERFORMANCE SUMMARY:")
            for class_name in ['Perfusion_Left', 'Perfusion_Right', 'Perfusion_Overlap']:
                dice_col = f'{class_name}_Dice_Score'
                if dice_col in df.columns:
                    values = df[dice_col].dropna()
                    if len(values) > 0:
                        print(f"{class_name:<17} - Mean: {values.mean():.4f} +/- {values.std():.4f}")
            
            # Per-fold performance
            print(f"\nPER-FOLD PERFORMANCE (HEMISPHERE DICE):")
            for fold_num in range(5):
                fold_data = df[df['Fold'] == fold_num]
                if len(fold_data) > 0:
                    left_mean = fold_data['Hemisphere_Left_Dice_Score'].mean()
                    right_mean = fold_data['Hemisphere_Right_Dice_Score'].mean()
                    print(f"Fold {fold_num}: Left {left_mean:.4f}, Right {right_mean:.4f} ({len(fold_data)} cases)")
        
        print("=" * 70)


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
    
    print("Multi-Class Cross-Validation nnUNet Evaluation Script")
    print("=" * 70)
    print(f"Results root: {results_root}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Initialize evaluator
    evaluator = MultiClassCrossValidationEvaluator(results_root, output_dir)
    
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