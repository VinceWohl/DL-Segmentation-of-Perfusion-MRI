#!/usr/bin/env python3
"""
Threshold Segmentation Evaluation Script

This script evaluates the performance of threshold-based segmentation by comparing
threshold masks with ground truth masks using standard metrics:

- Dice Score: Volume overlap coefficient
- Hausdorff Distance 95th percentile (HD95): Surface distance metric
- Basic cardinalities: True Positives, False Positives, False Negatives, True Negatives

Input folders:
- data/labels/ - Ground truth masks (PerfTerr###-v#-L/R.nii)
- data/thresholded_masks/ - Threshold masks (PerfTerr###-v#-L/R.nii)

Output: Excel file with comprehensive evaluation metrics

Author: Generated for threshold segmentation evaluation
"""

import os
import sys
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


class SegmentationEvaluator:
    def __init__(self, data_root, output_dir):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define folder paths
        self.labels_dir = self.data_root / "labels"
        self.threshold_dir = self.data_root / "thresholded_masks"
        
        # Results storage
        self.results = []
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
    
    def calculate_iou(self, gt_mask, pred_mask):
        """Calculate Intersection over Union (Jaccard Index)"""
        intersection = np.sum(gt_mask * pred_mask)
        union = np.sum((gt_mask | pred_mask).astype(np.uint8))
        
        if union == 0:
            return 1.0 if np.sum(gt_mask) == np.sum(pred_mask) else 0.0
        
        iou = intersection / union
        return iou
    
    def calculate_precision(self, tp, fp):
        """Calculate Precision (Positive Predictive Value)"""
        if (tp + fp) == 0:
            return 0.0
        return tp / (tp + fp)
    
    def calculate_relative_volume_error(self, gt_mask, pred_mask, spacing):
        """Calculate Relative Volume Error as percentage"""
        voxel_volume_mm3 = np.prod(spacing)
        gt_volume = np.sum(gt_mask) * voxel_volume_mm3
        pred_volume = np.sum(pred_mask) * voxel_volume_mm3
        
        if gt_volume == 0:
            return float('inf') if pred_volume > 0 else 0.0
        
        rve = (pred_volume - gt_volume) / gt_volume * 100.0
        return rve
    
    def calculate_assd(self, gt_mask, pred_mask, spacing):
        """Calculate Average Symmetric Surface Distance in millimeters"""
        try:
            # Get surface points for both masks
            gt_surface = self.get_surface_points(gt_mask, spacing)
            pred_surface = self.get_surface_points(pred_mask, spacing)
            
            # Handle edge cases
            if len(gt_surface) == 0 and len(pred_surface) == 0:
                return 0.0  # Both masks are empty
            elif len(gt_surface) == 0 or len(pred_surface) == 0:
                return float('inf')  # One mask is empty, other is not
            
            # Calculate mean distances from GT surface to predicted surface
            distances_gt_to_pred = cdist(gt_surface, pred_surface).min(axis=1)
            mean_gt_to_pred = np.mean(distances_gt_to_pred)
            
            # Calculate mean distances from predicted surface to GT surface  
            distances_pred_to_gt = cdist(pred_surface, gt_surface).min(axis=1)
            mean_pred_to_gt = np.mean(distances_pred_to_gt)
            
            # Average Symmetric Surface Distance
            assd = (mean_gt_to_pred + mean_pred_to_gt) / 2.0
            return float(assd)
            
        except Exception as e:
            print(f"Warning: ASSD calculation failed: {e}")
            return float('nan')
    
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
    
    def evaluate_segmentation(self, gt_file, pred_file, base_name):
        """Evaluate a single segmentation comparison"""
        print(f"Evaluating {base_name}...")
        
        # Load masks
        gt_mask, gt_spacing = self.load_nifti_file(gt_file)
        pred_mask, pred_spacing = self.load_nifti_file(pred_file)
        
        if gt_mask is None or pred_mask is None:
            print(f"  ERROR: Failed to load masks")
            return None
        
        # Use GT spacing (should be same for both)
        spacing = gt_spacing
        
        # Ensure shapes match
        if gt_mask.shape != pred_mask.shape:
            print(f"  ERROR: Shape mismatch - GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            return None
        
        # Calculate all metrics
        dice_score = self.calculate_dice_score(gt_mask, pred_mask)
        dice_score_slicewise = self.calculate_dice_score_slicewise(gt_mask, pred_mask)
        iou = self.calculate_iou(gt_mask, pred_mask)
        tp, fp, fn, tn = self.calculate_cardinalities(gt_mask, pred_mask)
        sensitivity, specificity = self.calculate_sensitivity_specificity(tp, fp, fn, tn)
        precision = self.calculate_precision(tp, fp)
        rve = self.calculate_relative_volume_error(gt_mask, pred_mask, spacing)
        hd95 = self.calculate_hausdorff_distance_95(gt_mask, pred_mask, spacing)
        assd = self.calculate_assd(gt_mask, pred_mask, spacing)
        
        print(f"  DSC: {dice_score:.4f}, IoU: {iou:.4f}, Sens: {sensitivity:.3f}, HD95: {hd95:.2f}")
        
        return {
            'Base_Name': base_name,
            'DSC_Volume': round(dice_score, 4),
            'DSC_Slicewise': round(dice_score_slicewise, 4) if not np.isnan(dice_score_slicewise) else np.nan,
            'IoU': round(iou, 4),
            'Sensitivity': round(sensitivity, 4),
            'Precision': round(precision, 4),
            'Specificity': round(specificity, 4),
            'RVE_Percent': round(rve, 4) if not np.isinf(rve) else rve,
            'HD95_mm': round(hd95, 4) if not (np.isnan(hd95) or np.isinf(hd95)) else hd95,
            'ASSD_mm': round(assd, 4) if not (np.isnan(assd) or np.isinf(assd)) else assd
        }
    
    def find_matching_files(self):
        """Find all matching GT and threshold mask file pairs"""
        matching_pairs = []
        
        if not self.labels_dir.exists() or not self.threshold_dir.exists():
            return matching_pairs
        
        # Get all GT files
        gt_files = list(self.labels_dir.glob("*.nii"))
        
        for gt_file in gt_files:
            base_name = gt_file.stem  # Remove .nii extension
            threshold_file = self.threshold_dir / f"{base_name}.nii"
            
            if threshold_file.exists():
                matching_pairs.append((gt_file, threshold_file, base_name))
            else:
                print(f"Warning: No matching threshold mask for {base_name}")
        
        return sorted(matching_pairs)
    
    def run_evaluation(self):
        """Run the complete segmentation evaluation"""
        print("Threshold Segmentation Evaluation")
        print("=" * 60)
        print(f"Data root: {self.data_root}")
        print(f"Ground truth labels: {self.labels_dir}")
        print(f"Threshold masks: {self.threshold_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        
        if not PANDAS_AVAILABLE:
            print("ERROR: pandas package is required. Please install with: pip install pandas openpyxl")
            return False
        
        # Check if required directories exist
        missing_dirs = []
        if not self.labels_dir.exists():
            missing_dirs.append(f"Labels: {self.labels_dir}")
        if not self.threshold_dir.exists():
            missing_dirs.append(f"Threshold masks: {self.threshold_dir}")
        
        if missing_dirs:
            print("ERROR: Missing required directories:")
            for missing in missing_dirs:
                print(f"  {missing}")
            return False
        
        # Find matching file pairs
        matching_pairs = self.find_matching_files()
        if not matching_pairs:
            print("ERROR: No matching GT and threshold mask pairs found")
            return False
        
        print(f"Found {len(matching_pairs)} matching file pairs")
        print()
        
        # Process each pair
        for gt_file, threshold_file, base_name in matching_pairs:
            self.processed_count += 1
            result = self.evaluate_segmentation(gt_file, threshold_file, base_name)
            
            if result is not None:
                self.results.append(result)
                self.successful_count += 1
            else:
                self.failed_files.append(base_name)
            print()
        
        # Save results to Excel
        self.save_results_to_excel()
        
        # Create visualization
        print("\nCreating metrics visualization...")
        self.create_metrics_visualization(self.output_dir)
        
        # Summary
        print("=" * 60)
        print("EVALUATION SUMMARY:")
        print(f"Total evaluations: {self.processed_count}")
        print(f"Successful evaluations: {self.successful_count}")
        print(f"Failed evaluations: {len(self.failed_files)}")
        if self.failed_files:
            print(f"Failed files: {', '.join(self.failed_files)}")
        print(f"Success rate: {self.successful_count/self.processed_count*100:.1f}%")
        
        if self.results:
            # Calculate summary statistics for key metrics
            dsc_vol = [r['DSC_Volume'] for r in self.results]
            dsc_slice = [r['DSC_Slicewise'] for r in self.results if not np.isnan(r['DSC_Slicewise'])]
            iou_values = [r['IoU'] for r in self.results]
            sensitivity_values = [r['Sensitivity'] for r in self.results]
            precision_values = [r['Precision'] for r in self.results]
            hd95_values = [r['HD95_mm'] for r in self.results if not np.isnan(r['HD95_mm']) and not np.isinf(r['HD95_mm'])]
            
            print(f"\nMETRIC SUMMARY:")
            print(f"DSC (Volume) - Mean: {np.mean(dsc_vol):.4f}, Std: {np.std(dsc_vol):.4f}")
            print(f"DSC (Slice) - Mean: {np.mean(dsc_slice):.4f}, Std: {np.std(dsc_slice):.4f}")
            print(f"IoU - Mean: {np.mean(iou_values):.4f}, Std: {np.std(iou_values):.4f}")
            print(f"Sensitivity - Mean: {np.mean(sensitivity_values):.4f}, Std: {np.std(sensitivity_values):.4f}")
            print(f"Precision - Mean: {np.mean(precision_values):.4f}, Std: {np.std(precision_values):.4f}")
            if hd95_values:
                print(f"HD95 - Mean: {np.mean(hd95_values):.2f}, Std: {np.std(hd95_values):.2f}")
        
        return True
    
    def save_results_to_excel(self):
        """Save evaluation results to Excel file"""
        if not self.results:
            print("No results to save")
            return
        
        try:
            # Create timestamp for output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = self.output_dir / f"threshold_evaluation_results_{timestamp}.xlsx"
            
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            # Parse subject information from base name
            df[['Subject', 'Visit', 'Hemisphere']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
            df['Subject'] = 'sub-p' + df['Subject'].str.zfill(3)
            df['Visit'] = 'v' + df['Visit']
            
            # Reorder columns for better readability
            column_order = [
                'Subject', 'Visit', 'Hemisphere', 'Base_Name',
                'DSC_Volume', 'DSC_Slicewise', 'IoU', 'Sensitivity', 'Precision', 'Specificity',
                'RVE_Percent', 'HD95_mm', 'ASSD_mm'
            ]
            df = df[column_order]
            
            # Define metrics for summaries (only the 9 specified metrics)
            metric_columns = ['DSC_Volume', 'DSC_Slicewise', 'IoU', 'Sensitivity', 'Precision', 
                            'Specificity', 'RVE_Percent', 'HD95_mm', 'ASSD_mm']
            
            # Create overall summary statistics
            overall_summary_stats = []
            for col in metric_columns:
                if col in df.columns:
                    values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) > 0:
                        overall_summary_stats.append({
                            'Metric': col,
                            'Mean': round(values.mean(), 4),
                            'Std': round(values.std(), 4)
                        })
            
            overall_summary_df = pd.DataFrame(overall_summary_stats)
            
            # Create per-hemisphere summary statistics
            hemisphere_summary_stats = []
            for hemisphere in ['L', 'R']:
                hemisphere_data = df[df['Hemisphere'] == hemisphere]
                if len(hemisphere_data) > 0:
                    for col in metric_columns:
                        if col in hemisphere_data.columns:
                            values = hemisphere_data[col].replace([np.inf, -np.inf], np.nan).dropna()
                            if len(values) > 0:
                                hemisphere_summary_stats.append({
                                    'Hemisphere': hemisphere,
                                    'Metric': col,
                                    'Mean': round(values.mean(), 4),
                                    'Std': round(values.std(), 4)
                                })
            
            hemisphere_summary_df = pd.DataFrame(hemisphere_summary_stats)
            
            # Write to Excel with the three specified sheets
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Per_Case_Results', index=False)
                hemisphere_summary_df.to_excel(writer, sheet_name='Per_Hemisphere_Summary', index=False)
                overall_summary_df.to_excel(writer, sheet_name='Overall_Summary', index=False)
                
                # Get the workbook and worksheet objects
                per_case_worksheet = writer.sheets['Per_Case_Results']
                hemisphere_worksheet = writer.sheets['Per_Hemisphere_Summary']
                overall_worksheet = writer.sheets['Overall_Summary']
                
                # Auto-adjust column widths for all sheets
                for worksheet in [per_case_worksheet, hemisphere_worksheet, overall_worksheet]:
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
    
    def create_metrics_visualization(self, output_dir):
        """Create violin and box plots for evaluation metrics"""
        if not self.results:
            print("No results available for visualization")
            return False
        
        if not VISUALIZATION_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Skipping visualization.")
            return False
        
        # Prepare data for plotting - focus on key metrics
        df = pd.DataFrame(self.results)
        
        metrics_data = {
            'DSC Volume': df['DSC_Volume'].values,
            'DSC Slicewise': df['DSC_Slicewise'].values,
            'IoU': df['IoU'].values,
            'Sensitivity': df['Sensitivity'].values,
            'Precision': df['Precision'].values,
            'HD95 (mm)': df['HD95_mm'].replace([np.inf, -np.inf], np.nan).dropna().values
        }
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with 2x3 grid for the 6 key metrics
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Threshold Segmentation Evaluation Metrics', fontsize=16, fontweight='bold')
        axes = axes.flatten()  # Flatten for easy indexing
        
        # Calculate shared y-axis limits for similarity metrics (0-1 range)
        similarity_metrics = ['DSC Volume', 'DSC Slicewise', 'IoU', 'Sensitivity', 'Precision']
        
        # Plot violin plots
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i >= len(axes):
                break
            ax = axes[i]
            
            # Create violin plot
            violin_parts = ax.violinplot([values], positions=[1], widths=0.6, showmeans=False, showmedians=False)
            
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
            ax.axhline(y=mean_val + std_val, color='orange', linestyle='--', linewidth=1.5, 
                      alpha=0.8, label=f'±1 Std: {std_val:.4f}', zorder=3)
            ax.axhline(y=mean_val - std_val, color='orange', linestyle='--', linewidth=1.5, 
                      alpha=0.8, zorder=3)
            
            # Find and plot outliers as dots
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            if len(outliers) > 0:
                # Add small jitter to x-position for better visibility
                x_positions = np.random.normal(1, 0.02, len(outliers))
                ax.scatter(x_positions, outliers, color='darkred', s=40, alpha=0.9, 
                          zorder=6, edgecolors='black', linewidth=0.5, label=f'Outliers: {len(outliers)}')
            
            ax.set_title(f'{metric_name.replace(chr(10), " ")}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11)
            ax.set_xticks([1])
            ax.set_xticklabels([''])
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
            
            # Set y-axis limits for similarity metrics (0-1 range)
            if metric_name in similarity_metrics:
                ax.set_ylim(-0.05, 1.05)
        
        # Hide unused subplot if any
        for i in range(len(metrics_data), len(axes)):
            axes[i].set_visible(False)
        
        # Create output filename
        output_file = output_dir / "threshold_segmentation_metrics_visualization.png"
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Metrics visualization saved: {output_file}")
        return True


def main():
    """Main function to run evaluation"""
    # Default paths
    script_dir = Path(__file__).parent
    default_data_root = script_dir.parent / "data"
    default_output_dir = script_dir
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = default_data_root
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output_dir
    
    print("Threshold Segmentation Evaluation Script")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = SegmentationEvaluator(data_root, output_dir)
    
    # Run evaluation
    success = evaluator.run_evaluation()
    
    if success:
        print(f"\nEvaluation completed!")
        print(f"Results saved in: {output_dir}")
    else:
        print("Evaluation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)