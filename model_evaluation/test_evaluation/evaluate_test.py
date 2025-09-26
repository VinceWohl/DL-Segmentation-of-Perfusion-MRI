#!/usr/bin/env python3
"""
Test Set nnUNet Results Evaluation Script
Evaluates predictions on test set and creates Excel aligned with cross-validation evaluation:
  - Sheets: Per_Case_Results, Per_Hemisphere_Summary, Overall_Summary
  - Metrics: DSC_Volume, DSC_Slicewise, IoU, Sensitivity, Precision,
             Specificity, RVE_Percent, HD95_mm, ASSD_mm
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
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class TestSetEvaluator:
    def __init__(self, predictions_dirs, gt_dir, output_dir):
        # Handle both single path (backward compatibility) and multiple paths
        if isinstance(predictions_dirs, (str, Path)):
            # Single prediction directory - backward compatibility
            self.predictions_dirs = {'single': Path(predictions_dirs)}
        else:
            # Multiple prediction directories
            self.predictions_dirs = {name: Path(path) for name, path in predictions_dirs.items()}

        self.gt_dir = Path(gt_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage - now includes input type
        self.results = []
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []

        # Store slice-wise data for plotting
        self.slice_wise_data = []

    # ---------- Loading & basic metrics ----------
    def load_nifti_file(self, file_path):
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            mask = (data > 0).astype(np.uint8)
            spacing = nii.header.get_zooms()[:3]
            return mask, spacing
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def calculate_dice_score(self, gt_mask, pred_mask):
        intersection = np.sum(gt_mask * pred_mask)
        union = np.sum(gt_mask) + np.sum(pred_mask)
        if union == 0:
            return 1.0 if np.sum(gt_mask) == np.sum(pred_mask) else 0.0
        return (2.0 * intersection) / union

    def calculate_dice_score_slicewise(self, gt_mask, pred_mask):
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
        tp = np.sum((gt_mask == 1) & (pred_mask == 1))
        fp = np.sum((gt_mask == 0) & (pred_mask == 1))
        fn = np.sum((gt_mask == 1) & (pred_mask == 0))
        tn = np.sum((gt_mask == 0) & (pred_mask == 0))
        return int(tp), int(fp), int(fn), int(tn)

    def get_surface_points(self, mask, spacing):
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure)
        boundary = mask & ~eroded
        coords = np.array(np.where(boundary)).T
        if len(coords) > 0:
            return coords * np.array(spacing)
        return coords

    def calculate_hausdorff_distance_95(self, gt_mask, pred_mask, spacing):
        try:
            gt_surface = self.get_surface_points(gt_mask, spacing)
            pred_surface = self.get_surface_points(pred_mask, spacing)
            if len(gt_surface) == 0 and len(pred_surface) == 0:
                return 0.0
            elif len(gt_surface) == 0 or len(pred_surface) == 0:
                return float('inf')
            d1 = cdist(gt_surface, pred_surface).min(axis=1)
            d2 = cdist(pred_surface, gt_surface).min(axis=1)
            all_d = np.concatenate([d1, d2])
            return float(np.percentile(all_d, 95))
        except Exception as e:
            print(f"Warning: HD95 calculation failed: {e}")
            return float('nan')

    def calculate_iou(self, gt_mask, pred_mask):
        intersection = np.sum(gt_mask * pred_mask)
        union = np.sum((gt_mask | pred_mask).astype(np.uint8))
        if union == 0:
            return 1.0 if np.sum(gt_mask) == np.sum(pred_mask) else 0.0
        return intersection / union

    def calculate_precision(self, tp, fp):
        return 0.0 if (tp + fp) == 0 else tp / (tp + fp)

    def calculate_relative_volume_error(self, gt_mask, pred_mask, spacing):
        voxel_vol = np.prod(spacing)
        gt_vol = np.sum(gt_mask) * voxel_vol
        pred_vol = np.sum(pred_mask) * voxel_vol
        if gt_vol == 0:
            return float('inf') if pred_vol > 0 else 0.0
        return (pred_vol - gt_vol) / gt_vol * 100.0

    def calculate_assd(self, gt_mask, pred_mask, spacing):
        try:
            gt_surface = self.get_surface_points(gt_mask, spacing)
            pred_surface = self.get_surface_points(pred_mask, spacing)
            if len(gt_surface) == 0 and len(pred_surface) == 0:
                return 0.0
            elif len(gt_surface) == 0 or len(pred_surface) == 0:
                return float('inf')
            d1 = cdist(gt_surface, pred_surface).min(axis=1).mean()
            d2 = cdist(pred_surface, gt_surface).min(axis=1).mean()
            return float((d1 + d2) / 2.0)
        except Exception as e:
            print(f"Warning: ASSD calculation failed: {e}")
            return float('nan')

    def calculate_slice_wise_assd(self, gt_mask, pred_mask, spacing, base_name):
        """Calculate ASSD for each slice and store for plotting"""
        slice_data = []
        nz = gt_mask.shape[2]

        # Extract group from base_name and handle input type suffix
        case_base = base_name.split('_')[0] if '_' in base_name else base_name
        subject_num = int(case_base.split('-')[0].replace('PerfTerr', ''))
        group = 'HC' if subject_num in [14, 15] else 'Patients'

        for z in range(nz):
            gt_slice = gt_mask[:, :, z]
            pred_slice = pred_mask[:, :, z]

            # Skip empty slices
            if np.sum(gt_slice) == 0 and np.sum(pred_slice) == 0:
                continue

            try:
                # Get 2D surface points for this slice
                gt_surface_2d = self.get_surface_points_2d(gt_slice, spacing[:2])
                pred_surface_2d = self.get_surface_points_2d(pred_slice, spacing[:2])

                if len(gt_surface_2d) == 0 and len(pred_surface_2d) == 0:
                    assd_slice = 0.0
                elif len(gt_surface_2d) == 0 or len(pred_surface_2d) == 0:
                    assd_slice = float('inf')
                else:
                    d1 = cdist(gt_surface_2d, pred_surface_2d).min(axis=1).mean()
                    d2 = cdist(pred_surface_2d, gt_surface_2d).min(axis=1).mean()
                    assd_slice = float((d1 + d2) / 2.0)

                if not np.isinf(assd_slice) and not np.isnan(assd_slice):
                    slice_data.append({
                        'Case': base_name,
                        'Group': group,
                        'Slice': z,
                        'ASSD_mm': assd_slice
                    })

            except Exception as e:
                continue  # Skip problematic slices

        return slice_data

    def get_surface_points_2d(self, mask_2d, spacing_2d):
        """Get surface points for 2D slice"""
        structure = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask_2d, structure)
        boundary = mask_2d & ~eroded
        coords = np.array(np.where(boundary)).T
        if len(coords) > 0:
            return coords * np.array(spacing_2d)
        return coords

    def convert_multilabel_to_binary(self, multilabel_mask, hemisphere):
        """Convert multi-label ground truth to binary mask for specific hemisphere"""
        if hemisphere == 'Left':
            return (multilabel_mask == 1).astype(np.uint8)
        elif hemisphere == 'Right':
            return (multilabel_mask == 2).astype(np.uint8)
        else:
            raise ValueError(f"Invalid hemisphere: {hemisphere}")

    def extract_hemisphere_from_prediction(self, pred_mask, hemisphere):
        """Extract hemisphere from 4D prediction (H, W, D, C) or handle 3D"""
        if len(pred_mask.shape) == 4:
            # 4D prediction: (H, W, D, Channels)
            if hemisphere == 'Left':
                return pred_mask[:, :, :, 0]
            elif hemisphere == 'Right':
                return pred_mask[:, :, :, 1]
        elif len(pred_mask.shape) == 3:
            # 3D prediction - assume single hemisphere based on filename
            return pred_mask
        else:
            raise ValueError(f"Unexpected prediction shape: {pred_mask.shape}")

    def evaluate_single_case(self, pred_file, gt_file, base_name, input_type):
        print(f"  Evaluating {base_name} ({input_type})...")

        # Load prediction and ground truth (both are already binary for the specific hemisphere)
        pred_mask, _ = self.load_nifti_file(pred_file)
        gt_mask, gt_spacing = self.load_nifti_file(gt_file)

        if gt_mask is None or pred_mask is None:
            print(f"    ERROR: Failed to load files")
            return None

        spacing = gt_spacing

        if gt_mask.shape != pred_mask.shape:
            print(f"    ERROR: Shape mismatch - GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            return None

        # Calculate all metrics
        dice_score = self.calculate_dice_score(gt_mask, pred_mask)
        dice_slice = self.calculate_dice_score_slicewise(gt_mask, pred_mask)
        tp, fp, fn, tn = self.calculate_cardinalities(gt_mask, pred_mask)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        hd95 = self.calculate_hausdorff_distance_95(gt_mask, pred_mask, spacing)
        iou = self.calculate_iou(gt_mask, pred_mask)
        precision = self.calculate_precision(tp, fp)
        rve = self.calculate_relative_volume_error(gt_mask, pred_mask, spacing)
        assd = self.calculate_assd(gt_mask, pred_mask, spacing)

        # Calculate slice-wise ASSD for plotting
        slice_assd_data = self.calculate_slice_wise_assd(gt_mask, pred_mask, spacing, base_name)
        self.slice_wise_data.extend(slice_assd_data)

        # Extract hemisphere from filename
        hemisphere = 'Left' if base_name.endswith('-L') else 'Right'

        print(f"    Dice: {dice_score:.4f}, Dice(slice): {dice_slice:.4f}, IoU: {iou:.4f}, HD95: {hd95:.2f}")

        return {
            'Base_Name': base_name,
            'Input_Type': input_type,
            'Hemisphere': hemisphere,
            'DSC_Volume': dice_score,
            'DSC_Slicewise': dice_slice,
            'IoU': iou,
            'Sensitivity': sensitivity,
            'Precision': precision,
            'Specificity': specificity,
            'RVE_Percent': rve,
            'HD95_mm': hd95,
            'ASSD_mm': assd
        }

    def find_test_cases(self):
        """Find matching prediction and ground truth files for all input types"""
        all_test_cases = []

        for input_type, predictions_dir in self.predictions_dirs.items():
            pred_files = list(predictions_dir.glob("*-[LR].nii*"))

            for pred_file in pred_files:
                # Extract base name (e.g., "PerfTerr014-v1-L" from "PerfTerr014-v1-L.nii")
                base_name = pred_file.stem
                if pred_file.suffix == '.gz':
                    base_name = pred_file.with_suffix('').stem

                # Find corresponding ground truth file
                gt_file = self.gt_dir / f"{base_name}.nii"
                if not gt_file.exists():
                    gt_file = self.gt_dir / f"{base_name}.nii.gz"

                if gt_file.exists():
                    all_test_cases.append((pred_file, gt_file, base_name, input_type))
                else:
                    print(f"Warning: No matching GT file for {base_name} ({input_type})")

        return all_test_cases

    def run_evaluation(self):
        print("Test Set nnUNet Results Evaluation - Multi-Input Comparison")
        print("=" * 70)
        print("Prediction directories:")
        for input_type, pred_dir in self.predictions_dirs.items():
            print(f"  {input_type}: {pred_dir}")
        print(f"Ground truth: {self.gt_dir}")
        print(f"Output directory: {self.output_dir}\n")

        # Check dependencies
        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        if not PANDAS_AVAILABLE:
            print("ERROR: pandas package is required. Please install with: pip install pandas openpyxl")
            return False
        if not PLOTTING_AVAILABLE:
            print("WARNING: matplotlib/seaborn not available. Plots will be skipped. Install with: pip install matplotlib seaborn")

        # Check all prediction directories exist
        for input_type, pred_dir in self.predictions_dirs.items():
            if not pred_dir.exists():
                print(f"ERROR: Predictions directory not found ({input_type}): {pred_dir}")
                return False

        if not self.gt_dir.exists():
            print(f"ERROR: Ground truth directory not found: {self.gt_dir}")
            return False

        # Find test cases
        test_cases = self.find_test_cases()
        if not test_cases:
            print("ERROR: No test cases found")
            return False

        print(f"Found {len(test_cases)} test cases across {len(self.predictions_dirs)} input types")

        # Evaluate each case
        for pred_file, gt_file, base_name, input_type in test_cases:
            print(f"\nProcessing {base_name} ({input_type}):")

            self.processed_count += 1
            result = self.evaluate_single_case(pred_file, gt_file, base_name, input_type)
            if result is not None:
                self.results.append(result)
                self.successful_count += 1
            else:
                self.failed_files.append(f"{base_name} ({input_type})")

        print("\nSaving results to Excel...")
        self.save_results_to_excel()

        print("\nCreating box plots...")
        self.create_box_plots()

        self.print_evaluation_summary()
        return True

    def create_excel_for_group(self, df, group_name, timestamp):
        """Create Excel file for a specific group (HC or Patients)"""
        excel_file = self.output_dir / f"test_set_results_{group_name}_{timestamp}.xlsx"

        # ----- Sheet 1: Per_Case_Results -----
        # Include Input_Type only if it exists and has multiple values
        base_cols = ['Subject', 'Visit', 'Hemisphere', 'Base_Name']
        metric_cols = ['DSC_Volume', 'DSC_Slicewise', 'IoU',
                      'Sensitivity', 'Precision', 'Specificity',
                      'RVE_Percent', 'HD95_mm', 'ASSD_mm']

        if 'Input_Type' in df.columns and len(df['Input_Type'].unique()) > 1:
            per_case_cols = base_cols[:2] + ['Input_Type'] + base_cols[2:] + metric_cols
        else:
            per_case_cols = base_cols + metric_cols

        per_case_df = df[per_case_cols].copy()

        # ----- Sheet 2: Per_Hemisphere_Summary -----
        metric_cols = ['DSC_Volume', 'DSC_Slicewise', 'IoU',
                       'Sensitivity', 'Precision', 'Specificity',
                       'RVE_Percent', 'HD95_mm', 'ASSD_mm']

        hemi_rows = []
        for hemi in ['Left', 'Right']:
            sub = per_case_df[per_case_df['Hemisphere'] == hemi]
            if len(sub) == 0:
                continue
            for col in metric_cols:
                vals = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals) == 0:
                    continue
                hemi_rows.append({
                    'Hemisphere': hemi,
                    'Metric': col,
                    'Mean': round(vals.mean(), 4),
                    'Std': round(vals.std(), 4)
                })
        hemi_df = pd.DataFrame(hemi_rows)

        # ----- Sheet 3: Overall_Summary -----
        overall_rows = []
        for col in metric_cols:
            vals = per_case_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                continue
            overall_rows.append({
                'Metric': col,
                'Mean': round(vals.mean(), 4),
                'Std': round(vals.std(), 4)
            })
        overall_df = pd.DataFrame(overall_rows)

        # Write to Excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            per_case_df.to_excel(writer, sheet_name='Per_Case_Results', index=False)
            hemi_df.to_excel(writer, sheet_name='Per_Hemisphere_Summary', index=False)
            overall_df.to_excel(writer, sheet_name='Overall_Summary', index=False)

            # Auto-adjust column widths
            for ws in writer.sheets.values():
                for column in ws.columns:
                    max_len = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            max_len = max(max_len, len(str(cell.value)))
                        except Exception:
                            pass
                    ws.column_dimensions[column[0].column_letter].width = min(max_len + 2, 50)

        print(f"Results saved to Excel ({group_name}): {excel_file}")
        return excel_file

    def save_results_to_excel(self):
        if not self.results:
            print("No results to save")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            df = pd.DataFrame(self.results)

            # Parse subject and visit information
            df[['Subject', 'Visit', 'Hemisphere_Code']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
            df['Subject'] = 'sub-p' + df['Subject'].str.zfill(3)
            df['Visit'] = 'v' + df['Visit']

            # Sort by Subject, Visit, Hemisphere
            df = df.sort_values(['Subject', 'Visit', 'Hemisphere'], na_position='last')

            # Extract subject number for grouping
            df['Subject_Num'] = df['Subject'].str.extract(r'sub-p(\d+)').astype(int)

            # Separate into healthy controls (PerfTerr014, PerfTerr015) and patients (PerfTerr017-023)
            hc_df = df[df['Subject_Num'].isin([14, 15])].copy()
            patients_df = df[df['Subject_Num'].isin([17, 18, 19, 20, 22, 23])].copy()

            # Remove the temporary Subject_Num column
            hc_df = hc_df.drop('Subject_Num', axis=1)
            patients_df = patients_df.drop('Subject_Num', axis=1)

            print(f"Healthy Controls: {len(hc_df)} cases")
            print(f"Patients: {len(patients_df)} cases")

            # Create separate Excel files for each group
            excel_files = []
            if len(hc_df) > 0:
                excel_file_hc = self.create_excel_for_group(hc_df, 'HC', timestamp)
                excel_files.append(excel_file_hc)

            if len(patients_df) > 0:
                excel_file_patients = self.create_excel_for_group(patients_df, 'patients', timestamp)
                excel_files.append(excel_file_patients)

            # Also create a combined file for comparison
            excel_file_combined = self.output_dir / f"test_set_results_combined_{timestamp}.xlsx"
            self.create_excel_for_group(df.drop('Subject_Num', axis=1, errors='ignore'), 'combined', timestamp)

            return excel_files

        except Exception as e:
            print(f"Error saving Excel file: {e}")
            return []

    def create_box_plots(self):
        """Create box plots comparing input types (CBF-only vs CBF+T1w) within HC and Patients groups for DSC and slice-wise ASSD"""
        if not PLOTTING_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Skipping box plots.")
            return

        if not self.results:
            print("No results available for plotting")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare data for volume-based DSC plot
            df = pd.DataFrame(self.results)
            df[['Subject', 'Visit', 'Hemisphere_Code']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
            df['Subject_Num'] = df['Subject'].astype(int)
            df['Group'] = df['Subject_Num'].apply(lambda x: 'HC' if x in [14, 15] else 'Patients')

            # Filter out invalid values
            df_plot = df[df['DSC_Volume'].replace([np.inf, -np.inf], np.nan).notna()].copy()

            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # Check if we have multiple input types
            if 'Input_Type' in df_plot.columns and len(df_plot['Input_Type'].unique()) > 1:
                # Create input type comparison plots
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                fig.suptitle('DSC Comparison by Input Type: CBF-only vs CBF+T1w\nTest Set Evaluation Results',
                            fontsize=16, fontweight='bold')

                input_types = sorted(df_plot['Input_Type'].unique())
                colors = ['#1f77b4', '#ff7f0e']  # Blue for first input type, Orange for second

                # Plot for HC group
                ax = axes[0]
                hc_data = df_plot[df_plot['Group'] == 'HC']
                if len(hc_data) > 0:
                    box_data = [hc_data[hc_data['Input_Type'] == it]['DSC_Volume'].values for it in input_types]
                    box_parts = ax.boxplot(box_data, positions=range(len(input_types)),
                                          notch=True, patch_artist=True, widths=0.5)

                    for patch, color in zip(box_parts['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1)

                    ax.set_title('Healthy Controls (HC)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Input Type', fontsize=12)
                    ax.set_ylabel('DSC (volume-based)', fontsize=12)
                    ax.set_xticks(range(len(input_types)))
                    ax.set_xticklabels(input_types, rotation=0)
                    ax.grid(True, alpha=0.6)

                    # Add median annotations
                    for i, input_type in enumerate(input_types):
                        data = hc_data[hc_data['Input_Type'] == input_type]['DSC_Volume']
                        if len(data) > 0:
                            median = data.median()
                            ax.text(i, ax.get_ylim()[1] * 0.95, f'{median:.4f}',
                                   ha='center', va='top', fontsize=10, weight='bold')

                # Plot for Patients group
                ax = axes[1]
                pat_data = df_plot[df_plot['Group'] == 'Patients']
                if len(pat_data) > 0:
                    box_data = [pat_data[pat_data['Input_Type'] == it]['DSC_Volume'].values for it in input_types]
                    box_parts = ax.boxplot(box_data, positions=range(len(input_types)),
                                          notch=True, patch_artist=True, widths=0.5)

                    for patch, color in zip(box_parts['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1)

                    ax.set_title('Patients', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Input Type', fontsize=12)
                    ax.set_ylabel('DSC (volume-based)', fontsize=12)
                    ax.set_xticks(range(len(input_types)))
                    ax.set_xticklabels(input_types, rotation=0)
                    ax.grid(True, alpha=0.6)

                    # Add median annotations
                    for i, input_type in enumerate(input_types):
                        data = pat_data[pat_data['Input_Type'] == input_type]['DSC_Volume']
                        if len(data) > 0:
                            median = data.median()
                            ax.text(i, ax.get_ylim()[1] * 0.95, f'{median:.4f}',
                                   ha='center', va='top', fontsize=10, weight='bold')

                plt.tight_layout()
                dsc_plot_file = self.output_dir / f"DSC_comparison_InputTypes_by_Groups_{timestamp}.png"
                plt.savefig(dsc_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"DSC input type comparison plot saved: {dsc_plot_file}")

            else:
                # Fallback to standard HC vs Patients comparison
                fig, ax = plt.subplots(figsize=(14, 8))

                box_parts = ax.boxplot(
                    [df_plot[df_plot['Group'] == 'HC']['DSC_Volume'].values,
                     df_plot[df_plot['Group'] == 'Patients']['DSC_Volume'].values],
                    positions=[0, 1], notch=True, patch_artist=True, widths=0.5)

                colors = ['#1f77b4', '#ff7f0e']
                for patch, color in zip(box_parts['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)

                ax.set_title('Volume-based Dice Score Comparison: HC vs Patients\nTest Set Evaluation Results',
                            fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Group', fontsize=14, fontweight='bold')
                ax.set_ylabel('DSC (volume-based)', fontsize=14, fontweight='bold')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['HC', 'Patients'])
                ax.grid(True, alpha=0.6)

                plt.tight_layout()
                dsc_plot_file = self.output_dir / f"DSC_comparison_HC_vs_Patients_{timestamp}.png"
                plt.savefig(dsc_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"DSC box plot saved: {dsc_plot_file}")

            # Plot 2: Slice-wise ASSD comparison
            if self.slice_wise_data:
                slice_df = pd.DataFrame(self.slice_wise_data)
                slice_df_plot = slice_df[slice_df['ASSD_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

                if len(slice_df_plot) > 0:
                    # Extract input type information from case names
                    slice_df_plot['Input_Type'] = slice_df_plot['Case'].str.extract(r'.*_(CBF_\w+)$')[0].fillna('Unknown')

                    if 'Input_Type' in slice_df_plot.columns and len(slice_df_plot['Input_Type'].unique()) > 1:
                        # Create input type comparison for slice-wise ASSD
                        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                        fig.suptitle('Slice-wise ASSD Comparison by Input Type: CBF-only vs CBF+T1w\nTest Set Evaluation Results',
                                    fontsize=16, fontweight='bold')

                        input_types = sorted(slice_df_plot['Input_Type'].unique())
                        colors = ['#1f77b4', '#ff7f0e']

                        # Plot for HC group
                        ax = axes[0]
                        hc_slice_data = slice_df_plot[slice_df_plot['Group'] == 'HC']
                        if len(hc_slice_data) > 0:
                            box_data = [hc_slice_data[hc_slice_data['Input_Type'] == it]['ASSD_mm'].values for it in input_types]
                            box_parts = ax.boxplot(box_data, positions=range(len(input_types)),
                                                  notch=True, patch_artist=True, widths=0.5)

                            for patch, color in zip(box_parts['boxes'], colors):
                                patch.set_facecolor(color)
                                patch.set_alpha(0.7)
                                patch.set_edgecolor('black')
                                patch.set_linewidth(1)

                            ax.set_title('Healthy Controls (HC)', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Input Type', fontsize=12)
                            ax.set_ylabel('ASSD (mm) - per slice', fontsize=12)
                            ax.set_xticks(range(len(input_types)))
                            ax.set_xticklabels(input_types, rotation=0)
                            ax.grid(True, alpha=0.6)

                            # Add median annotations
                            for i, input_type in enumerate(input_types):
                                data = hc_slice_data[hc_slice_data['Input_Type'] == input_type]['ASSD_mm']
                                if len(data) > 0:
                                    median = data.median()
                                    ax.text(i, ax.get_ylim()[1] * 0.95, f'{median:.3f}',
                                           ha='center', va='top', fontsize=10, weight='bold')

                        # Plot for Patients group
                        ax = axes[1]
                        pat_slice_data = slice_df_plot[slice_df_plot['Group'] == 'Patients']
                        if len(pat_slice_data) > 0:
                            box_data = [pat_slice_data[pat_slice_data['Input_Type'] == it]['ASSD_mm'].values for it in input_types]
                            box_parts = ax.boxplot(box_data, positions=range(len(input_types)),
                                                  notch=True, patch_artist=True, widths=0.5)

                            for patch, color in zip(box_parts['boxes'], colors):
                                patch.set_facecolor(color)
                                patch.set_alpha(0.7)
                                patch.set_edgecolor('black')
                                patch.set_linewidth(1)

                            ax.set_title('Patients', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Input Type', fontsize=12)
                            ax.set_ylabel('ASSD (mm) - per slice', fontsize=12)
                            ax.set_xticks(range(len(input_types)))
                            ax.set_xticklabels(input_types, rotation=0)
                            ax.grid(True, alpha=0.6)

                            # Add median annotations
                            for i, input_type in enumerate(input_types):
                                data = pat_slice_data[pat_slice_data['Input_Type'] == input_type]['ASSD_mm']
                                if len(data) > 0:
                                    median = data.median()
                                    ax.text(i, ax.get_ylim()[1] * 0.95, f'{median:.3f}',
                                           ha='center', va='top', fontsize=10, weight='bold')

                        plt.tight_layout()
                        assd_plot_file = self.output_dir / f"ASSD_slicewise_comparison_InputTypes_by_Groups_{timestamp}.png"
                        plt.savefig(assd_plot_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Slice-wise ASSD input type comparison plot saved: {assd_plot_file}")

                    else:
                        # Fallback to standard HC vs Patients comparison
                        fig, ax = plt.subplots(figsize=(14, 8))

                        box_parts = ax.boxplot(
                            [slice_df_plot[slice_df_plot['Group'] == 'HC']['ASSD_mm'].values,
                             slice_df_plot[slice_df_plot['Group'] == 'Patients']['ASSD_mm'].values],
                            positions=[0, 1], notch=True, patch_artist=True, widths=0.5)

                        colors = ['#1f77b4', '#ff7f0e']
                        for patch, color in zip(box_parts['boxes'], colors):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                            patch.set_edgecolor('black')
                            patch.set_linewidth(1)

                        ax.set_title('Slice-wise ASSD Comparison: HC vs Patients\nTest Set Evaluation Results',
                                    fontsize=16, fontweight='bold', pad=20)
                        ax.set_xlabel('Group', fontsize=14, fontweight='bold')
                        ax.set_ylabel('ASSD (mm) - per slice', fontsize=14, fontweight='bold')
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(['HC', 'Patients'])
                        ax.grid(True, alpha=0.6)

                        plt.tight_layout()
                        assd_plot_file = self.output_dir / f"ASSD_slicewise_comparison_HC_vs_Patients_{timestamp}.png"
                        plt.savefig(assd_plot_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Slice-wise ASSD box plot saved: {assd_plot_file}")
                else:
                    print("No valid slice-wise ASSD data for plotting")
            else:
                print("No slice-wise data collected for ASSD plotting")

        except Exception as e:
            print(f"Error creating box plots: {e}")

    def print_evaluation_summary(self):
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION SUMMARY:")
        print("=" * 60)
        print(f"Total evaluations: {self.processed_count}")
        print(f"Successful evaluations: {self.successful_count}")
        print(f"Failed evaluations: {len(self.failed_files)}")
        if self.failed_files:
            print(f"Failed files: {', '.join(self.failed_files)}")
        print(f"Success rate: {self.successful_count/self.processed_count*100:.1f}%")

        if self.results:
            df = pd.DataFrame(self.results)

            # Parse subject information for grouping
            df[['Subject', 'Visit', 'Hemisphere_Code']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
            df['Subject'] = 'sub-p' + df['Subject'].str.zfill(3)
            df['Subject_Num'] = df['Subject'].str.extract(r'sub-p(\d+)').astype(int)

            # Separate groups - corrected classification
            hc_df = df[df['Subject_Num'].isin([14, 15])]
            patients_df = df[df['Subject_Num'].isin([17, 18, 19, 20, 22, 23])]

            print(f"\nGROUP BREAKDOWN:")
            print(f"Healthy Controls (PerfTerr014, PerfTerr015): {len(hc_df)} cases")
            print(f"Patients (PerfTerr017-023): {len(patients_df)} cases")

            print(f"\nOVERALL PERFORMANCE SUMMARY:")
            metrics = ['DSC_Volume', 'DSC_Slicewise', 'IoU', 'HD95_mm', 'ASSD_mm']
            for metric in metrics:
                vals = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals) > 0:
                    print(f"{metric:15} - Mean: {vals.mean():.4f} ± {vals.std():.4f}")

            print(f"\nPER-HEMISPHERE PERFORMANCE:")
            for hemi in ['Left', 'Right']:
                hemi_data = df[df['Hemisphere'] == hemi]
                if len(hemi_data) > 0:
                    dice_mean = hemi_data['DSC_Volume'].mean()
                    print(f"{hemi:5} hemisphere: {dice_mean:.4f} Dice ({len(hemi_data)} cases)")

            # Group-specific performance
            print(f"\nGROUP-SPECIFIC PERFORMANCE:")
            for group_name, group_df in [('Healthy Controls', hc_df), ('Patients', patients_df)]:
                if len(group_df) > 0:
                    dice_vals = group_df['DSC_Volume'].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(dice_vals) > 0:
                        print(f"{group_name:16}: {dice_vals.mean():.4f} ± {dice_vals.std():.4f} Dice ({len(group_df)} cases)")
        print("=" * 60)


def main():
    # Define both prediction directories for comparison by default
    predictions_dirs = {
        'CBF_only': "/home/ubuntu/DLSegPerf/data/other/nnUNet_results_Single-class_CBF/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/fold_all/predictions",
        'CBF_T1w': "/home/ubuntu/DLSegPerf/data/other/nnUNet_results_Single-class_CBF_T1w/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/fold_all/predictions"
    }

    default_gt = "/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories/labelsTs"
    default_output = "/home/ubuntu/DLSegPerf/model_evaluation/test_evaluation/results"

    # Parse command line arguments - support backwards compatibility
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # If it looks like a single directory path, use backwards compatibility mode
        if Path(first_arg).exists():
            predictions_dirs = {'single': first_arg}

    gt_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_gt
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else default_output

    print("Test Set nnUNet Evaluation Script - CBF vs CBF+T1w Comparison")
    print("=" * 70)
    if len(predictions_dirs) > 1:
        print("Input Types:")
        print("  CBF_only: CBF LICA + CBF RICA channels")
        print("  CBF_T1w:  CBF LICA + CBF RICA + T1w channels")
    print("Prediction directories:")
    for input_type, pred_dir in predictions_dirs.items():
        print(f"  {input_type}: {pred_dir}")
    print(f"Ground truth: {gt_dir}")
    print(f"Output dir: {output_dir}\n")

    evaluator = TestSetEvaluator(predictions_dirs, gt_dir, output_dir)
    success = evaluator.run_evaluation()

    if success:
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved in: {output_dir}")

        # Print input type comparison if multiple input types
        if len(predictions_dirs) > 1 and evaluator.results:
            import pandas as pd
            import numpy as np

            df = pd.DataFrame(evaluator.results)
            print("\nINPUT TYPE PERFORMANCE SUMMARY:")
            print("-" * 40)

            for input_type in sorted(df['Input_Type'].unique()):
                input_data = df[df['Input_Type'] == input_type]
                dice_vals = input_data['DSC_Volume'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(dice_vals) > 0:
                    print(f"{input_type:15}: {dice_vals.mean():.4f} ± {dice_vals.std():.4f} DSC ({len(input_data)} cases)")

            # Calculate difference if exactly 2 input types
            if len(df['Input_Type'].unique()) == 2:
                input_types = sorted(df['Input_Type'].unique())
                type1_dice = df[df['Input_Type'] == input_types[0]]['DSC_Volume'].replace([np.inf, -np.inf], np.nan).dropna()
                type2_dice = df[df['Input_Type'] == input_types[1]]['DSC_Volume'].replace([np.inf, -np.inf], np.nan).dropna()

                if len(type1_dice) > 0 and len(type2_dice) > 0:
                    diff = type2_dice.mean() - type1_dice.mean()
                    print(f"\nDifference ({input_types[1]} - {input_types[0]}): {diff:+.4f} DSC")

        return 0
    else:
        print("Evaluation failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)