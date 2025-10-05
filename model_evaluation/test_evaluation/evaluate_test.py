#!/usr/bin/env python3
"""
Test Set nnUNet Results Evaluation Script
Evaluates predictions on test set for all segmentation approaches and creates Excel files and boxplots.
Compares: Thresholding, CBF-only, CBF+T1w, CBF+FLAIR, CBF+T1w+FLAIR
  - Separate analysis for HC (sub-p014, sub-p015) and Patients (sub-p017-023)
  - Sheets per Excel: Per_Case_Details, Per_Hemisphere_Summary, Overall_Summary
  - Metrics: DSC_Volume, DSC_Slicewise, IoU, Sensitivity, Precision,
             Specificity, RVE_Percent, HD95_mm, ASSD_mm
  - Boxplots: DSC (volume-based) and ASSD (slice-wise) with Left/Right hemisphere separation
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

        # Results storage - now includes approach type
        self.results = []
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []

        # Store slice-wise data for plotting (includes ASSD per slice)
        self.slice_wise_data = []

        # Store statistical results for significance brackets
        self.statistical_results = {}  # Format: {group_name: DataFrame with statistical comparisons}

        # Mapping from internal keys to display names
        self.approach_display_names = {
            'Thresholding': 'Thresholding',
            'CBF': 'CBF',
            'CBF_T1w': 'CBF+T1w',
            'CBF_FLAIR': 'CBF+FLAIR',
            'CBF_T1w_FLAIR': 'CBF+T1w+FLAIR'
        }

        # Mapping for Excel file naming
        self.approach_file_names = {
            'Thresholding': 'Thresholding',
            'CBF': 'Single-class_CBF',
            'CBF_T1w': 'Single-class_CBF_T1w',
            'CBF_FLAIR': 'Single-class_CBF_FLAIR',
            'CBF_T1w_FLAIR': 'Single-class_CBF_T1w_FLAIR'
        }

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

    def calculate_slice_wise_assd(self, gt_mask, pred_mask, spacing, base_name, approach):
        """Calculate ASSD for each slice and store for plotting"""
        slice_data = []
        nz = gt_mask.shape[2]

        # Extract group from base_name
        case_base = base_name.split('_')[0] if '_' in base_name else base_name
        subject_num = int(case_base.split('-')[0].replace('PerfTerr', ''))
        group = 'HC' if subject_num in [14, 15] else 'Patients'

        # Extract hemisphere from filename
        hemisphere = 'Left' if base_name.endswith('-L') else 'Right'

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
                        'Hemisphere': hemisphere,
                        'Approach': approach,
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

    def evaluate_single_case(self, pred_file, gt_file, base_name, approach):
        print(f"  Evaluating {base_name} ({approach})...")

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

        # EXCLUDE volumes with DSC of zero from further evaluation
        if dice_score == 0.0:
            print(f"    EXCLUDED: DSC = 0.0 (volume excluded from evaluation)")
            return None

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
        slice_assd_data = self.calculate_slice_wise_assd(gt_mask, pred_mask, spacing, base_name, approach)
        self.slice_wise_data.extend(slice_assd_data)

        # Extract hemisphere from filename
        hemisphere = 'Left' if base_name.endswith('-L') else 'Right'

        print(f"    Dice: {dice_score:.4f}, Dice(slice): {dice_slice:.4f}, IoU: {iou:.4f}, HD95: {hd95:.2f}")

        return {
            'Base_Name': base_name,
            'Approach': approach,
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
        """Find matching prediction and ground truth files for all approaches"""
        all_test_cases = []

        for approach, predictions_dir in self.predictions_dirs.items():
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
                    all_test_cases.append((pred_file, gt_file, base_name, approach))
                else:
                    print(f"Warning: No matching GT file for {base_name} ({approach})")

        return all_test_cases

    def run_evaluation(self):
        print("Test Set Results Evaluation - All Segmentation Approaches")
        print("=" * 70)
        print("Segmentation approaches:")
        for approach, pred_dir in self.predictions_dirs.items():
            print(f"  {self.approach_display_names.get(approach, approach)}: {pred_dir}")
        print(f"Ground truth: {self.gt_dir}")
        print(f"Output directory: {self.output_dir}\n")

        # Delete old result files before creating new ones
        print("Cleaning up old result files...")
        for old_file in self.output_dir.glob("*"):
            if old_file.is_file():
                try:
                    old_file.unlink()
                    print(f"  Deleted: {old_file.name}")
                except Exception as e:
                    print(f"  Warning: Could not delete {old_file.name}: {e}")
        print()

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
        for approach, pred_dir in self.predictions_dirs.items():
            if not pred_dir.exists():
                print(f"ERROR: Predictions directory not found ({approach}): {pred_dir}")
                return False

        if not self.gt_dir.exists():
            print(f"ERROR: Ground truth directory not found: {self.gt_dir}")
            return False

        # Find test cases
        test_cases = self.find_test_cases()
        if not test_cases:
            print("ERROR: No test cases found")
            return False

        print(f"Found {len(test_cases)} test cases across {len(self.predictions_dirs)} approaches")

        # Evaluate each case
        for pred_file, gt_file, base_name, approach in test_cases:
            print(f"\nProcessing {base_name} ({approach}):")

            self.processed_count += 1
            result = self.evaluate_single_case(pred_file, gt_file, base_name, approach)
            if result is not None:
                self.results.append(result)
                self.successful_count += 1
            else:
                self.failed_files.append(f"{base_name} ({approach})")

        print("\nSaving results to Excel...")
        self.save_results_to_excel()

        print("\nPerforming statistical testing...")
        self.perform_statistical_testing()

        print("\nCreating box plots...")
        self.create_box_plots()

        self.print_evaluation_summary()
        return True

    def create_excel_for_approach_and_group(self, df, approach, group_name, timestamp):
        """Create Excel file for a specific approach and group (HC or Patients)"""
        # Get file name components
        approach_file_name = self.approach_file_names.get(approach, approach)
        excel_file = self.output_dir / f"test_results_{group_name}_{approach_file_name}_{timestamp}.xlsx"

        # ----- Sheet 1: Per_Case_Details -----
        base_cols = ['Subject', 'Visit', 'Hemisphere', 'Base_Name']
        metric_cols = ['DSC_Volume', 'DSC_Slicewise', 'IoU',
                      'Sensitivity', 'Precision', 'Specificity',
                      'RVE_Percent', 'HD95_mm', 'ASSD_mm']

        per_case_cols = base_cols + metric_cols
        per_case_df = df[per_case_cols].copy()

        # ----- Sheet 2: Per_Hemisphere_Summary -----
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
            per_case_df.to_excel(writer, sheet_name='Per_Case_Details', index=False)
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

        print(f"  Excel saved ({group_name}, {approach}): {excel_file.name}")
        return excel_file

    def save_results_to_excel(self):
        """Create separate Excel files for each approach and group combination"""
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

            # Extract subject number for grouping
            df['Subject_Num'] = df['Subject'].str.extract(r'sub-p(\d+)').astype(int)

            # Sort by Approach, Subject, Visit, Hemisphere
            df = df.sort_values(['Approach', 'Subject', 'Visit', 'Hemisphere'], na_position='last')

            # Separate into healthy controls (PerfTerr014, PerfTerr015) and patients (PerfTerr017-023)
            hc_df = df[df['Subject_Num'].isin([14, 15])].copy()
            patients_df = df[df['Subject_Num'].isin([17, 18, 19, 20, 22, 23])].copy()

            print(f"Total: {len(df)} cases")
            print(f"  Healthy Controls: {len(hc_df)} cases")
            print(f"  Patients: {len(patients_df)} cases")

            # Create separate Excel files for each approach and group
            excel_files = []
            for approach in df['Approach'].unique():
                # HC group
                hc_approach_df = hc_df[hc_df['Approach'] == approach].copy()
                if len(hc_approach_df) > 0:
                    hc_approach_df = hc_approach_df.drop(['Subject_Num', 'Hemisphere_Code', 'Approach'], axis=1, errors='ignore')
                    excel_file = self.create_excel_for_approach_and_group(hc_approach_df, approach, 'HC', timestamp)
                    excel_files.append(excel_file)

                # Patients group
                patients_approach_df = patients_df[patients_df['Approach'] == approach].copy()
                if len(patients_approach_df) > 0:
                    patients_approach_df = patients_approach_df.drop(['Subject_Num', 'Hemisphere_Code', 'Approach'], axis=1, errors='ignore')
                    excel_file = self.create_excel_for_approach_and_group(patients_approach_df, approach, 'patients', timestamp)
                    excel_files.append(excel_file)

            print(f"Created {len(excel_files)} Excel files")
            return excel_files

        except Exception as e:
            print(f"Error saving Excel file: {e}")
            import traceback
            traceback.print_exc()
            return []

    def create_box_plots(self):
        """
        Create separate boxplot PNG files for HC and Patients groups.
        Each PNG shows all 5 approaches with Left and Right hemisphere boxes.
        Creates plots for both DSC (volume-based) and ASSD (slice-wise).
        """
        if not PLOTTING_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Skipping box plots.")
            return

        if not self.results:
            print("No results available for plotting")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare data
            df = pd.DataFrame(self.results)
            df[['Subject_str', 'Visit', 'Hemisphere_Code']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
            df['Subject_Num'] = df['Subject_str'].astype(int)
            df['Group'] = df['Subject_Num'].apply(lambda x: 'HC' if x in [14, 15] else 'Patients')

            # Set plot style
            plt.style.use('default')
            sns.set_palette("husl")

            # Get sorted approach order for consistent plotting
            approach_order = sorted(df['Approach'].unique(),
                                   key=lambda x: ['Thresholding', 'CBF', 'CBF_T1w', 'CBF_FLAIR', 'CBF_T1w_FLAIR'].index(x)
                                   if x in ['Thresholding', 'CBF', 'CBF_T1w', 'CBF_FLAIR', 'CBF_T1w_FLAIR'] else 999)

            # ====== Plot 1: DSC (volume-based) boxplots ======
            # print("  Creating DSC boxplots...")
            # self._create_dsc_boxplots(df, approach_order, timestamp)

            # ====== Plot 2: ASSD (slice-wise) boxplots ======
            # if self.slice_wise_data:
            #     print("  Creating ASSD boxplots...")
            #     self._create_assd_boxplots(approach_order, timestamp)

            # ====== Plot 3: Combined HC plot (DSC + ASSD) ======
            if self.slice_wise_data:
                print("  Creating combined HC plot...")
                self._create_combined_hc_plot(df, approach_order, timestamp)
            else:
                print("  No slice-wise ASSD data available for plotting")

        except Exception as e:
            print(f"Error creating box plots: {e}")
            import traceback
            traceback.print_exc()

    def perform_statistical_testing(self):
        """Perform Wilcoxon signed-rank test comparing segmentation approaches"""
        from scipy import stats
        from itertools import combinations

        if not self.slice_wise_data:
            print("  No slice-wise data available for statistical testing")
            return

        print("\n  Performing Statistical Testing (Wilcoxon signed-rank test)...")

        slice_df = pd.DataFrame(self.slice_wise_data)
        slice_df_plot = slice_df[slice_df['ASSD_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        if len(slice_df_plot) == 0:
            print("  No valid ASSD data for statistical testing")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Process each group separately (HC and Patients)
        for group_name in ['HC', 'Patients']:
            group_data = slice_df_plot[slice_df_plot['Group'] == group_name].copy()

            if len(group_data) == 0:
                print(f"  No data for {group_name}")
                continue

            print(f"\n  {group_name} Statistical Analysis:")
            print(f"  {'-' * 40}")

            all_stats_results = []

            # For HC: compare approaches across combined hemispheres
            # For Patients: compare approaches within each hemisphere separately
            if group_name == 'HC':
                # HC: Combined hemisphere comparison
                available_approaches = group_data['Approach'].unique().tolist()

                if len(available_approaches) < 2:
                    print(f"    Insufficient approaches for comparison")
                    continue

                print(f"    Combined Hemispheres: {len(available_approaches)} approaches")

                # Perform all pairwise comparisons across approaches
                for approach1, approach2 in combinations(available_approaches, 2):
                    data1_df = group_data[group_data['Approach'] == approach1]
                    data2_df = group_data[group_data['Approach'] == approach2]

                    # Find paired slices (same case and slice number)
                    merged_df = data1_df.merge(data2_df, on=['Case', 'Slice'], suffixes=('_1', '_2'))

                    if len(merged_df) < 10:  # Minimum sample size
                        continue

                    # Get paired slice-wise data
                    paired_data1 = merged_df['ASSD_mm_1'].values
                    paired_data2 = merged_df['ASSD_mm_2'].values

                    try:
                        # Wilcoxon signed-rank test
                        statistic, p_value = stats.wilcoxon(paired_data1, paired_data2, alternative='two-sided')

                        # Calculate effect size (r = Z / sqrt(N))
                        n = len(paired_data1)
                        z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0
                        effect_size = z_score / np.sqrt(n)

                        # Calculate medians
                        median1 = np.median(paired_data1)
                        median2 = np.median(paired_data2)
                        median_diff = median1 - median2

                        # Determine significance
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "ns"

                        all_stats_results.append({
                            'Hemisphere': 'Combined',  # For HC, mark as combined
                            'Config1': approach1,
                            'Config2': approach2,
                            'Median1': median1,
                            'Median2': median2,
                            'Median_Diff': median_diff,
                            'Statistic': statistic,
                            'P_Value': p_value,
                            'Effect_Size': effect_size,
                            'Significance': significance,
                            'N_Paired_Slices': n,
                            'N1_Total_Slices': len(data1_df),
                            'N2_Total_Slices': len(data2_df)
                        })

                    except Exception as e:
                        print(f"      Error comparing {approach1} vs {approach2}: {e}")

            else:
                # Patients: Process each hemisphere separately
                for hemisphere in ['Left', 'Right']:
                    hemi_data = group_data[group_data['Hemisphere'] == hemisphere]
                    available_approaches = hemi_data['Approach'].unique().tolist()

                    if len(available_approaches) < 2:
                        print(f"    {hemisphere}: Insufficient approaches for comparison")
                        continue

                    print(f"    {hemisphere} Hemisphere: {len(available_approaches)} approaches")

                    # Perform all pairwise comparisons
                    for approach1, approach2 in combinations(available_approaches, 2):
                        data1_df = hemi_data[hemi_data['Approach'] == approach1]
                        data2_df = hemi_data[hemi_data['Approach'] == approach2]

                        # Find paired slices (same case and slice number)
                        merged_df = data1_df.merge(data2_df, on=['Case', 'Slice'], suffixes=('_1', '_2'))

                        if len(merged_df) < 10:  # Minimum sample size
                            continue

                        # Get paired slice-wise data
                        paired_data1 = merged_df['ASSD_mm_1'].values
                        paired_data2 = merged_df['ASSD_mm_2'].values

                        try:
                            # Wilcoxon signed-rank test
                            statistic, p_value = stats.wilcoxon(paired_data1, paired_data2, alternative='two-sided')

                            # Calculate effect size (r = Z / sqrt(N))
                            n = len(paired_data1)
                            z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0
                            effect_size = z_score / np.sqrt(n)

                            # Calculate medians
                            median1 = np.median(paired_data1)
                            median2 = np.median(paired_data2)
                            median_diff = median1 - median2

                            # Determine significance
                            if p_value < 0.001:
                                significance = "***"
                            elif p_value < 0.01:
                                significance = "**"
                            elif p_value < 0.05:
                                significance = "*"
                            else:
                                significance = "ns"

                            all_stats_results.append({
                                'Hemisphere': hemisphere,
                                'Config1': approach1,
                                'Config2': approach2,
                                'Median1': median1,
                                'Median2': median2,
                                'Median_Diff': median_diff,
                                'Statistic': statistic,
                                'P_Value': p_value,
                                'Effect_Size': effect_size,
                                'Significance': significance,
                                'N_Paired_Slices': n,
                                'N1_Total_Slices': len(data1_df),
                                'N2_Total_Slices': len(data2_df)
                            })

                        except Exception as e:
                            print(f"      Error comparing {approach1} vs {approach2}: {e}")

            # Save results for this group
            if all_stats_results:
                stats_df = self._save_statistical_results(all_stats_results, group_name, timestamp)
                # Store for significance brackets in plots
                self.statistical_results[group_name] = stats_df
            else:
                print(f"    No statistical comparisons performed for {group_name}")

    def _save_statistical_results(self, all_stats_results, group_name, timestamp):
        """Save statistical test results to Excel file"""
        stats_df = pd.DataFrame(all_stats_results)

        # Sort by hemisphere and p-value
        stats_df = stats_df.sort_values(['Hemisphere', 'P_Value'])

        # Add effect size interpretation
        def interpret_effect_size(r):
            r = abs(r)
            if r < 0.1:
                return "Negligible"
            elif r < 0.3:
                return "Small"
            elif r < 0.5:
                return "Medium"
            else:
                return "Large"

        stats_df['Effect_Size_Interpretation'] = stats_df['Effect_Size'].apply(interpret_effect_size)

        # Apply Bonferroni correction within each hemisphere
        corrected_results = []
        for hemisphere, group in stats_df.groupby('Hemisphere'):
            n_comparisons = len(group)
            group = group.copy()
            group['P_Value_Bonferroni'] = group['P_Value'] * n_comparisons
            group['P_Value_Bonferroni'] = np.minimum(group['P_Value_Bonferroni'], 1.0)

            # Bonferroni significance
            def get_bonferroni_significance(p_bonf):
                if p_bonf < 0.001:
                    return "***"
                elif p_bonf < 0.01:
                    return "**"
                elif p_bonf < 0.05:
                    return "*"
                else:
                    return "ns"

            group['Significance_Bonferroni'] = group['P_Value_Bonferroni'].apply(get_bonferroni_significance)
            corrected_results.append(group)

        stats_df = pd.concat(corrected_results, ignore_index=True)

        # Save to Excel
        stats_file = self.output_dir / f"test_statistical_comparison_{group_name}_{timestamp}.xlsx"

        with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
            # Main results sheet
            stats_df.to_excel(writer, sheet_name='All_Comparisons', index=False)

            # Significant results only (uncorrected)
            significant_df = stats_df[stats_df['P_Value'] < 0.05]
            if not significant_df.empty:
                significant_df.to_excel(writer, sheet_name='Significant_Uncorrected', index=False)

            # Significant results with Bonferroni correction
            bonferroni_significant_df = stats_df[stats_df['P_Value_Bonferroni'] < 0.05]
            if not bonferroni_significant_df.empty:
                bonferroni_significant_df.to_excel(writer, sheet_name='Significant_Bonferroni', index=False)

            # Summary by hemisphere
            summary_by_hemisphere = stats_df.groupby('Hemisphere').agg({
                'P_Value': ['count', lambda x: sum(x < 0.05), lambda x: sum(x < 0.01)],
                'P_Value_Bonferroni': [lambda x: sum(x < 0.05)],
                'Effect_Size': ['mean', 'std'],
                'N_Paired_Slices': 'mean'
            }).round(4)

            summary_by_hemisphere.columns = ['Total_Comparisons', 'Significant_p05', 'Significant_p01',
                                           'Bonferroni_Significant', 'Mean_Effect_Size', 'Std_Effect_Size',
                                           'Avg_Paired_Slices']
            summary_by_hemisphere.to_excel(writer, sheet_name='Summary_by_Hemisphere')

        print(f"    Statistical results saved: {stats_file.name}")
        print(f"      Total comparisons: {len(stats_df)}")
        print(f"      Significant (p<0.05): {sum(stats_df['P_Value'] < 0.05)}")
        print(f"      Bonferroni significant: {sum(stats_df['P_Value_Bonferroni'] < 0.05)}")

        # Return the DataFrame for use in plotting
        return stats_df

    def _create_dsc_boxplots(self, df, approach_order, timestamp):
        """Create DSC boxplots for HC and Patients separately - matching reference style
        For HC: combine left and right hemispheres
        For Patients: keep hemispheres separate
        """
        # Filter out invalid DSC values
        df_plot = df[df['DSC_Volume'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Define distinct colors for each approach (matching inter-input style)
        approach_colors = {
            'Thresholding': '#d62728',       # Red (matching CBF+T1w+FLAIR reference)
            'CBF': '#1f77b4',                # Blue (matching CBF reference)
            'CBF_T1w': '#ff7f0e',            # Orange (matching CBF+T1w reference)
            'CBF_FLAIR': '#2ca02c',          # Green (matching CBF+FLAIR reference)
            'CBF_T1w_FLAIR': '#9467bd'       # Purple
        }

        for group_name in ['HC', 'Patients']:
            group_data = df_plot[df_plot['Group'] == group_name].copy()
            if len(group_data) == 0:
                continue

            # Create figure (matching reference)
            fig, ax = plt.subplots(figsize=(14, 8))

            # Prepare data for grouped plotting
            plot_positions = []
            plot_data_list = []
            plot_colors = []

            position = 0
            hemisphere_positions = {}
            hemisphere_centers = {}

            # For HC: combine hemispheres, for Patients: keep separate
            if group_name == 'HC':
                # HC: Single group combining both hemispheres
                hemispheres = ['Combined']
                hemisphere_positions['Combined'] = []
                start_pos = position

                for approach in approach_order:
                    # Combine data from both hemispheres
                    approach_data = group_data[group_data['Approach'] == approach]

                    if not approach_data.empty:
                        plot_data_list.append(approach_data['DSC_Volume'].values)
                    else:
                        plot_data_list.append([])

                    plot_colors.append(approach_colors.get(approach, '#95a5a6'))
                    plot_positions.append(position)
                    hemisphere_positions['Combined'].append(position)
                    position += 0.6  # Spacing within group (balanced spacing)

                # Calculate center for labeling
                hemisphere_centers['Combined'] = (start_pos + position - 0.6) / 2

            else:
                # Patients: Separate hemispheres
                hemispheres = ['Left', 'Right']

                for hemi_idx, hemisphere in enumerate(hemispheres):
                    hemisphere_positions[hemisphere] = []
                    start_pos = position

                    for approach in approach_order:
                        approach_data = group_data[(group_data['Approach'] == approach) &
                                                  (group_data['Hemisphere'] == hemisphere)]

                        if not approach_data.empty:
                            plot_data_list.append(approach_data['DSC_Volume'].values)
                        else:
                            plot_data_list.append([])

                        plot_colors.append(approach_colors.get(approach, '#95a5a6'))
                        plot_positions.append(position)
                        hemisphere_positions[hemisphere].append(position)
                        position += 0.6  # Spacing within hemisphere (balanced spacing)

                    # Calculate hemisphere center for labeling
                    hemisphere_centers[hemisphere] = (start_pos + position - 0.6) / 2

                    # Add gap between hemispheres
                    if hemi_idx < len(hemispheres) - 1:
                        position += 0.25  # Gap between hemispheres (balanced spacing)

            # Create boxplot (matching reference style)
            bp = ax.boxplot(
                plot_data_list,
                positions=plot_positions,
                notch=False,
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color='black', linewidth=2)
            )

            # Color the boxes
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)

            # Customize the plot (matching reference)
            if group_name == 'HC':
                ax.set_title(f'{group_name} Test Set Evaluation: Volume-wise DSC by Segmentation Approach / Input Configuration',
                           fontsize=20, fontweight='bold', pad=25)
                ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=16, fontweight='bold')
            else:
                ax.set_title(f'{group_name} Test Set: DSC by Segmentation Approach and Hemisphere\n'
                            f'Test Set Evaluation Results',
                           fontsize=20, fontweight='bold', pad=25)
                ax.set_xlabel('Hemisphere', fontsize=16, fontweight='bold')

            ax.set_ylabel('DSC per volume', fontsize=16, fontweight='bold')

            # Set custom x-axis labels
            if group_name == 'HC':
                # Custom labels for each approach with descriptive names
                approach_labels = {
                    'Thresholding': 'Thresholding',
                    'CBF': 'nnUNet w/\nCBF',
                    'CBF_T1w': 'nnUNet w/\nCBF+MP-RAGE',
                    'CBF_FLAIR': 'nnUNet w/\nCBF+FLAIR',
                    'CBF_T1w_FLAIR': 'nnUNet w/\nCBF+MP-RAGE+FLAIR'
                }
                tick_positions = [hemisphere_positions['Combined'][i] for i in range(len(approach_order))]
                tick_labels = [approach_labels.get(a, a) for a in approach_order]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
            else:
                ax.set_xticks([hemisphere_centers['Left'], hemisphere_centers['Right']])
                ax.set_xticklabels(['Left', 'Right'])
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            # Add grid (matching reference)
            ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
            ax.set_axisbelow(True)

            # Set y-axis limits
            if group_name == 'HC':
                ax.set_ylim(0.8, 1.0)
            else:
                ax.set_ylim(0, 1.0)

            # Create legend for approaches - only for Patients
            if group_name == 'Patients':
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=approach_colors[approach],
                                                alpha=0.7, edgecolor='black')
                                  for approach in approach_order]
                ax.legend(legend_elements, [self.approach_display_names.get(a, a) for a in approach_order],
                         title='Segmentation Approach', title_fontsize=12, fontsize=11,
                         loc='lower right', bbox_to_anchor=(1.0, 0.0))
            else:
                # HC: Add legend box with "Median [Q1-Q3]" explanation
                ax.text(0.98, 0.02, 'Median [Q1-Q3]\nn = sample size',
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                alpha=0.9, edgecolor='gray', linewidth=1.5))

            # Add median [IQR] and n annotations (matching reference positioning)
            self._add_dsc_annotations(ax, group_data, hemisphere_positions, approach_order, approach_colors, hemispheres)

            # Add significance brackets (Bonferroni-corrected)
            # For DSC plots, brackets go above boxes
            if group_name == 'HC':
                # HC: Show significance between approaches (combined hemispheres)
                stats_df = self.statistical_results.get(group_name, None)
                if stats_df is not None:
                    self._add_significance_brackets_approaches(ax, stats_df, hemisphere_positions['Combined'],
                                                              approach_order, position='above')
            else:
                # Patients: Show significance for both hemispheres separately
                stats_df = self.statistical_results.get(group_name, None)
                if stats_df is not None:
                    self._add_significance_brackets_both_hemispheres(ax, stats_df, hemisphere_positions, approach_order,
                                                                     hemispheres, position='above')

            plt.tight_layout()
            plot_file = self.output_dir / f"DSC_volume_{group_name}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    Saved: {plot_file.name}")

    def _add_significance_brackets_approaches(self, ax, stats_df, approach_positions, approach_order, position='above'):
        """Add significance brackets for approach comparisons (used for HC combined hemispheres)

        Args:
            ax: matplotlib axis
            stats_df: DataFrame with statistical results
            approach_positions: list of x-positions for each approach
            approach_order: list of approach names in order
            position: 'above' or 'below' - where to place brackets relative to boxes
        """
        if stats_df is None or stats_df.empty:
            return

        # Filter for Bonferroni-significant results only
        significant_stats = stats_df[stats_df['P_Value_Bonferroni'] < 0.05].copy()

        if significant_stats.empty:
            return

        # Get the mapping from approach names to positions
        approach_to_pos = {approach: approach_positions[i]
                          for i, approach in enumerate(approach_order)
                          if i < len(approach_positions)}

        # Get y-axis limits
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Prepare list of significant pairs with their positions
        significant_pairs = []
        for _, row in significant_stats.iterrows():
            approach1 = row['Config1']
            approach2 = row['Config2']
            p_bonf = row['P_Value_Bonferroni']

            if approach1 in approach_to_pos and approach2 in approach_to_pos:
                pos1 = approach_to_pos[approach1]
                pos2 = approach_to_pos[approach2]

                # Determine significance symbol
                if p_bonf < 0.001:
                    symbol = '***'
                elif p_bonf < 0.01:
                    symbol = '**'
                elif p_bonf < 0.05:
                    symbol = '*'
                else:
                    continue

                significant_pairs.append({
                    'pos1': min(pos1, pos2),
                    'pos2': max(pos1, pos2),
                    'symbol': symbol,
                    'span': abs(pos2 - pos1)
                })

        if not significant_pairs:
            return

        # Sort by span (smallest first) to minimize overlap
        significant_pairs.sort(key=lambda x: x['span'])

        # Assign bracket heights to avoid overlaps
        bracket_heights = []
        bracket_height_increment = y_range * 0.045  # 4.5% of y-range per bracket level (increased to prevent overlap)
        bracket_base_offset = y_range * 0.008  # Start with some separation (0.8% from boxes)

        for pair in significant_pairs:
            # Find the first available height that doesn't overlap
            level = 0
            while True:
                if position == 'above':
                    height = y_max + bracket_base_offset + (level * bracket_height_increment)
                else:  # below
                    height = y_min - bracket_base_offset - (level * bracket_height_increment)

                # Check if this height overlaps with any existing bracket in the same x-range
                overlaps = False
                for existing_pair, existing_height in bracket_heights:
                    # Check x-range overlap
                    if not (pair['pos2'] < existing_pair['pos1'] or pair['pos1'] > existing_pair['pos2']):
                        # Check height overlap (within one increment)
                        if abs(height - existing_height) < bracket_height_increment * 0.8:
                            overlaps = True
                            break

                if not overlaps:
                    bracket_heights.append((pair, height))
                    break

                level += 1
                if level > 15:  # Safety limit
                    break

        # Draw all brackets
        for pair, height in bracket_heights:
            pos1 = pair['pos1']
            pos2 = pair['pos2']
            symbol = pair['symbol']

            # Draw horizontal line
            ax.plot([pos1, pos2], [height, height], 'k-', linewidth=1.5)

            # Draw vertical ticks at ends
            tick_height = y_range * 0.01
            if position == 'above':
                ax.plot([pos1, pos1], [height, height - tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height - tick_height], 'k-', linewidth=1.5)
                # Add significance symbol above the line (closer to bracket)
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height - y_range * 0.012, symbol, ha='center', va='bottom',
                       fontsize=16, fontweight='bold')
            else:  # below
                ax.plot([pos1, pos1], [height, height + tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height + tick_height], 'k-', linewidth=1.5)
                # Add significance symbol below the line (with proper spacing)
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height - y_range * 0.010, symbol, ha='center', va='top',
                       fontsize=16, fontweight='bold')

        # Adjust y-axis to accommodate brackets (ensure brackets stay within plot)
        if bracket_heights:
            if position == 'above':
                max_bracket_height = max(h for _, h in bracket_heights)
                # Calculate new y_range based on extended limits
                new_y_max = max_bracket_height + y_range * 0.050  # 5% padding for symbol
                ax.set_ylim(y_min, new_y_max)
            else:  # below
                min_bracket_height = min(h for _, h in bracket_heights)
                # Calculate new y_range based on extended limits
                new_y_min = min_bracket_height - y_range * 0.050  # 5% padding for symbol
                ax.set_ylim(new_y_min, y_max)

    def _add_significance_brackets_both_hemispheres(self, ax, stats_df, hemisphere_positions, approach_order, hemispheres, position='above'):
        """Add significance brackets for both hemispheres with coordinated heights"""
        if stats_df is None or stats_df.empty:
            return

        # Get y-axis limits
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Collect all significant pairs from both hemispheres
        all_pairs_by_hemisphere = {}

        for hemisphere in hemispheres:
            # Filter for this hemisphere and Bonferroni-significant results only
            hemi_stats = stats_df[(stats_df['Hemisphere'] == hemisphere) &
                                  (stats_df['P_Value_Bonferroni'] < 0.05)].copy()

            if hemi_stats.empty:
                all_pairs_by_hemisphere[hemisphere] = []
                continue

            # Get the mapping from approach names to positions
            approach_to_pos = {approach: hemisphere_positions[hemisphere][i]
                              for i, approach in enumerate(approach_order)
                              if i < len(hemisphere_positions[hemisphere])}

            # Prepare list of significant pairs with their positions
            significant_pairs = []
            for _, row in hemi_stats.iterrows():
                approach1 = row['Config1']
                approach2 = row['Config2']
                p_bonf = row['P_Value_Bonferroni']

                if approach1 in approach_to_pos and approach2 in approach_to_pos:
                    pos1 = approach_to_pos[approach1]
                    pos2 = approach_to_pos[approach2]

                    # Determine significance symbol
                    if p_bonf < 0.001:
                        symbol = '***'
                    elif p_bonf < 0.01:
                        symbol = '**'
                    elif p_bonf < 0.05:
                        symbol = '*'
                    else:
                        continue

                    significant_pairs.append({
                        'pos1': min(pos1, pos2),
                        'pos2': max(pos1, pos2),
                        'symbol': symbol,
                        'span': abs(pos2 - pos1),
                        'hemisphere': hemisphere
                    })

            # Sort by span (smallest first) to minimize overlap
            significant_pairs.sort(key=lambda x: x['span'])
            all_pairs_by_hemisphere[hemisphere] = significant_pairs

        # Check if we have any brackets to draw
        total_brackets = sum(len(pairs) for pairs in all_pairs_by_hemisphere.values())
        if total_brackets == 0:
            return

        # Assign bracket heights with coordination across hemispheres
        # All brackets share the same height levels
        bracket_heights = []
        bracket_height_increment = y_range * 0.045  # 4.5% of y-range per bracket level (increased to prevent overlap)
        bracket_base_offset = y_range * 0.008  # Start with some separation (0.8% from boxes)

        # Process all pairs together, alternating between hemispheres to pack efficiently
        all_pairs_flat = []
        for hemisphere in hemispheres:
            for pair in all_pairs_by_hemisphere[hemisphere]:
                all_pairs_flat.append(pair)

        # Sort all pairs by span
        all_pairs_flat.sort(key=lambda x: x['span'])

        for pair in all_pairs_flat:
            # Find the first available height that doesn't overlap
            level = 0
            while True:
                if position == 'above':
                    height = y_max + bracket_base_offset + (level * bracket_height_increment)
                else:  # below
                    height = y_min - bracket_base_offset - (level * bracket_height_increment)

                # Check if this height overlaps with any existing bracket in the same x-range
                overlaps = False
                for existing_pair, existing_height in bracket_heights:
                    # Check x-range overlap
                    if not (pair['pos2'] < existing_pair['pos1'] or pair['pos1'] > existing_pair['pos2']):
                        # Check height overlap (within one increment)
                        if abs(height - existing_height) < bracket_height_increment * 0.8:
                            overlaps = True
                            break

                if not overlaps:
                    bracket_heights.append((pair, height))
                    break

                level += 1
                if level > 15:  # Safety limit
                    break

        # Draw all brackets
        for pair, height in bracket_heights:
            pos1 = pair['pos1']
            pos2 = pair['pos2']
            symbol = pair['symbol']

            # Draw horizontal line
            ax.plot([pos1, pos2], [height, height], 'k-', linewidth=1.5)

            # Draw vertical ticks at ends
            tick_height = y_range * 0.01
            if position == 'above':
                ax.plot([pos1, pos1], [height, height - tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height - tick_height], 'k-', linewidth=1.5)
                # Add significance symbol above the line (with proper spacing)
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height + y_range * 0.005, symbol, ha='center', va='bottom',
                       fontsize=16, fontweight='bold')
            else:  # below
                ax.plot([pos1, pos1], [height, height + tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height + tick_height], 'k-', linewidth=1.5)
                # Add significance symbol below the line (with proper spacing)
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height - y_range * 0.010, symbol, ha='center', va='top',
                       fontsize=16, fontweight='bold')

        # Adjust y-axis to accommodate brackets (ensure brackets stay within plot)
        if bracket_heights:
            if position == 'above':
                max_bracket_height = max(h for _, h in bracket_heights)
                # Add padding for the symbol above the bracket
                new_y_max = max_bracket_height + y_range * 0.050  # 5% padding for symbol
                ax.set_ylim(y_min, new_y_max)
            else:  # below
                min_bracket_height = min(h for _, h in bracket_heights)
                # Add padding for the symbol below the bracket
                new_y_min = min_bracket_height - y_range * 0.050  # 5% padding for symbol
                ax.set_ylim(new_y_min, y_max)

    def _add_significance_brackets(self, ax, stats_df, hemisphere, hemisphere_positions, approach_order, position='above'):
        """Add significance brackets showing Bonferroni-corrected significant differences

        Args:
            position: 'above' or 'below' - where to place brackets relative to boxes
        """
        if stats_df is None or stats_df.empty:
            return

        # Filter for this hemisphere and Bonferroni-significant results only
        hemi_stats = stats_df[(stats_df['Hemisphere'] == hemisphere) &
                              (stats_df['P_Value_Bonferroni'] < 0.05)].copy()

        if hemi_stats.empty:
            return

        # Get the mapping from approach names to positions
        approach_to_pos = {approach: hemisphere_positions[hemisphere][i]
                          for i, approach in enumerate(approach_order)
                          if i < len(hemisphere_positions[hemisphere])}

        # Get y-axis limits (CURRENT limits, before adding brackets)
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Prepare list of significant pairs with their positions
        significant_pairs = []
        for _, row in hemi_stats.iterrows():
            approach1 = row['Config1']
            approach2 = row['Config2']
            p_bonf = row['P_Value_Bonferroni']

            if approach1 in approach_to_pos and approach2 in approach_to_pos:
                pos1 = approach_to_pos[approach1]
                pos2 = approach_to_pos[approach2]

                # Determine significance symbol
                if p_bonf < 0.001:
                    symbol = '***'
                elif p_bonf < 0.01:
                    symbol = '**'
                elif p_bonf < 0.05:
                    symbol = '*'
                else:
                    continue  # Skip non-significant

                significant_pairs.append({
                    'pos1': min(pos1, pos2),
                    'pos2': max(pos1, pos2),
                    'symbol': symbol,
                    'span': abs(pos2 - pos1)
                })

        if not significant_pairs:
            return

        # Sort by span (smallest first) to minimize overlap
        significant_pairs.sort(key=lambda x: x['span'])

        # Assign bracket heights to avoid overlaps
        bracket_heights = []
        bracket_height_increment = y_range * 0.05  # 5% of y-range per bracket level

        if position == 'above':
            bracket_base_offset = y_range * 0.015  # Start very close to top (1.5% above)
        else:  # below
            bracket_base_offset = y_range * 0.015  # Start very close to bottom (1.5% below)

        for pair in significant_pairs:
            # Find the first available height that doesn't overlap with existing brackets
            level = 0
            while True:
                if position == 'above':
                    height = y_max + bracket_base_offset + (level * bracket_height_increment)
                else:  # below
                    height = y_min - bracket_base_offset - (level * bracket_height_increment)

                # Check if this height overlaps with any existing bracket in the same x-range
                overlaps = False
                for existing_pair, existing_height in bracket_heights:
                    # Check x-range overlap
                    if not (pair['pos2'] < existing_pair['pos1'] or pair['pos1'] > existing_pair['pos2']):
                        # Check height overlap (within one increment)
                        if abs(height - existing_height) < bracket_height_increment * 0.8:
                            overlaps = True
                            break

                if not overlaps:
                    bracket_heights.append((pair, height))
                    break

                level += 1
                if level > 10:  # Safety limit
                    break

        # Draw the brackets
        for pair, height in bracket_heights:
            pos1 = pair['pos1']
            pos2 = pair['pos2']
            symbol = pair['symbol']

            # Draw horizontal line
            ax.plot([pos1, pos2], [height, height], 'k-', linewidth=1.5)

            # Draw vertical ticks at ends
            tick_height = y_range * 0.01
            if position == 'above':
                ax.plot([pos1, pos1], [height, height - tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height - tick_height], 'k-', linewidth=1.5)
                # Add significance symbol above the line
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height, symbol, ha='center', va='bottom',
                       fontsize=12, fontweight='bold')
            else:  # below
                ax.plot([pos1, pos1], [height, height + tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height + tick_height], 'k-', linewidth=1.5)
                # Add significance symbol below the line
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height, symbol, ha='center', va='top',
                       fontsize=12, fontweight='bold')

        # Adjust y-axis to accommodate brackets
        if bracket_heights:
            if position == 'above':
                max_bracket_height = max(h for _, h in bracket_heights)
                new_y_max = max_bracket_height + y_range * 0.03  # Add 3% padding above highest bracket
                ax.set_ylim(y_min, new_y_max)
            else:  # below
                min_bracket_height = min(h for _, h in bracket_heights)
                new_y_min = min_bracket_height - y_range * 0.03  # Add 3% padding below lowest bracket
                ax.set_ylim(new_y_min, y_max)

    def _add_dsc_annotations(self, ax, group_data, hemisphere_positions, approach_order, approach_colors, hemispheres):
        """Add median [IQR] and sample size annotations matching reference style"""
        for hemi_idx, hemisphere in enumerate(hemispheres):
            for approach_idx, approach in enumerate(approach_order):
                if approach_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][approach_idx]

                    # For combined HC hemispheres, get all data regardless of hemisphere
                    if hemisphere == 'Combined':
                        approach_data = group_data[group_data['Approach'] == approach]
                    else:
                        approach_data = group_data[(group_data['Approach'] == approach) &
                                                  (group_data['Hemisphere'] == hemisphere)]

                    if not approach_data.empty and len(approach_data) > 0:
                        values = approach_data['DSC_Volume'].dropna()

                        if len(values) > 0:
                            median = values.median()
                            q1 = values.quantile(0.25)
                            q3 = values.quantile(0.75)
                            iqr = q3 - q1
                            n = len(values)

                            # Median [IQR] annotation above boxes
                            y_max = ax.get_ylim()[1]
                            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                            label_y_offset = y_range * 0.02
                            y_pos = y_max + label_y_offset

                            # Format label: median [IQR] with 4 decimal places (matching reference)
                            label = f'{median:.4f} [{iqr:.4f}]'

                            # Use approach color for labels (matching reference)
                            color = approach_colors[approach]

                            ax.text(box_position, y_pos, label,
                                   ha='center', va='bottom', fontsize=12,
                                   color=color, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           alpha=0.8, edgecolor=color, linewidth=0.8))

                            # Sample size annotation lower with better separation from median/IQR (matching ASSD spacing)
                            y_pos_n = y_max - y_range * 0.06
                            ax.text(box_position, y_pos_n, f'n={n}',
                                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Adjust y-axis limits to accommodate labels (matching reference)
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.15)

    def _create_assd_boxplots(self, approach_order, timestamp):
        """Create ASSD slice-wise boxplots for HC and Patients separately - matching reference style
        For HC: combine left and right hemispheres
        For Patients: keep hemispheres separate
        """
        slice_df = pd.DataFrame(self.slice_wise_data)
        slice_df_plot = slice_df[slice_df['ASSD_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        if len(slice_df_plot) == 0:
            print("    No valid ASSD data for plotting")
            return

        # Define distinct colors for each approach (matching DSC and reference)
        approach_colors = {
            'Thresholding': '#d62728',       # Red
            'CBF': '#1f77b4',                # Blue
            'CBF_T1w': '#ff7f0e',            # Orange
            'CBF_FLAIR': '#2ca02c',          # Green
            'CBF_T1w_FLAIR': '#9467bd'       # Purple
        }

        for group_name in ['HC', 'Patients']:
            group_data = slice_df_plot[slice_df_plot['Group'] == group_name].copy()
            if len(group_data) == 0:
                continue

            # Create figure (matching reference)
            fig, ax = plt.subplots(figsize=(14, 8))

            # Prepare data for grouped plotting
            plot_positions = []
            plot_data_list = []
            plot_colors = []

            position = 0
            hemisphere_positions = {}
            hemisphere_centers = {}

            # For HC: combine hemispheres, for Patients: keep separate
            if group_name == 'HC':
                # HC: Single group combining both hemispheres
                hemispheres = ['Combined']
                hemisphere_positions['Combined'] = []
                start_pos = position

                for approach in approach_order:
                    # Combine data from both hemispheres
                    approach_data = group_data[group_data['Approach'] == approach]

                    if not approach_data.empty:
                        plot_data_list.append(approach_data['ASSD_mm'].values)
                    else:
                        plot_data_list.append([])

                    plot_colors.append(approach_colors.get(approach, '#95a5a6'))
                    plot_positions.append(position)
                    hemisphere_positions['Combined'].append(position)
                    position += 0.6  # Spacing within group (balanced spacing)

                # Calculate center for labeling
                hemisphere_centers['Combined'] = (start_pos + position - 0.6) / 2

            else:
                # Patients: Separate hemispheres
                hemispheres = ['Left', 'Right']

                for hemi_idx, hemisphere in enumerate(hemispheres):
                    hemisphere_positions[hemisphere] = []
                    start_pos = position

                    for approach in approach_order:
                        approach_data = group_data[(group_data['Approach'] == approach) &
                                                  (group_data['Hemisphere'] == hemisphere)]

                        if not approach_data.empty:
                            plot_data_list.append(approach_data['ASSD_mm'].values)
                        else:
                            plot_data_list.append([])

                        plot_colors.append(approach_colors.get(approach, '#95a5a6'))
                        plot_positions.append(position)
                        hemisphere_positions[hemisphere].append(position)
                        position += 0.6  # Spacing within hemisphere (balanced spacing)

                    # Calculate hemisphere center for labeling
                    hemisphere_centers[hemisphere] = (start_pos + position - 0.6) / 2

                    # Add gap between hemispheres
                    if hemi_idx < len(hemispheres) - 1:
                        position += 0.25  # Gap between hemispheres (balanced spacing)

            # Create boxplot (matching reference style)
            bp = ax.boxplot(
                plot_data_list,
                positions=plot_positions,
                notch=False,
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color='black', linewidth=2)
            )

            # Color the boxes
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)

            # Customize the plot (matching reference style)
            if group_name == 'HC':
                ax.set_title(f'{group_name} Test Set Evaluation: Slice-wise ASSD by Segmentation Approach / Input Configuration',
                           fontsize=20, fontweight='bold', pad=25)
                ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=16, fontweight='bold')
            else:
                ax.set_title(f'{group_name} Test Set: Slice-wise ASSD by Segmentation Approach and Hemisphere\n'
                            f'Test Set Evaluation Results',
                           fontsize=20, fontweight='bold', pad=25)
                ax.set_xlabel('Hemisphere', fontsize=16, fontweight='bold')

            ax.set_ylabel('ASSD (mm) per slice', fontsize=16, fontweight='bold')

            # Set custom x-axis labels
            if group_name == 'HC':
                # Custom labels for each approach with descriptive names
                approach_labels = {
                    'Thresholding': 'Thresholding',
                    'CBF': 'nnUNet w/\nCBF',
                    'CBF_T1w': 'nnUNet w/\nCBF+MP-RAGE',
                    'CBF_FLAIR': 'nnUNet w/\nCBF+FLAIR',
                    'CBF_T1w_FLAIR': 'nnUNet w/\nCBF+MP-RAGE+FLAIR'
                }
                tick_positions = [hemisphere_positions['Combined'][i] for i in range(len(approach_order))]
                tick_labels = [approach_labels.get(a, a) for a in approach_order]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
            else:
                ax.set_xticks([hemisphere_centers['Left'], hemisphere_centers['Right']])
                ax.set_xticklabels(['Left', 'Right'])
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)

            # Add grid (matching reference)
            ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
            ax.set_axisbelow(True)

            # Create legend for approaches - only for Patients
            if group_name == 'Patients':
                legend_handles = [plt.Rectangle((0,0),1,1, facecolor=approach_colors[approach],
                                               alpha=0.7, edgecolor='black')
                                 for approach in approach_order]
                ax.legend(legend_handles, [self.approach_display_names.get(a, a) for a in approach_order],
                         title='Segmentation Approach', title_fontsize=12, fontsize=11,
                         loc='upper right', bbox_to_anchor=(1.0, 1.0))
            else:
                # HC: Add legend box with "Median [IQR]" explanation
                ax.text(0.98, 0.98, 'Median [Q1-Q3]\nn = sample size',
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                alpha=0.9, edgecolor='gray', linewidth=1.5))

            # Add median [IQR] and n annotations (matching reference positioning - below boxes)
            self._add_assd_annotations(ax, group_data, hemisphere_positions, approach_order, approach_colors, hemispheres)

            # Add significance brackets (Bonferroni-corrected)
            # For ASSD plots, brackets go below boxes
            if group_name == 'HC':
                # HC: Show significance between approaches (combined hemispheres)
                stats_df = self.statistical_results.get(group_name, None)
                if stats_df is not None:
                    self._add_significance_brackets_approaches(ax, stats_df, hemisphere_positions['Combined'],
                                                              approach_order, position='below')
            else:
                # Patients: Show significance for both hemispheres separately
                stats_df = self.statistical_results.get(group_name, None)
                if stats_df is not None:
                    self._add_significance_brackets_both_hemispheres(ax, stats_df, hemisphere_positions, approach_order,
                                                                     hemispheres, position='below')

            plt.tight_layout()
            plot_file = self.output_dir / f"ASSD_slicewise_{group_name}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    Saved: {plot_file.name}")

    def _add_assd_annotations(self, ax, group_data, hemisphere_positions, approach_order, approach_colors, hemispheres):
        """Add median [IQR] and sample size annotations for ASSD - matching reference style with labels below"""
        for hemi_idx, hemisphere in enumerate(hemispheres):
            for approach_idx, approach in enumerate(approach_order):
                if approach_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][approach_idx]

                    # For combined HC hemispheres, get all data regardless of hemisphere
                    if hemisphere == 'Combined':
                        approach_data = group_data[group_data['Approach'] == approach]
                    else:
                        approach_data = group_data[(group_data['Approach'] == approach) &
                                                  (group_data['Hemisphere'] == hemisphere)]

                    if not approach_data.empty and len(approach_data) > 0:
                        values = approach_data['ASSD_mm'].dropna()

                        if len(values) > 0:
                            median = values.median()
                            q1 = values.quantile(0.25)
                            q3 = values.quantile(0.75)
                            n = len(values)

                            # Median [Q1-Q3] annotation below boxes (matching ASSD reference)
                            y_min = ax.get_ylim()[0]
                            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                            y_pos = y_min - y_offset

                            # Format label: median [Q1-Q3] (matching reference)
                            label = f'{median:.1f} [{q1:.1f}-{q3:.1f}]'

                            # Use approach color for labels (matching reference)
                            color = approach_colors[approach]

                            ax.text(box_position, y_pos, label,
                                   ha='center', va='top', fontsize=12,
                                   color=color, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           alpha=0.9, edgecolor=color, linewidth=0.8))

                            # Sample size annotation further below with more separation
                            y_offset_n = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12
                            y_pos_n = y_min - y_offset_n

                            ax.text(box_position, y_pos_n, f'n={n}',
                                   ha='center', va='top', fontsize=12,
                                   color='black', weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           alpha=0.9, edgecolor='black', linewidth=0.8))

        # Adjust y-axis limits to accommodate labels below (matching reference with more space)
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.25
        ax.set_ylim(new_y_min, current_ylim[1])

    def _create_combined_hc_plot(self, df, approach_order, timestamp):
        """Create a combined plot with DSC, ASSD, RVE, and HD95 for HC group only"""
        # Filter for HC only
        group_data = df[df['Group'] == 'HC'].copy()

        if len(group_data) == 0:
            print("    No HC data available")
            return

        slice_df = pd.DataFrame(self.slice_wise_data)
        slice_df_hc = slice_df[slice_df['Group'] == 'HC'].copy()

        if len(slice_df_hc) == 0:
            print("    No HC slice-wise data available")
            return

        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # Common settings
        approach_colors = {
            'Thresholding': '#d62728',
            'CBF': '#1f77b4',
            'CBF_T1w': '#ff7f0e',
            'CBF_FLAIR': '#2ca02c',
            'CBF_T1w_FLAIR': '#9467bd'
        }

        approach_labels = {
            'Thresholding': 'Thresholding',
            'CBF': 'nnUNet w/\nCBF',
            'CBF_T1w': 'nnUNet w/\nCBF+MP-RAGE',
            'CBF_FLAIR': 'nnUNet w/\nCBF+FLAIR',
            'CBF_T1w_FLAIR': 'nnUNet w/\nCBF+MP-RAGE+FLAIR'
        }

        # ===== TOP LEFT SUBPLOT: DSC =====
        self._plot_hc_dsc_subplot(ax1, group_data, approach_order, approach_colors, approach_labels)

        # ===== TOP RIGHT SUBPLOT: ASSD =====
        self._plot_hc_assd_subplot(ax2, slice_df_hc, approach_order, approach_colors, approach_labels)

        # ===== BOTTOM LEFT SUBPLOT: Relative Volume Error (Thresholding vs CBF only) =====
        self._plot_hc_rve_subplot(ax3, group_data, approach_colors, approach_labels)

        # ===== BOTTOM RIGHT SUBPLOT: HD95 (Thresholding vs CBF only) =====
        self._plot_hc_hd95_subplot(ax4, group_data, approach_colors, approach_labels)

        # Overall title
        fig.suptitle('HC Test Set Evaluation: Segmentation Approach / Input Configuration',
                    fontsize=22, fontweight='bold', y=0.99)

        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle

        plot_file = self.output_dir / f"HC_box-plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"    Saved: {plot_file.name}")

    def _plot_hc_dsc_subplot(self, ax, group_data, approach_order, approach_colors, approach_labels):
        """Plot DSC boxplot for HC in a subplot"""
        # Filter valid DSC values
        df_plot = group_data[group_data['DSC_Volume'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Prepare box plot data
        plot_data_list = []
        plot_positions = []
        plot_colors = []
        position = 0
        box_positions_dict = {}

        for approach in approach_order:
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                plot_data_list.append(approach_data['DSC_Volume'].values)
            else:
                plot_data_list.append([])

            plot_colors.append(approach_colors.get(approach, '#95a5a6'))
            plot_positions.append(position)
            box_positions_dict[approach] = position
            position += 0.6

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(A) DSC per volume', fontsize=18, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=14, fontweight='bold')
        ax.set_ylabel('DSC per volume', fontsize=14, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approach_order])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Y-axis limits
        ax.set_ylim(0.8, 1.0)

        # Add annotations (median, IQR, n)
        for i, approach in enumerate(approach_order):
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                values = approach_data['DSC_Volume'].dropna()
                if len(values) > 0:
                    median = values.median()
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    n = len(values)

                    # Median [IQR]
                    y_max = ax.get_ylim()[1]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    label = f'{median:.4f} [{iqr:.4f}]'
                    color = approach_colors[approach]

                    ax.text(plot_positions[i], y_max - y_range * 0.08, label,
                           ha='center', va='bottom', fontsize=10,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.8, edgecolor=color, linewidth=0.8))

                    # Sample size
                    ax.text(plot_positions[i], y_max - y_range * 0.17, f'n={n}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Add significance brackets if available
        stats_df = self.statistical_results.get('HC', None)
        if stats_df is not None:
            self._add_significance_brackets_approaches(ax, stats_df, list(box_positions_dict.values()),
                                                      approach_order, position='above')

        # Add legend box
        ax.text(0.98, 0.02, 'Median [IQR]\nn = sample size',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_hc_assd_subplot(self, ax, slice_df_hc, approach_order, approach_colors, approach_labels):
        """Plot ASSD boxplot for HC in a subplot"""
        # Filter valid ASSD values
        df_plot = slice_df_hc[slice_df_hc['ASSD_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Prepare box plot data
        plot_data_list = []
        plot_positions = []
        plot_colors = []
        position = 0
        box_positions_dict = {}

        for approach in approach_order:
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                plot_data_list.append(approach_data['ASSD_mm'].values)
            else:
                plot_data_list.append([])

            plot_colors.append(approach_colors.get(approach, '#95a5a6'))
            plot_positions.append(position)
            box_positions_dict[approach] = position
            position += 0.6

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(B) ASSD per slice', fontsize=18, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=14, fontweight='bold')
        ax.set_ylabel('ASSD (mm) per slice', fontsize=14, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approach_order])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add annotations (median, Q1-Q3, n)
        for i, approach in enumerate(approach_order):
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                values = approach_data['ASSD_mm'].dropna()
                if len(values) > 0:
                    median = values.median()
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    n = len(values)

                    # Median [IQR]
                    y_min = ax.get_ylim()[0]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    iqr = q3 - q1
                    label = f'{median:.1f} [{iqr:.1f}]'
                    color = approach_colors[approach]

                    ax.text(plot_positions[i], y_min - y_range * 0.02, label,
                           ha='center', va='top', fontsize=10,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor=color, linewidth=0.8))

                    # Sample size
                    ax.text(plot_positions[i], y_min - y_range * 0.12, f'n={n}',
                           ha='center', va='top', fontsize=10,
                           color='black', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor='black', linewidth=0.8))

        # Adjust y-limits for annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.25
        ax.set_ylim(new_y_min, current_ylim[1])

        # Add significance brackets if available
        stats_df = self.statistical_results.get('HC', None)
        if stats_df is not None:
            self._add_significance_brackets_approaches(ax, stats_df, list(box_positions_dict.values()),
                                                      approach_order, position='below')

        # Add legend box
        ax.text(0.98, 0.98, 'Median [IQR]\nn = sample size',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_hc_rve_subplot(self, ax, group_data, approach_colors, approach_labels):
        """Plot Relative Volume Error boxplot for HC (Thresholding vs CBF only)"""
        # Filter for Thresholding and CBF only
        approaches_to_plot = ['Thresholding', 'CBF']
        df_plot = group_data[group_data['Approach'].isin(approaches_to_plot)].copy()
        df_plot = df_plot[df_plot['RVE_Percent'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Prepare box plot data
        plot_data_list = []
        plot_positions = []
        plot_colors = []
        position = 0

        for approach in approaches_to_plot:
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                plot_data_list.append(approach_data['RVE_Percent'].values)
            else:
                plot_data_list.append([])

            plot_colors.append(approach_colors.get(approach, '#95a5a6'))
            plot_positions.append(position)
            position += 0.8

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(C) Relative Volume Error', fontsize=18, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Relative Volume Error (%)', fontsize=14, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approaches_to_plot])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add annotations (median, IQR, n)
        for i, approach in enumerate(approaches_to_plot):
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                values = approach_data['RVE_Percent'].dropna()
                if len(values) > 0:
                    median = values.median()
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    n = len(values)

                    # Median [IQR]
                    y_min = ax.get_ylim()[0]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    label = f'{median:.1f} [{iqr:.1f}]'
                    color = approach_colors[approach]

                    ax.text(plot_positions[i], y_min - y_range * 0.02, label,
                           ha='center', va='top', fontsize=10,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor=color, linewidth=0.8))

                    # Sample size
                    ax.text(plot_positions[i], y_min - y_range * 0.12, f'n={n}',
                           ha='center', va='top', fontsize=10,
                           color='black', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor='black', linewidth=0.8))

        # Adjust y-limits for annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.25
        ax.set_ylim(new_y_min, current_ylim[1])

        # Add legend box (top right)
        ax.text(0.98, 0.98, 'Median [IQR]\nn = sample size',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_hc_hd95_subplot(self, ax, group_data, approach_colors, approach_labels):
        """Plot HD95 boxplot for HC (Thresholding vs CBF only)"""
        # Filter for Thresholding and CBF only
        approaches_to_plot = ['Thresholding', 'CBF']
        df_plot = group_data[group_data['Approach'].isin(approaches_to_plot)].copy()
        df_plot = df_plot[df_plot['HD95_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Prepare box plot data
        plot_data_list = []
        plot_positions = []
        plot_colors = []
        position = 0

        for approach in approaches_to_plot:
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                plot_data_list.append(approach_data['HD95_mm'].values)
            else:
                plot_data_list.append([])

            plot_colors.append(approach_colors.get(approach, '#95a5a6'))
            plot_positions.append(position)
            position += 0.8

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(D) 95th Percentile Hausdorff Distance per volume', fontsize=18, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=14, fontweight='bold')
        ax.set_ylabel('HD95 (mm) per volume', fontsize=14, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approaches_to_plot])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add annotations (median, IQR, n)
        for i, approach in enumerate(approaches_to_plot):
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                values = approach_data['HD95_mm'].dropna()
                if len(values) > 0:
                    median = values.median()
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    n = len(values)

                    # Median [IQR]
                    y_min = ax.get_ylim()[0]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    label = f'{median:.1f} [{iqr:.1f}]'
                    color = approach_colors[approach]

                    ax.text(plot_positions[i], y_min - y_range * 0.02, label,
                           ha='center', va='top', fontsize=10,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor=color, linewidth=0.8))

                    # Sample size
                    ax.text(plot_positions[i], y_min - y_range * 0.12, f'n={n}',
                           ha='center', va='top', fontsize=10,
                           color='black', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor='black', linewidth=0.8))

        # Adjust y-limits for annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.25
        ax.set_ylim(new_y_min, current_ylim[1])

        # Add legend box (top right)
        ax.text(0.98, 0.98, 'Median [IQR]\nn = sample size',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

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
    # Define all 5 prediction directories for all segmentation approaches
    predictions_dirs = {
        'Thresholding': "/home/ubuntu/DLSegPerf/data/other/thresholding_results/thresholded_labelsTs",
        'CBF': "/home/ubuntu/DLSegPerf/data/other/nnUNet_results_Single-class_CBF/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/fold_all/predictions",
        'CBF_T1w': "/home/ubuntu/DLSegPerf/data/other/nnUNet_results_Single-class_CBF_T1w/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/fold_all/predictions",
        'CBF_FLAIR': "/home/ubuntu/DLSegPerf/data/other/nnUNet_results_Single-class_CBF_FLAIR/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/fold_all/predictions",
        'CBF_T1w_FLAIR': "/home/ubuntu/DLSegPerf/data/other/nnUNet_results_Single-class_CBF_T1w_FLAIR/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/fold_all/predictions"
    }

    default_gt = "/home/ubuntu/DLSegPerf/data/other/GroundTruthMasks"
    default_output = "/home/ubuntu/DLSegPerf/model_evaluation/test_evaluation/results"

    # Parse command line arguments - support backwards compatibility
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # If it looks like a single directory path, use backwards compatibility mode
        if Path(first_arg).exists():
            predictions_dirs = {'single': first_arg}

    gt_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_gt
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else default_output

    print("Test Set Evaluation Script - All Segmentation Approaches")
    print("=" * 70)
    print("Segmentation approaches:")
    print("  Thresholding")
    print("  CBF (CBF LICA + CBF RICA)")
    print("  CBF+T1w (CBF LICA + CBF RICA + T1w)")
    print("  CBF+FLAIR (CBF LICA + CBF RICA + FLAIR)")
    print("  CBF+T1w+FLAIR (CBF LICA + CBF RICA + T1w + FLAIR)")
    print(f"\nGround truth: {gt_dir}")
    print(f"Output dir: {output_dir}\n")

    evaluator = TestSetEvaluator(predictions_dirs, gt_dir, output_dir)
    success = evaluator.run_evaluation()

    if success:
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved in: {output_dir}")

        # Print approach comparison summary
        if evaluator.results:
            import pandas as pd
            import numpy as np

            df = pd.DataFrame(evaluator.results)
            print("\nAPPROACH PERFORMANCE SUMMARY:")
            print("-" * 60)

            for approach in ['Thresholding', 'CBF', 'CBF_T1w', 'CBF_FLAIR', 'CBF_T1w_FLAIR']:
                if approach in df['Approach'].unique():
                    approach_data = df[df['Approach'] == approach]
                    dice_vals = approach_data['DSC_Volume'].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(dice_vals) > 0:
                        display_name = evaluator.approach_display_names.get(approach, approach)
                        print(f"{display_name:20}: {dice_vals.mean():.4f} ± {dice_vals.std():.4f} DSC ({len(approach_data)} cases)")

        return 0
    else:
        print("Evaluation failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)