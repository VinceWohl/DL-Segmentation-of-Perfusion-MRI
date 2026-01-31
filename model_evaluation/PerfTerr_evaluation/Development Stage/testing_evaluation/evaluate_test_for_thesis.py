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
    import matplotlib as mpl
    import seaborn as sns
    PLOTTING_AVAILABLE = True

    # Configure Times New Roman font for thesis
    # Use Liberation Serif as Times New Roman equivalent on Linux
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
    })
except ImportError:
    PLOTTING_AVAILABLE = False


class TestSetEvaluator:
    def __init__(self, predictions_dirs, gt_dir, output_dir, excel_data_dir=None):
        # Handle both single path (backward compatibility) and multiple paths
        if isinstance(predictions_dirs, (str, Path)):
            # Single prediction directory - backward compatibility
            self.predictions_dirs = {'single': Path(predictions_dirs)}
        else:
            # Multiple prediction directories
            self.predictions_dirs = {name: Path(path) for name, path in predictions_dirs.items()}

        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Directory containing pre-computed Excel results (if provided, skip raw computation)
        self.excel_data_dir = Path(excel_data_dir) if excel_data_dir else None

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
            'CBF': 'Perf.',
            'CBF_T1w': 'Perf.+T1w',
            'CBF_FLAIR': 'Perf.+FLAIR',
            'CBF_T1w_FLAIR': 'Perf.+T1w+FLAIR'
        }

        # Mapping for Excel file naming
        self.approach_file_names = {
            'Thresholding': 'Thresholding',
            'CBF': 'Single-class_CBF',
            'CBF_T1w': 'Single-class_CBF_T1w',
            'CBF_FLAIR': 'Single-class_CBF_FLAIR',
            'CBF_T1w_FLAIR': 'Single-class_CBF_T1w_FLAIR'
        }

        # Reverse mapping for loading from Excel
        self.file_names_to_approach = {v: k for k, v in self.approach_file_names.items()}

    def load_results_from_excel(self):
        """Load pre-computed results from existing Excel files instead of computing from raw data"""
        print("Loading Results from Pre-computed Excel Files")
        print("=" * 70)
        print(f"Excel data directory: {self.excel_data_dir}")

        if not self.excel_data_dir.exists():
            print(f"ERROR: Excel data directory not found: {self.excel_data_dir}")
            return False

        # Define the approaches and groups to load
        approaches = ['Thresholding', 'Single-class_CBF', 'Single-class_CBF_T1w',
                     'Single-class_CBF_FLAIR', 'Single-class_CBF_T1w_FLAIR']
        groups = ['HC', 'patients']

        # Find Excel files matching the expected pattern
        loaded_count = 0
        for group in groups:
            for approach_file_name in approaches:
                # Find matching Excel file with exact approach name
                # Pattern requires timestamp (digits) after approach name to avoid partial matches
                pattern = f"test_results_{group}_{approach_file_name}_[0-9]*.xlsx"
                matching_files = list(self.excel_data_dir.glob(pattern))

                if not matching_files:
                    print(f"  Warning: No file found for {group}/{approach_file_name}")
                    continue

                # Filter to ensure exact match (avoid Single-class_CBF matching Single-class_CBF_T1w)
                exact_matches = []
                for f in matching_files:
                    # Extract the approach part from filename
                    fname = f.name
                    expected_prefix = f"test_results_{group}_{approach_file_name}_"
                    if fname.startswith(expected_prefix):
                        # Check that what follows is a timestamp (digits)
                        remainder = fname[len(expected_prefix):]
                        if remainder and remainder[0].isdigit():
                            exact_matches.append(f)

                if not exact_matches:
                    print(f"  Warning: No exact match found for {group}/{approach_file_name}")
                    continue

                # Use the most recent file if multiple exist
                excel_file = sorted(exact_matches)[-1]
                print(f"  Loading: {excel_file.name}")

                try:
                    # Load Per_Case_Details sheet
                    df = pd.read_excel(excel_file, sheet_name='Per_Case_Details')

                    # Map file name back to internal approach key
                    approach_key = self.file_names_to_approach.get(approach_file_name, approach_file_name)

                    # Convert each row to a result dictionary
                    for _, row in df.iterrows():
                        result = {
                            'Subject': row['Subject'],
                            'Visit': row['Visit'],
                            'Hemisphere': row['Hemisphere'],
                            'Base_Name': row['Base_Name'],
                            'DSC_Volume': row['DSC_Volume'],
                            'DSC_Slicewise': row.get('DSC_Slicewise', row['DSC_Volume']),
                            'IoU': row.get('IoU', 0),
                            'Sensitivity': row.get('Sensitivity', 0),
                            'Precision': row.get('Precision', 0),
                            'Specificity': row.get('Specificity', 0),
                            'RVE_Percent': row['RVE_Percent'],
                            'HD95_mm': row['HD95_mm'],
                            'ASSD_mm': row['ASSD_mm'],
                            'Approach': approach_key,
                            'Group': 'HC' if group == 'HC' else 'Patients'
                        }
                        self.results.append(result)

                        # Also create slice-wise data entry for ASSD plotting
                        # (using the per-volume ASSD as representative value)
                        slice_data = {
                            'Subject': row['Subject'],
                            'Visit': row['Visit'],
                            'Hemisphere': row['Hemisphere'],
                            'Base_Name': row['Base_Name'],
                            'Case': row['Base_Name'],  # Alias for compatibility with plotting code
                            'ASSD_mm': row['ASSD_mm'],
                            'Approach': approach_key,
                            'Group': 'HC' if group == 'HC' else 'Patients'
                        }
                        self.slice_wise_data.append(slice_data)

                        self.processed_count += 1
                        self.successful_count += 1

                    loaded_count += 1

                except Exception as e:
                    print(f"  Error loading {excel_file.name}: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"\nLoaded {self.successful_count} data points from {loaded_count} Excel files")
        print(f"  HC datapoints: {len([r for r in self.results if r.get('Group') == 'HC'])}")
        print(f"  Patient datapoints: {len([r for r in self.results if r.get('Group') == 'Patients'])}")

        return loaded_count > 0

    def run_evaluation_from_excel(self):
        """Run evaluation using pre-computed results from Excel files"""
        print("Test Set Results Evaluation - Loading from Pre-computed Excel Files")
        print("=" * 70)
        print(f"Excel data directory: {self.excel_data_dir}")
        print(f"Output directory: {self.output_dir}\n")

        # Check dependencies
        if not PANDAS_AVAILABLE:
            print("ERROR: pandas package is required. Please install with: pip install pandas openpyxl")
            return False
        if not PLOTTING_AVAILABLE:
            print("WARNING: matplotlib/seaborn not available. Plots will be skipped.")

        # Load results from Excel files
        if not self.load_results_from_excel():
            print("ERROR: Failed to load results from Excel files")
            return False

        if not self.results:
            print("ERROR: No results loaded from Excel files")
            return False

        print("\nPerforming statistical testing (volume-based)...")
        # Skip slice-wise statistical testing (perform_statistical_testing) since we have volume-based data
        self.perform_volumewise_statistical_testing()
        self.perform_patient_volumewise_statistical_testing()

        print("\nCreating box plots...")
        self.create_box_plots()

        self.print_evaluation_summary()
        return True

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
        # Check if we should load from Excel files instead of computing from raw data
        if self.excel_data_dir is not None:
            return self.run_evaluation_from_excel()

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
        self.perform_volumewise_statistical_testing()
        self.perform_patient_volumewise_statistical_testing()

        print("\nCreating box plots...")
        self.create_box_plots()

        self.print_evaluation_summary()
        return True

    def create_excel_for_approach_and_group(self, df, approach, group_name, timestamp):
        """Create Excel file for a specific approach and group (HC or Patients)"""
        # Get file name components
        approach_file_name = self.approach_file_names.get(approach, approach)
        excel_file = self.output_dir / f"test_results_{group_name}_{approach_file_name}_for_thesis_{timestamp}.xlsx"

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

            # Re-apply Times New Roman font after style reset
            mpl.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Liberation Serif', 'DejaVu Serif'],
                'mathtext.fontset': 'stix',
            })

            # Get sorted approach order for consistent plotting
            approach_order = sorted(df['Approach'].unique(),
                                   key=lambda x: ['Thresholding', 'CBF', 'CBF_T1w', 'CBF_FLAIR', 'CBF_T1w_FLAIR'].index(x)
                                   if x in ['Thresholding', 'CBF', 'CBF_T1w', 'CBF_FLAIR', 'CBF_T1w_FLAIR'] else 999)

            # ====== Plot 1: Combined HC plot (DSC + ASSD + RVE + HD95) ======
            if self.slice_wise_data:
                print("  Creating combined HC plot...")
                self._create_combined_hc_plot(df, approach_order, timestamp)
            else:
                print("  No slice-wise ASSD data available for plotting")

            # ====== Plot 2: Combined Patient plot (DSC + RVE + ASSD + HD95) ======
            print("  Creating combined Patient plot...")
            self._create_combined_patient_plot(df, timestamp)

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

        # Save to Excel with ASSD metric name
        stats_file = self.output_dir / f"test_statistical_comparison_{group_name}_ASSD_for_thesis_{timestamp}.xlsx"

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

    def perform_volumewise_statistical_testing(self):
        """Perform Wilcoxon signed-rank test for volume-wise metrics (DSC, RVE, HD95)"""
        from scipy import stats
        from itertools import combinations

        if not self.results:
            print("  No volume-wise data available for statistical testing")
            return

        print("\n  Performing Volume-wise Statistical Testing (Wilcoxon signed-rank test)...")

        df = pd.DataFrame(self.results)
        df[['Subject_str', 'Visit', 'Hemisphere_Code']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
        df['Subject_Num'] = df['Subject_str'].astype(int)
        df['Group'] = df['Subject_Num'].apply(lambda x: 'HC' if x in [14, 15] else 'Patients')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Process HC group only for now
        for group_name in ['HC']:
            group_data = df[df['Group'] == group_name].copy()

            if len(group_data) == 0:
                print(f"  No data for {group_name}")
                continue

            print(f"\n  {group_name} Volume-wise Statistical Analysis:")
            print(f"  {'-' * 40}")

            # Test each metric separately
            for metric_name, metric_col in [('DSC', 'DSC_Volume'), ('RVE', 'RVE_Percent'), ('ASSD', 'ASSD_mm'), ('HD95', 'HD95_mm')]:
                print(f"\n    Testing {metric_name}...")

                # Filter valid values for this metric
                metric_data = group_data[group_data[metric_col].replace([np.inf, -np.inf], np.nan).notna()].copy()

                if len(metric_data) == 0:
                    print(f"      No valid {metric_name} data")
                    continue

                all_stats_results = []
                available_approaches = metric_data['Approach'].unique().tolist()

                if len(available_approaches) < 2:
                    print(f"      Insufficient approaches for comparison")
                    continue

                # Perform all pairwise comparisons across approaches
                for approach1, approach2 in combinations(available_approaches, 2):
                    data1_df = metric_data[metric_data['Approach'] == approach1]
                    data2_df = metric_data[metric_data['Approach'] == approach2]

                    # Find paired volumes (same case)
                    merged_df = data1_df.merge(data2_df, on='Base_Name', suffixes=('_1', '_2'))

                    if len(merged_df) < 5:  # Minimum sample size
                        continue

                    # Get paired volume-wise data
                    paired_data1 = merged_df[f'{metric_col}_1'].values
                    paired_data2 = merged_df[f'{metric_col}_2'].values

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
                            'Metric': metric_name,
                            'Config1': approach1,
                            'Config2': approach2,
                            'Median1': median1,
                            'Median2': median2,
                            'Median_Diff': median_diff,
                            'Statistic': statistic,
                            'P_Value': p_value,
                            'Effect_Size': effect_size,
                            'Significance': significance,
                            'N_Paired_Volumes': n,
                            'N1_Total_Volumes': len(data1_df),
                            'N2_Total_Volumes': len(data2_df)
                        })

                    except Exception as e:
                        print(f"        Error comparing {approach1} vs {approach2}: {e}")

                # Save results for this metric
                if all_stats_results:
                    stats_df = self._save_volumewise_statistical_results(all_stats_results, group_name, metric_name, timestamp)
                    # Store for significance brackets in plots
                    if not hasattr(self, 'volumewise_statistical_results'):
                        self.volumewise_statistical_results = {}
                    self.volumewise_statistical_results[f'{group_name}_{metric_name}'] = stats_df
                else:
                    print(f"      No statistical comparisons performed for {metric_name}")

    def _save_volumewise_statistical_results(self, all_stats_results, group_name, metric_name, timestamp):
        """Save volume-wise statistical test results to Excel file"""
        stats_df = pd.DataFrame(all_stats_results)

        # Sort by p-value
        stats_df = stats_df.sort_values(['P_Value'])

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

        # Apply Bonferroni correction
        n_comparisons = len(stats_df)
        stats_df['P_Value_Bonferroni'] = stats_df['P_Value'] * n_comparisons
        stats_df['P_Value_Bonferroni'] = np.minimum(stats_df['P_Value_Bonferroni'], 1.0)

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

        stats_df['Significance_Bonferroni'] = stats_df['P_Value_Bonferroni'].apply(get_bonferroni_significance)

        # Save to Excel
        stats_file = self.output_dir / f"test_statistical_comparison_{group_name}_{metric_name}_for_thesis_{timestamp}.xlsx"

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

        print(f"      {metric_name} results saved: {stats_file.name}")
        print(f"        Total comparisons: {len(stats_df)}")
        print(f"        Significant (p<0.05): {sum(stats_df['P_Value'] < 0.05)}")
        print(f"        Bonferroni significant: {sum(stats_df['P_Value_Bonferroni'] < 0.05)}")

        # Return the DataFrame for use in plotting
        return stats_df

    def perform_patient_volumewise_statistical_testing(self):
        """Perform Wilcoxon signed-rank test for patient volume-wise metrics (DSC, RVE, HD95)
        comparing Thresholding vs Perf. across ipsilateral and contralateral hemispheres"""
        from scipy import stats
        import pandas as pd

        if not self.results:
            print("  No volume-wise data available for patient statistical testing")
            return

        print("\n  Performing Patient Volume-wise Statistical Testing (Thresholding vs Perf.)...")

        df = pd.DataFrame(self.results)
        df[['Subject_str', 'Visit', 'Hemisphere_Code']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
        df['Subject_Num'] = df['Subject_str'].astype(int)
        df['Group'] = df['Subject_Num'].apply(lambda x: 'HC' if x in [14, 15] else 'Patients')

        # Get patient data only
        patient_data = df[df['Group'] == 'Patients'].copy()

        if len(patient_data) == 0:
            print("  No patient data available")
            return

        # Load pathology mapping to categorize hemispheres
        pathology_file = self.output_dir / 'data_completeness_report_20250820_171756.xlsx'
        try:
            pathology_df = pd.read_excel(pathology_file)
            patient_mapping = pathology_df[pathology_df['Subject'].str.match('sub-p0(1[6-9]|2[0-3])', na=False)].copy()
            patient_mapping = patient_mapping[patient_mapping['AVM/ICAS side'] != 'x']
            patient_mapping['PerfTerr_ID'] = patient_mapping['Subject'].str.replace('sub-p0', 'PerfTerr0')
            visit_map = {'First_visit': 'v1', 'Second_visit': 'v2', 'Third_visit': 'v3'}
            patient_mapping['Visit_Code'] = patient_mapping['Visit'].map(visit_map)

            pathology_lookup = {}
            for _, row in patient_mapping.iterrows():
                key = f"{row['PerfTerr_ID']}-{row['Visit_Code']}"
                pathology_lookup[key] = row['AVM/ICAS side']

            # Add hemisphere categorization
            def categorize_hemisphere(row):
                case_key = row['Base_Name'].rsplit('-', 1)[0]  # Remove -L or -R
                pathology_side = pathology_lookup.get(case_key, None)
                if pathology_side is None:
                    return None
                actual_hemi = 'left' if row['Base_Name'].endswith('-L') else 'right'
                return 'Ipsilateral' if actual_hemi == pathology_side else 'Contralateral'

            patient_data['Hemisphere_Category'] = patient_data.apply(categorize_hemisphere, axis=1)
            patient_data = patient_data[patient_data['Hemisphere_Category'].notna()].copy()

            # Exclude PerfTerr022-v1-L specifically
            patient_data = patient_data[patient_data['Base_Name'] != 'PerfTerr022-v1-L'].copy()

        except Exception as e:
            print(f"  Error loading pathology mapping: {e}")
            return

        # Filter to only Thresholding and CBF
        patient_data = patient_data[patient_data['Approach'].isin(['Thresholding', 'CBF'])].copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n  Patient Statistical Analysis (Ipsilateral vs Contralateral):")
        print(f"  {'-' * 60}")

        # Test each metric separately
        for metric_name, metric_col in [('DSC', 'DSC_Volume'), ('RVE', 'RVE_Percent'), ('ASSD', 'ASSD_mm'), ('HD95', 'HD95_mm')]:
            print(f"\n    Testing {metric_name}...")

            metric_data = patient_data[patient_data[metric_col].replace([np.inf, -np.inf], np.nan).notna()].copy()

            if len(metric_data) == 0:
                print(f"      No valid {metric_name} data")
                continue

            all_stats_results = []

            # Test for each hemisphere category
            for hemi_cat in ['Ipsilateral', 'Contralateral']:
                hemi_data = metric_data[metric_data['Hemisphere_Category'] == hemi_cat]

                # Get Thresholding and CBF data
                thresh_data = hemi_data[hemi_data['Approach'] == 'Thresholding']
                cbf_data = hemi_data[hemi_data['Approach'] == 'CBF']

                # Find paired volumes (same case)
                merged_df = thresh_data.merge(cbf_data, on='Base_Name', suffixes=('_Thresh', '_CBF'))

                if len(merged_df) < 5:  # Minimum sample size
                    print(f"      {hemi_cat}: Insufficient paired data (n={len(merged_df)})")
                    continue

                # Get paired data
                paired_thresh = merged_df[f'{metric_col}_Thresh'].values
                paired_cbf = merged_df[f'{metric_col}_CBF'].values

                try:
                    # Wilcoxon signed-rank test
                    statistic, p_value = stats.wilcoxon(paired_thresh, paired_cbf, alternative='two-sided')

                    # Calculate effect size
                    n = len(paired_thresh)
                    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0
                    effect_size = z_score / np.sqrt(n)

                    # Calculate medians
                    median_thresh = np.median(paired_thresh)
                    median_cbf = np.median(paired_cbf)
                    median_diff = median_thresh - median_cbf

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
                        'Hemisphere_Category': hemi_cat,
                        'Approach1': 'Thresholding',
                        'Approach2': 'CBF',
                        'Median_Thresholding': median_thresh,
                        'Median_CBF': median_cbf,
                        'Median_Diff': median_diff,
                        'Statistic': statistic,
                        'P_Value': p_value,
                        'Effect_Size': effect_size,
                        'Significance': significance,
                        'N_Paired': n,
                        'N_Thresholding': len(thresh_data),
                        'N_CBF': len(cbf_data)
                    })

                    print(f"      {hemi_cat}: n={n}, p={p_value:.4f} {significance}")

                except Exception as e:
                    print(f"      Error testing {hemi_cat}: {e}")

            # Save results if we have any
            if all_stats_results:
                stats_df = pd.DataFrame(all_stats_results)

                # Add effect size interpretation
                stats_df['Effect_Size_Interpretation'] = stats_df['Effect_Size'].apply(
                    lambda x: 'Large' if abs(x) >= 0.5 else ('Medium' if abs(x) >= 0.3 else 'Small')
                )

                # No Bonferroni correction needed (only 1 comparison per hemisphere category)
                stats_df['P_Value_Bonferroni'] = stats_df['P_Value']
                stats_df['Significance_Bonferroni'] = stats_df['Significance']

                # Save to Excel
                stats_file = self.output_dir / f"test_statistical_comparison_Patients_{metric_name}_for_thesis_{timestamp}.xlsx"
                with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
                    stats_df.to_excel(writer, sheet_name='All_Comparisons', index=False)

                    if sum(stats_df['P_Value'] < 0.05) > 0:
                        significant_df = stats_df[stats_df['P_Value'] < 0.05]
                        significant_df.to_excel(writer, sheet_name='Significant', index=False)

                print(f"      {metric_name} results saved: {stats_file.name}")
                print(f"        Significant (p<0.05): {sum(stats_df['P_Value'] < 0.05)}/{len(stats_df)}")

                # Store for plotting
                if not hasattr(self, 'patient_statistical_results'):
                    self.patient_statistical_results = {}
                self.patient_statistical_results[metric_name] = stats_df

    def _add_significance_brackets_approaches(self, ax, stats_df, approach_positions, approach_order, position='above', y_offset=0.0):
        """Add significance brackets for approach comparisons (used for HC combined hemispheres)

        Args:
            ax: matplotlib axis
            stats_df: DataFrame with statistical results
            approach_positions: list of x-positions for each approach
            approach_order: list of approach names in order
            position: 'above' or 'below' - where to place brackets relative to boxes
            y_offset: additional y-axis offset (fraction of y_range, negative=lower, positive=higher)
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
                    height = y_max + bracket_base_offset + (level * bracket_height_increment) + (y_offset * y_range)
                else:  # below
                    height = y_min - bracket_base_offset - (level * bracket_height_increment) + (y_offset * y_range)

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
            'CBF': 'nnUNet w/\nPerf.',
            'CBF_T1w': 'nnUNet w/\nPerf.\n+MP-RAGE',
            'CBF_FLAIR': 'nnUNet w/\nPerf.\n+FLAIR',
            'CBF_T1w_FLAIR': 'nnUNet w/\nPerf.\n+MP-RAGE+FLAIR'
        }

        # ===== TOP LEFT SUBPLOT: DSC =====
        self._plot_hc_dsc_subplot(ax1, group_data, approach_order, approach_colors, approach_labels)

        # ===== TOP RIGHT SUBPLOT: Relative Volume Error (all approaches) =====
        self._plot_hc_rve_subplot(ax2, group_data, approach_order, approach_colors, approach_labels)

        # ===== BOTTOM LEFT SUBPLOT: ASSD =====
        self._plot_hc_assd_subplot(ax3, slice_df_hc, approach_order, approach_colors, approach_labels)

        # ===== BOTTOM RIGHT SUBPLOT: HD95 (all approaches) =====
        self._plot_hc_hd95_subplot(ax4, group_data, approach_order, approach_colors, approach_labels)

        # Overall title
        fig.suptitle('HC Test Set Evaluation: Input Configuration',
                    fontsize=24, fontweight='bold', y=0.99)

        plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4, w_pad=2.5)  # Reduced horizontal spacing

        fig.subplots_adjust(hspace=0.33)  # Spacing between upper and lower subplots

        plot_file = self.output_dir / f"HC_box-plots_for_thesis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"    Saved: {plot_file.name}")

    def _create_combined_patient_plot(self, df, timestamp):
        """Create a combined plot with DSC, RVE, ASSD, and HD95 for Patients
        comparing Thresholding vs nnUNet w/ CBF
        Grouped by Ipsilateral (pathological) vs Contralateral (healthy) hemispheres
        """
        import pandas as pd

        # Load pathology mapping
        pathology_file = self.output_dir / 'data_completeness_report_20250820_171756.xlsx'
        pathology_df = pd.read_excel(pathology_file)

        # Filter and map patients
        patient_mapping = pathology_df[pathology_df['Subject'].str.match('sub-p0(1[6-9]|2[0-3])', na=False)].copy()
        patient_mapping = patient_mapping[patient_mapping['AVM/ICAS side'] != 'x']
        patient_mapping['PerfTerr_ID'] = patient_mapping['Subject'].str.replace('sub-p0', 'PerfTerr0')
        visit_map = {'First_visit': 'v1', 'Second_visit': 'v2', 'Third_visit': 'v3'}
        patient_mapping['Visit_Code'] = patient_mapping['Visit'].map(visit_map)

        # Create lookup dictionary
        pathology_lookup = {}
        for _, row in patient_mapping.iterrows():
            key = f"{row['PerfTerr_ID']}-{row['Visit_Code']}"
            pathology_lookup[key] = row['AVM/ICAS side']  # 'left' or 'right'

        # Filter patient data (PerfTerr017-023, excluding 016, 021, and PerfTerr022-v1-L)
        patient_df = df[df['Base_Name'].str.contains('PerfTerr0(17|18|19|20|22|23)', regex=True)].copy()

        # Exclude PerfTerr022-v1-L specifically
        patient_df = patient_df[patient_df['Base_Name'] != 'PerfTerr022-v1-L'].copy()

        # Only keep Thresholding and CBF approaches
        patient_df = patient_df[patient_df['Approach'].isin(['Thresholding', 'CBF'])].copy()

        if len(patient_df) == 0:
            print("    No patient data available")
            return

        # Add pathology side and categorize as ipsilateral/contralateral
        def categorize_hemisphere(row):
            case_key = f"{row['Base_Name'].split('-L')[0].split('-R')[0]}"
            pathology_side = pathology_lookup.get(case_key, None)

            if pathology_side is None:
                return None

            # Determine actual hemisphere from filename
            if '-L' in row['Base_Name']:
                actual_hemi = 'left'
            elif '-R' in row['Base_Name']:
                actual_hemi = 'right'
            else:
                return None

            # Categorize as ipsilateral (pathological) or contralateral (healthy)
            if actual_hemi == pathology_side:
                return 'Ipsilateral'
            else:
                return 'Contralateral'

        patient_df['Hemisphere_Category'] = patient_df.apply(categorize_hemisphere, axis=1)
        patient_df = patient_df[patient_df['Hemisphere_Category'].notna()].copy()

        if len(patient_df) == 0:
            print("    No patient data after categorization")
            return

        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # Common settings
        approach_colors = {
            'Thresholding': '#d62728',
            'CBF': '#1f77b4'
        }

        approach_labels = {
            'Thresholding': 'Thresholding',
            'CBF': 'nnUNet w/\nPerf.'
        }

        # ===== TOP LEFT SUBPLOT: DSC =====
        self._plot_patient_dsc_subplot(ax1, patient_df, approach_colors, approach_labels)

        # ===== TOP RIGHT SUBPLOT: Relative Volume Error =====
        self._plot_patient_rve_subplot(ax2, patient_df, approach_colors, approach_labels)

        # ===== BOTTOM LEFT SUBPLOT: ASSD (slice-wise) =====
        # Prepare slice-wise patient data with ipsi/contra categorization
        slice_df = pd.DataFrame(self.slice_wise_data)
        patient_slice_df = slice_df[slice_df['Case'].str.contains('PerfTerr0(17|18|19|20|22|23)', regex=True)].copy()

        # Exclude PerfTerr022-v1-L specifically
        patient_slice_df = patient_slice_df[patient_slice_df['Case'] != 'PerfTerr022-v1-L'].copy()

        patient_slice_df = patient_slice_df[patient_slice_df['Approach'].isin(['Thresholding', 'CBF'])].copy()

        # Add hemisphere categorization to slice data
        def categorize_hemisphere_slice(row):
            case_key = f"{row['Case'].split('-L')[0].split('-R')[0]}"
            pathology_side = pathology_lookup.get(case_key, None)

            if pathology_side is None:
                return None

            # Get hemisphere from row (already has 'Left'/'Right')
            actual_hemi = 'left' if row['Hemisphere'] == 'Left' else 'right'

            # Categorize as ipsilateral (pathological) or contralateral (healthy)
            if actual_hemi == pathology_side:
                return 'Ipsilateral'
            else:
                return 'Contralateral'

        patient_slice_df['Hemisphere_Category'] = patient_slice_df.apply(categorize_hemisphere_slice, axis=1)
        patient_slice_df = patient_slice_df[patient_slice_df['Hemisphere_Category'].notna()].copy()

        self._plot_patient_assd_subplot(ax3, patient_slice_df, approach_colors, approach_labels)

        # ===== BOTTOM RIGHT SUBPLOT: HD95 =====
        self._plot_patient_hd95_subplot(ax4, patient_df, approach_colors, approach_labels)

        # Overall title
        fig.suptitle('Patients Test Set Evaluation: Segmentation Approach',
                    fontsize=24, fontweight='bold', y=0.99)

        plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4, w_pad=4)

        plot_file = self.output_dir / f"Patients_box-plots_for_thesis_{timestamp}.png"
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
            position += 1.0

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.7,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(A) Dice Similarity Coefficient (n=12)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=18, fontweight='bold')
        ax.set_ylabel('Dice per volume', fontsize=18, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approach_order])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

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
                    label = f'{median:.3f} [{iqr:.3f}]'
                    color = approach_colors[approach]

                    ax.text(plot_positions[i], y_max - y_range * 0.12, label,
                           ha='center', va='bottom', fontsize=13,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.8, edgecolor=color, linewidth=0.8))

        # Add significance brackets if available
        # First try volume-wise results, then fall back to slice-wise
        stats_df = None
        if hasattr(self, 'volumewise_statistical_results'):
            stats_df = self.volumewise_statistical_results.get('HC_DSC', None)
        if stats_df is None:
            stats_df = self.statistical_results.get('HC', None)
        if stats_df is not None:
            self._add_significance_brackets_approaches(ax, stats_df, list(box_positions_dict.values()),
                                                      approach_order, position='above', y_offset=-0.03)

        # Add legend box
        ax.text(0.98, 0.02, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
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
            position += 1.0

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.7,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(C) Average Symmetric Surface Distance (n=12)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=18, fontweight='bold')
        ax.set_ylabel('ASSD (mm) per slice', fontsize=18, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approach_order])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

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
                           ha='center', va='top', fontsize=13,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor=color, linewidth=0.8))

        # Adjust y-limits for annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.25
        ax.set_ylim(new_y_min, current_ylim[1])

        # Add significance brackets if available
        # First try volume-wise results, then fall back to slice-wise
        stats_df = None
        if hasattr(self, 'volumewise_statistical_results'):
            stats_df = self.volumewise_statistical_results.get('HC_ASSD', None)
        if stats_df is None:
            stats_df = self.statistical_results.get('HC', None)
        if stats_df is not None:
            self._add_significance_brackets_approaches(ax, stats_df, list(box_positions_dict.values()),
                                                      approach_order, position='below', y_offset=0.05)

        # Add legend box
        ax.text(0.98, 0.98, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_hc_rve_subplot(self, ax, group_data, approach_order, approach_colors, approach_labels):
        """Plot Relative Volume Error boxplot for HC (all approaches)"""
        # Filter valid RVE values
        df_plot = group_data[group_data['RVE_Percent'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Prepare box plot data
        plot_data_list = []
        plot_positions = []
        plot_colors = []
        position = 0

        for approach in approach_order:
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                plot_data_list.append(approach_data['RVE_Percent'].values)
            else:
                plot_data_list.append([])

            plot_colors.append(approach_colors.get(approach, '#95a5a6'))
            plot_positions.append(position)
            position += 1.0

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.7,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(B) Relative Volume Error (n=12)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=18, fontweight='bold')
        ax.set_ylabel('RVE (%) per volume', fontsize=18, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approach_order])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add annotations (median, IQR, n)
        for i, approach in enumerate(approach_order):
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
                           ha='center', va='top', fontsize=13,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor=color, linewidth=0.8))

        # Adjust y-limits for annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.25
        ax.set_ylim(new_y_min, current_ylim[1])

        # Add significance brackets if available
        if hasattr(self, 'volumewise_statistical_results'):
            stats_df = self.volumewise_statistical_results.get('HC_RVE', None)
            if stats_df is not None:
                box_positions_dict = {approach: plot_positions[i] for i, approach in enumerate(approach_order)}
                self._add_significance_brackets_approaches(ax, stats_df, list(box_positions_dict.values()),
                                                          approach_order, position='below')

        # Add legend box (top right)
        ax.text(0.98, 0.98, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_hc_hd95_subplot(self, ax, group_data, approach_order, approach_colors, approach_labels):
        """Plot HD95 boxplot for HC (all approaches)"""
        # Filter valid HD95 values
        df_plot = group_data[group_data['HD95_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Prepare box plot data
        plot_data_list = []
        plot_positions = []
        plot_colors = []
        position = 0

        for approach in approach_order:
            approach_data = df_plot[df_plot['Approach'] == approach]
            if not approach_data.empty:
                plot_data_list.append(approach_data['HD95_mm'].values)
            else:
                plot_data_list.append([])

            plot_colors.append(approach_colors.get(approach, '#95a5a6'))
            plot_positions.append(position)
            position += 1.0

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.7,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(D) 95th Percentile Hausdorff Distance (n=12)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=18, fontweight='bold')
        ax.set_ylabel('HD95 (mm) per volume', fontsize=18, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approach_order])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add annotations (median, IQR, n)
        for i, approach in enumerate(approach_order):
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
                           ha='center', va='top', fontsize=13,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor=color, linewidth=0.8))

        # Adjust y-limits for annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.25
        ax.set_ylim(new_y_min, current_ylim[1])

        # Add significance brackets if available
        if hasattr(self, 'volumewise_statistical_results'):
            stats_df = self.volumewise_statistical_results.get('HC_HD95', None)
            if stats_df is not None:
                box_positions_dict = {approach: plot_positions[i] for i, approach in enumerate(approach_order)}
                self._add_significance_brackets_approaches(ax, stats_df, list(box_positions_dict.values()),
                                                          approach_order, position='below', y_offset=0.05)

        # Add legend box (top right)
        ax.text(0.98, 0.98, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _add_patient_significance_bracket(self, ax, pos1, pos2, y_pos, significance, metric='DSC', direction='down'):
        """Add significance bracket between two positions for patient plot

        Args:
            direction: 'down' for downward-facing brackets (DSC subplot), 'up' for upward-facing (RVE, HD95 subplots)
        """
        if significance == 'ns':
            return  # Don't show non-significant comparisons

        # Bracket positions
        x1, x2 = pos1, pos2
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        if direction == 'up':
            # Upward-facing bracket: starts at y_pos, goes down, then back up
            bracket_height = -0.02 * y_range  # Negative for downward tick
            ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bracket_height, y_pos + bracket_height, y_pos],
                    lw=1.5, c='black')
            # Stars below the bracket
            ax.text((x1 + x2) * 0.5, y_pos + bracket_height - 0.01 * y_range,
                    significance, ha='center', va='top', fontsize=14, fontweight='bold')
        else:
            # Downward-facing bracket: starts at y_pos, goes up, then back down
            bracket_height = 0.02 * y_range  # Positive for upward tick
            ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bracket_height, y_pos + bracket_height, y_pos],
                    lw=1.5, c='black')
            # Stars above the bracket (very close to bracket)
            ax.text((x1 + x2) * 0.5, y_pos + bracket_height,
                    significance, ha='center', va='bottom', fontsize=14, fontweight='bold')

    def _plot_patient_dsc_subplot(self, ax, patient_df, approach_colors, approach_labels):
        """Plot DSC boxplot for patients (Ipsilateral vs Contralateral)"""
        df_plot = patient_df[patient_df['DSC_Volume'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        # Create box positions: Ipsi-Thresh, Ipsi-CBF | Contra-Thresh, Contra-CBF
        positions = [0, 0.8, 1.8, 2.6]  # Reduced gap between ipsi and contra groups
        box_data = []
        box_colors = []
        box_labels = []

        for hemi_cat in ['Ipsilateral', 'Contralateral']:
            for approach in ['Thresholding', 'CBF']:
                data = df_plot[(df_plot['Hemisphere_Category'] == hemi_cat) &
                             (df_plot['Approach'] == approach)]['DSC_Volume'].values
                box_data.append(data if len(data) > 0 else [])
                box_colors.append(approach_colors[approach])
                hemi_label = 'Ipsi' if hemi_cat == 'Ipsilateral' else 'Contra'
                box_labels.append(f"{approach_labels[approach]}\n{hemi_label}")

        # Create boxplot
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                       widths=0.6, medianprops=dict(color='black', linewidth=2))

        # Color boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title('(A) Dice Similarity Coefficient (ipsi. n=10; contra. n=11)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Approach / Hemisphere Category', fontsize=18, fontweight='bold')
        ax.set_ylabel('Dice per volume', fontsize=18, fontweight='bold')

        # X-axis setup
        ax.set_xticks(positions)
        ax.set_xticklabels(box_labels)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Add vertical line to separate ipsi and contra
        ax.axvline(x=1.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add annotations
        for i, pos in enumerate(positions):
            if len(box_data[i]) > 0:
                values = box_data[i]
                median = np.median(values)
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                n = len(values)

                # Will add annotations after adjusting y-axis
                pass

        # Extend y-axis to accommodate annotations with proper spacing
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_max = current_ylim[1] + y_range * 0.35  # Space above for headers
        new_y_min = current_ylim[0] - y_range * 0.20  # More space below for legend
        ax.set_ylim(new_y_min, new_y_max)

        # Now add all annotations in proper order (top to bottom) with proper spacing
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        y_range = y_max - y_min

        # 1. Headers at the very top (moved higher)
        header_y = y_max - y_range * 0.010  # Headers moved higher
        ax.text(0.4, header_y, 'Ipsilateral', fontsize=14, fontweight='bold', ha='center', va='top')
        ax.text(2.2, header_y, 'Contralateral', fontsize=14, fontweight='bold', ha='center', va='top')

        # 2. Significance brackets below headers (moved down)
        if hasattr(self, 'patient_statistical_results') and 'DSC' in self.patient_statistical_results:
            stats_df = self.patient_statistical_results['DSC']

            # Ipsilateral comparison
            ipsi_stats = stats_df[stats_df['Hemisphere_Category'] == 'Ipsilateral']
            if not ipsi_stats.empty:
                significance = ipsi_stats.iloc[0]['Significance']
                if significance != 'ns':
                    bracket_y = y_max - y_range * 0.130  # Moved lower
                    self._add_patient_significance_bracket(ax, positions[0], positions[1], bracket_y, significance, direction='down')

            # Contralateral comparison
            contra_stats = stats_df[stats_df['Hemisphere_Category'] == 'Contralateral']
            if not contra_stats.empty:
                significance = contra_stats.iloc[0]['Significance']
                if significance != 'ns':
                    bracket_y = y_max - y_range * 0.130  # Moved lower
                    self._add_patient_significance_bracket(ax, positions[2], positions[3], bracket_y, significance, direction='down')

        # 3. Median [IQR] boxes below brackets (moved lower)
        for i, pos in enumerate(positions):
            if len(box_data[i]) > 0:
                values = box_data[i]
                median = np.median(values)
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                n = len(values)

                label = f'{median:.3f} [{iqr:.3f}]'
                approach_idx = i % 2
                color = approach_colors[['Thresholding', 'CBF'][approach_idx]]

                median_y = y_max - y_range * 0.160  # Moved lower
                ax.text(pos, median_y, label,
                       ha='center', va='top', fontsize=13,
                       color=color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.8, edgecolor=color, linewidth=0.8))

        # Legend - positioned in bottom right without overlapping
        ax.text(0.98, 0.02, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_patient_rve_subplot(self, ax, patient_df, approach_colors, approach_labels):
        """Plot RVE boxplot for patients (Ipsilateral vs Contralateral)"""
        df_plot = patient_df[patient_df['RVE_Percent'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        positions = [0, 0.8, 1.8, 2.6]
        box_data = []
        box_colors = []
        box_labels = []

        for hemi_cat in ['Ipsilateral', 'Contralateral']:
            for approach in ['Thresholding', 'CBF']:
                data = df_plot[(df_plot['Hemisphere_Category'] == hemi_cat) &
                             (df_plot['Approach'] == approach)]['RVE_Percent'].values
                box_data.append(data if len(data) > 0 else [])
                box_colors.append(approach_colors[approach])
                hemi_label = 'Ipsi' if hemi_cat == 'Ipsilateral' else 'Contra'
                box_labels.append(f"{approach_labels[approach]}\n{hemi_label}")

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                       widths=0.6, medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        ax.set_title('(B) Relative Volume Error (ipsi. n=10; contra. n=11)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Approach / Hemisphere Category', fontsize=18, fontweight='bold')
        ax.set_ylabel('RVE (%) per volume', fontsize=18, fontweight='bold')

        ax.set_xticks(positions)
        ax.set_xticklabels(box_labels)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.axvline(x=1.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Extend y-axis downward to accommodate annotations and upward for legend
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.42  # MUCH MORE space below for headers lower
        new_y_max = current_ylim[1] + y_range * 0.18  # Add space at top for legend
        ax.set_ylim(new_y_min, new_y_max)

        # Now add annotations with properly calculated positions - BOTTOM TO TOP ORDER
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        y_range = y_max - y_min

        # 1. Headers MUCH LOWER at the very bottom
        header_y = y_min + y_range * 0.008  # Lower than before
        ax.text(0.4, header_y, 'Ipsilateral', fontsize=14, fontweight='bold', ha='center', va='bottom')
        ax.text(2.2, header_y, 'Contralateral', fontsize=14, fontweight='bold', ha='center', va='bottom')

        # 2. Significance brackets above headers - upward-facing - MOVED HIGHER
        if hasattr(self, 'patient_statistical_results') and 'RVE' in self.patient_statistical_results:
            stats_df = self.patient_statistical_results['RVE']

            for i, hemi_cat in enumerate(['Ipsilateral', 'Contralateral']):
                hemi_stats = stats_df[stats_df['Hemisphere_Category'] == hemi_cat]
                if not hemi_stats.empty:
                    significance = hemi_stats.iloc[0]['Significance']
                    if significance != 'ns':
                        # Position brackets HIGHER
                        bracket_y = y_min + y_range * 0.130  # Moved higher
                        pos1, pos2 = (positions[0], positions[1]) if i == 0 else (positions[2], positions[3])
                        self._add_patient_significance_bracket(ax, pos1, pos2, bracket_y, significance, direction='up')

        # 3. Median boxes above brackets - MOVED HIGHER
        for i, pos in enumerate(positions):
            if len(box_data[i]) > 0:
                values = box_data[i]
                median = np.median(values)
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                n = len(values)

                label = f'{median:.1f} [{iqr:.1f}]'
                approach_idx = i % 2
                color = approach_colors[['Thresholding', 'CBF'][approach_idx]]

                # Median boxes - moved higher
                ax.text(pos, y_min + y_range * 0.170, label,
                       ha='center', va='bottom', fontsize=13,
                       color=color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.9, edgecolor=color, linewidth=0.8))

        # Position legend in top-right corner (y-axis extended upward to avoid overlap)
        ax.text(0.98, 0.98, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_patient_assd_subplot(self, ax, patient_df, approach_colors, approach_labels):
        """Plot ASSD boxplot for patients (Ipsilateral vs Contralateral)"""
        df_plot = patient_df[patient_df['ASSD_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        positions = [0, 0.8, 1.8, 2.6]
        box_data = []
        box_colors = []
        box_labels = []

        for hemi_cat in ['Ipsilateral', 'Contralateral']:
            for approach in ['Thresholding', 'CBF']:
                data = df_plot[(df_plot['Hemisphere_Category'] == hemi_cat) &
                             (df_plot['Approach'] == approach)]['ASSD_mm'].values
                box_data.append(data if len(data) > 0 else [])
                box_colors.append(approach_colors[approach])
                hemi_label = 'Ipsi' if hemi_cat == 'Ipsilateral' else 'Contra'
                box_labels.append(f"{approach_labels[approach]}\n{hemi_label}")

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                       widths=0.6, medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        ax.set_title('(C) Average Symmetric Surface Distance (ipsi. n=10; contra. n=11)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Approach / Hemisphere Category', fontsize=18, fontweight='bold')
        ax.set_ylabel('ASSD (mm) per slice', fontsize=18, fontweight='bold')

        ax.set_xticks(positions)
        ax.set_xticklabels(box_labels)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.axvline(x=1.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        for i, pos in enumerate(positions):
            if len(box_data[i]) > 0:
                values = box_data[i]
                median = np.median(values)
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                n = len(values)

                y_min = ax.get_ylim()[0]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

                label = f'{median:.1f} [{iqr:.1f}]'
                approach_idx = i % 2
                color = approach_colors[['Thresholding', 'CBF'][approach_idx]]

                ax.text(pos, y_min - y_range * 0.02, label,
                       ha='center', va='top', fontsize=13,
                       color=color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.9, edgecolor=color, linewidth=0.8))

        # Extend y-axis downward to accommodate annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.30
        ax.set_ylim(new_y_min, current_ylim[1])

        # Add headers below the boxes (ASSD subplot) - after extending y-axis with more spacing
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        y_range = y_max - y_min
        header_y = y_min + y_range * 0.02  # Position headers closer to the bottom

        ax.text(0.4, header_y, 'Ipsilateral', fontsize=14, fontweight='bold', ha='center', va='bottom')
        ax.text(2.2, header_y, 'Contralateral', fontsize=14, fontweight='bold', ha='center', va='bottom')

        # Add significance brackets for ASSD - moved higher
        if hasattr(self, 'patient_statistical_results') and 'ASSD' in self.patient_statistical_results:
            stats_df = self.patient_statistical_results['ASSD']
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            y_range_bracket = y_max - y_min

            # Ipsilateral comparison
            ipsi_stats = stats_df[stats_df['Hemisphere_Category'] == 'Ipsilateral']
            if not ipsi_stats.empty:
                significance = ipsi_stats.iloc[0]['Significance']
                if significance != 'ns':
                    bracket_y = y_min + y_range_bracket * 0.14  # Moved higher
                    self._add_patient_significance_bracket(ax, positions[0], positions[1], bracket_y, significance, direction='up')

            # Contralateral comparison
            contra_stats = stats_df[stats_df['Hemisphere_Category'] == 'Contralateral']
            if not contra_stats.empty:
                significance = contra_stats.iloc[0]['Significance']
                if significance != 'ns':
                    bracket_y = y_min + y_range_bracket * 0.14  # Moved higher
                    self._add_patient_significance_bracket(ax, positions[2], positions[3], bracket_y, significance, direction='up')

        ax.text(0.98, 0.98, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _plot_patient_hd95_subplot(self, ax, patient_df, approach_colors, approach_labels):
        """Plot HD95 boxplot for patients (Ipsilateral vs Contralateral)"""
        df_plot = patient_df[patient_df['HD95_mm'].replace([np.inf, -np.inf], np.nan).notna()].copy()

        positions = [0, 0.8, 1.8, 2.6]
        box_data = []
        box_colors = []
        box_labels = []

        for hemi_cat in ['Ipsilateral', 'Contralateral']:
            for approach in ['Thresholding', 'CBF']:
                data = df_plot[(df_plot['Hemisphere_Category'] == hemi_cat) &
                             (df_plot['Approach'] == approach)]['HD95_mm'].values
                box_data.append(data if len(data) > 0 else [])
                box_colors.append(approach_colors[approach])
                hemi_label = 'Ipsi' if hemi_cat == 'Ipsilateral' else 'Contra'
                box_labels.append(f"{approach_labels[approach]}\n{hemi_label}")

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                       widths=0.6, medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        ax.set_title('(D) 95th Percentile Hausdorff Distance (ipsi. n=10; contra. n=11)', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Approach / Hemisphere Category', fontsize=18, fontweight='bold')
        ax.set_ylabel('HD95 (mm) per volume', fontsize=18, fontweight='bold')

        ax.set_xticks(positions)
        ax.set_xticklabels(box_labels)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.axvline(x=1.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Extend y-axis downward to accommodate annotations
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        new_y_min = current_ylim[0] - y_range * 0.42  # MUCH MORE space below for headers lower
        ax.set_ylim(new_y_min, current_ylim[1])

        # Now add annotations with properly calculated positions - BOTTOM TO TOP ORDER
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        y_range = y_max - y_min

        # 1. Headers moved MINIMALLY HIGHER at the bottom
        header_y = y_min + y_range * 0.012  # Minimally higher from 0.008
        ax.text(0.4, header_y, 'Ipsilateral', fontsize=14, fontweight='bold', ha='center', va='bottom')
        ax.text(2.2, header_y, 'Contralateral', fontsize=14, fontweight='bold', ha='center', va='bottom')

        # 2. Significance brackets HIGHER above headers - upward-facing
        if hasattr(self, 'patient_statistical_results') and 'HD95' in self.patient_statistical_results:
            stats_df = self.patient_statistical_results['HD95']

            for i, hemi_cat in enumerate(['Ipsilateral', 'Contralateral']):
                hemi_stats = stats_df[stats_df['Hemisphere_Category'] == hemi_cat]
                if not hemi_stats.empty:
                    significance = hemi_stats.iloc[0]['Significance']
                    if significance != 'ns':
                        # Position brackets HIGHER
                        bracket_y = y_min + y_range * 0.135  # Moved higher
                        pos1, pos2 = (positions[0], positions[1]) if i == 0 else (positions[2], positions[3])
                        self._add_patient_significance_bracket(ax, pos1, pos2, bracket_y, significance, direction='up')

        # 3. Median boxes moved HIGHER
        for i, pos in enumerate(positions):
            if len(box_data[i]) > 0:
                values = box_data[i]
                median = np.median(values)
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                n = len(values)

                label = f'{median:.1f} [{iqr:.1f}]'
                approach_idx = i % 2
                color = approach_colors[['Thresholding', 'CBF'][approach_idx]]

                # Median boxes - moved higher
                ax.text(pos, y_min + y_range * 0.180, label,
                       ha='center', va='bottom', fontsize=13,
                       color=color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               alpha=0.9, edgecolor=color, linewidth=0.8))

        ax.text(0.98, 0.98, 'Median [IQR]\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
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
    # Directory containing pre-computed Excel results
    # (same directory as this script)
    script_dir = Path(__file__).parent
    excel_data_dir = script_dir  # Excel files are in the same directory

    # Output directory for new plots
    output_dir = script_dir

    # These are not used when loading from Excel, but kept for compatibility
    predictions_dirs = {}
    gt_dir = None

    print("Test Set Evaluation Script - Loading from Pre-computed Excel Files")
    print("=" * 70)
    print("Loading data for approaches:")
    print("  Thresholding")
    print("  Single-class CBF (Perfusion only)")
    print("  Single-class CBF+T1w (Perfusion + MP-RAGE)")
    print("  Single-class CBF+FLAIR (Perfusion + FLAIR)")
    print("  Single-class CBF+T1w+FLAIR (Perfusion + MP-RAGE + FLAIR)")
    print(f"\nExcel data directory: {excel_data_dir}")
    print(f"Output directory: {output_dir}\n")

    evaluator = TestSetEvaluator(predictions_dirs, gt_dir, output_dir, excel_data_dir=excel_data_dir)
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