#!/usr/bin/env python3
"""
Cross-Validation nnUNet Results Evaluation Script — single-class
Creates an Excel aligned with run_evaluation.py:
  - Sheets: Per_Case_Results, Per_Hemisphere_Summary, Overall_Summary
  - Metrics: DSC_Volume, DSC_Slicewise, IoU, Sensitivity, Precision,
             Specificity, RVE_Percent, HD95_mm, ASSD_mm
IMPORTANT: Data loading and file discovery logic remain unchanged.
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



class CrossValidationEvaluator:
    def __init__(self, results_root, output_dir):
        self.results_root = Path(results_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # (kept) GT path
        self.gt_dir = Path("/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories/labelsTr")

        # Results storage
        self.results = []
        self.fold_summaries = []
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []
        
        # Load fold splits for subject-to-fold mapping
        self.subject_to_fold_map = self._load_fold_splits()

    def _load_fold_splits(self):
        """Load fold splits and create subject-to-fold mapping"""
        splits_file = Path("/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories/splits_final.json")
        subject_to_fold = {}
        
        try:
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            # Create mapping from subject to fold based on validation cases
            for fold_idx, fold_data in enumerate(splits):
                for case in fold_data['val']:
                    # Extract subject from case name (e.g., "PerfTerr001-v1" -> "001")
                    subject_match = case.split('-')[0].replace('PerfTerr', '')
                    if subject_match:
                        subject_to_fold[subject_match] = fold_idx
            
            return subject_to_fold
        except Exception as e:
            print(f"Warning: Could not load fold splits: {e}")
            return {}

    def _get_fold_for_case(self, base_name):
        """Get fold number for a given case name"""
        # Extract subject from base name (e.g., "PerfTerr001-v1-L" -> "001")
        subject_match = base_name.split('-')[0].replace('PerfTerr', '')
        if subject_match in self.subject_to_fold_map:
            return self.subject_to_fold_map[subject_match]
        return None

    # ---------- Loading & basic metrics (unchanged where it matters) ----------
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

    # ---------- New: extra metrics to match run_evaluation.py ----------
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

    # ---------- Evaluation over a single case (kept, extended metrics only) ----------
    def evaluate_single_case(self, pred_file, gt_file, fold, base_name):
        print(f"  Evaluating {base_name} (Fold {fold})...")

        pred_mask, _ = self.load_nifti_file(pred_file)
        gt_mask, gt_spacing = self.load_nifti_file(gt_file)
        if gt_mask is None or pred_mask is None:
            print(f"    ERROR: Failed to load masks")
            return None

        spacing = gt_spacing
        if gt_mask.shape != pred_mask.shape:
            print(f"    ERROR: Shape mismatch - GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            return None

        # Core metrics
        dice_score = self.calculate_dice_score(gt_mask, pred_mask)
        dice_slice = self.calculate_dice_score_slicewise(gt_mask, pred_mask)
        tp, fp, fn, tn = self.calculate_cardinalities(gt_mask, pred_mask)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        hd95 = self.calculate_hausdorff_distance_95(gt_mask, pred_mask, spacing)

        # Added metrics to match threshold evaluation
        iou = self.calculate_iou(gt_mask, pred_mask)
        precision = self.calculate_precision(tp, fp)
        rve = self.calculate_relative_volume_error(gt_mask, pred_mask, spacing)
        assd = self.calculate_assd(gt_mask, pred_mask, spacing)

        # Volumes (kept for plotting/compat)
        voxel_volume_mm3 = np.prod(spacing)
        gt_volume_mm3 = np.sum(gt_mask) * voxel_volume_mm3
        pred_volume_mm3 = np.sum(pred_mask) * voxel_volume_mm3
        volume_difference_mm3 = pred_volume_mm3 - gt_volume_mm3

        print(f"    Dice: {dice_score:.4f}, Dice(slice): {dice_slice:.4f}, IoU: {iou:.4f}, HD95: {hd95:.2f}")

        # Keep original keys for existing plotting; add standardized keys used for Excel
        return {
            'Fold': fold,
            'Base_Name': base_name,

            # Original names (kept)
            'Dice_Score': dice_score,
            'Dice_Score_Slicewise': dice_slice,
            'HD95_mm': hd95,
            'True_Positives': tp,
            'False_Positives': fp,
            'False_Negatives': fn,
            'True_Negatives': tn,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'GT_Volume_mm3': gt_volume_mm3,
            'Pred_Volume_mm3': pred_volume_mm3,
            'Volume_Difference_mm3': volume_difference_mm3,

            # New names to mirror run_evaluation.py
            'DSC_Volume': dice_score,
            'DSC_Slicewise': dice_slice,
            'IoU': iou,
            'Precision': precision,
            'RVE_Percent': rve,
            'ASSD_mm': assd
        }

    # ---------- Discovery (unchanged) ----------
    def find_fold_cases(self):
        fold_cases = {}
        for fold_num in range(5):
            fold_dir = self.results_root / f"fold_{fold_num}" / "validation"
            if not fold_dir.exists():
                print(f"Warning: Fold {fold_num} directory not found: {fold_dir}")
                continue
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

    # ---------- Run ----------
    def run_evaluation(self):
        print("Cross-Validation nnUNet Results Evaluation")
        print("=" * 60)
        print(f"Results root: {self.results_root}")
        print(f"Ground truth: {self.gt_dir}")
        print(f"Output directory: {self.output_dir}\n")

        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        if not PANDAS_AVAILABLE:
            print("ERROR: pandas package is required. Please install with: pip install pandas openpyxl")
            return False
        if not self.results_root.exists():
            print(f"ERROR: Results directory not found: {self.results_root}")
            return False
        if not self.gt_dir.exists():
            print(f"ERROR: Ground truth directory not found: {self.gt_dir}")
            return False

        print("Loading fold summaries...")
        for fold_num in range(5):
            fold_summary = self.load_fold_summary(fold_num)
            if fold_summary:
                self.fold_summaries.append(fold_summary)
                print(f"  Fold {fold_num}: {fold_summary['mean_dice']:.4f} Dice, {fold_summary['cases_count']} cases")
            else:
                print(f"  Fold {fold_num}: Summary not available")
        print()

        fold_cases = self.find_fold_cases()
        total_cases = sum(len(cases) for cases in fold_cases.values())
        if total_cases == 0:
            print("ERROR: No validation cases found")
            return False
        print(f"Found {total_cases} validation cases across {len(fold_cases)} folds")

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

        print("\nSaving results to Excel...")
        self.save_results_to_excel()


        self.print_evaluation_summary()
        return True

    # ---------- Excel output to match run_evaluation.py ----------
    def save_results_to_excel(self):
        if not self.results:
            print("No results to save")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = self.output_dir / f"crossval_singleclass_results_{timestamp}.xlsx"

            df = pd.DataFrame(self.results)

            # Subject/Visit/Hemisphere parsing
            df[['Subject', 'Visit', 'Hemisphere']] = df['Base_Name'].str.extract(r'PerfTerr(\d+)-v(\d+)-([LR])')
            # Convert L/R to Left/Right
            df['Hemisphere'] = df['Hemisphere'].map({'L': 'Left', 'R': 'Right'})
            df['Subject'] = 'sub-p' + df['Subject'].str.zfill(3)
            df['Visit'] = 'v' + df['Visit']
            
            # Add Fold column using mapping
            df['Fold'] = df['Base_Name'].apply(self._get_fold_for_case)
            
            # Sort by Fold, Subject, Visit, Hemisphere
            df = df.sort_values(['Fold', 'Subject', 'Visit', 'Hemisphere'], na_position='last')

            # ----- Sheet 1: Per_Case_Results (same columns & names) -----
            per_case_cols = [
                'Fold', 'Subject', 'Visit', 'Hemisphere', 'Base_Name',
                'DSC_Volume', 'DSC_Slicewise', 'IoU',
                'Sensitivity', 'Precision', 'Specificity',
                'RVE_Percent', 'HD95_mm', 'ASSD_mm'
            ]
            per_case_df = df[per_case_cols].copy()

            # ----- Sheet 2: Per_Hemisphere_Summary (Mean/Std for the 9 metrics) -----
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

            # ----- Sheet 3: Overall_Summary (Mean/Std for the 9 metrics) -----
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

            # Write to Excel (Per_Case_Details first)
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Write sheets in order: Per_Case_Details, Per_Hemisphere_Summary, Overall_Summary
                per_case_df.to_excel(writer, sheet_name='Per_Case_Details', index=False)
                hemi_df.to_excel(writer, sheet_name='Per_Hemisphere_Summary', index=False)
                overall_df.to_excel(writer, sheet_name='Overall_Summary', index=False)

                # Auto width
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

            print(f"Results saved to Excel: {excel_file}")

        except Exception as e:
            print(f"Error saving Excel file: {e}")


    def print_evaluation_summary(self):
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
            dice_scores = df['Dice_Score'].values
            dice_scores_slice = df['Dice_Score_Slicewise'].values
            hd95_values = df['HD95_mm'].replace([np.inf, -np.inf], np.nan).dropna().values
            print(f"\nOVERALL PERFORMANCE SUMMARY:")
            print(f"Dice Score (Volume)   - Mean: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
            print(f"Dice Score (Slice)    - Mean: {np.mean(dice_scores_slice):.4f} ± {np.std(dice_scores_slice):.4f}")
            if len(hd95_values) > 0:
                print(f"HD95 (mm)             - Mean: {np.mean(hd95_values):.2f} ± {np.std(hd95_values):.2f}")
                print(f"HD95 Range            - Min: {np.min(hd95_values):.2f}, Max: {np.max(hd95_values):.2f}")
            print(f"\nPER-FOLD PERFORMANCE:")
            for fold_num in range(5):
                fold_data = df[df['Fold'] == fold_num]
                if len(fold_data) > 0:
                    print(f"Fold {fold_num}: {fold_data['Dice_Score'].mean():.4f} Dice ({len(fold_data)} cases)")
        print("=" * 60)


def main():
    default_results_root = Path("/home/ubuntu/DLSegPerf/data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d")
    default_output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/evaluation_results")

    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_results_root
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output_dir

    print("Cross-Validation nnUNet Evaluation Script")
    print("=" * 60)
    print(f"Results root: {results_root}")
    print(f"Output dir: {output_dir}\n")

    evaluator = CrossValidationEvaluator(results_root, output_dir)
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
