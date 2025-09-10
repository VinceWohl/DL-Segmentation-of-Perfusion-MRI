#!/usr/bin/env python3
"""
Multi-label cross-validation evaluation for nnU-Net - v1
Evaluates dual-channel binary segmentation predictions where:
- Channel 0: Left hemisphere perfusion territory predictions
- Channel 1: Right hemisphere perfusion territory predictions

Generates Excel sheets: Per_Case_Details, Per_Hemisphere_Summary, Overall_Summary
Computes 9 metrics: DSC_Volume, DSC_Slicewise, IoU, Sensitivity, Precision, 
                   Specificity, RVE_Percent, HD95_mm, ASSD_mm
"""

import sys, re, json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy.spatial.distance import cdist

try:
    import nibabel as nib
    import pandas as pd
    from openpyxl import Workbook
except Exception as e:
    print("Missing package:", e)
    sys.exit(1)


class MultiLabelCrossValidationEvaluator:
    def __init__(self, results_root: Path, output_dir: Path):
        self.results_root = Path(results_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # GT path remains the same
        self.gt_dir = Path("/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories/labelsTr")

        self.hemisphere_names = {
            0: "Left",
            1: "Right"
        }

        self.results = []
        self.fold_summaries = []
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []

    # ---------------- I/O ----------------
    def load_nifti_file(self, file_path: Path):
        try:
            nii = nib.load(str(file_path))
            data = nii.get_fdata()
            spacing = nii.header.get_zooms()[:3]  # Only spatial dimensions
            return data, spacing
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def load_multilabel_gt(self, file_path: Path):
        """Load multi-label GT and convert to binary channels"""
        try:
            nii = nib.load(str(file_path))
            data = nii.get_fdata()
            spacing = nii.header.get_zooms()[:3]
            
            # Check if already in channel format (H,W,D,2) or needs conversion
            if len(data.shape) == 4 and data.shape[-1] == 2:
                # Already in binary channel format
                binary_data = (data > 0.5).astype(np.uint8)
            else:
                # Convert from multi-label to binary channels
                # Assume format (H,W,D) with labels 0=background, 1=left, 2=right
                binary_data = np.zeros(data.shape + (2,), dtype=np.uint8)
                binary_data[..., 0] = (data == 1).astype(np.uint8)  # Left hemisphere
                binary_data[..., 1] = (data == 2).astype(np.uint8)  # Right hemisphere
            
            return binary_data, spacing
        except Exception as e:
            print(f"Error loading multi-label GT {file_path}: {e}")
            return None, None

    def load_prediction(self, file_path: Path):
        """Load prediction and ensure it's in binary format"""
        try:
            nii = nib.load(str(file_path))
            data = nii.get_fdata()
            spacing = nii.header.get_zooms()[:3]
            
            # Convert to binary if needed (threshold at 0.5)
            if len(data.shape) == 4 and data.shape[-1] == 2:
                binary_data = (data > 0.5).astype(np.uint8)
            else:
                print(f"Unexpected prediction shape: {data.shape}")
                return None, None
                
            return binary_data, spacing
        except Exception as e:
            print(f"Error loading prediction {file_path}: {e}")
            return None, None

    # ---------------- Metrics (9) ----------------
    def calculate_dice_score(self, gt_mask, pred_mask):
        inter = np.sum(gt_mask * pred_mask)
        denom = np.sum(gt_mask) + np.sum(pred_mask)
        if denom == 0:
            return 1.0 if np.sum(gt_mask) == np.sum(pred_mask) else 0.0
        return float(2.0 * inter / denom)

    def calculate_dice_score_slicewise(self, gt_mask, pred_mask):
        if gt_mask.shape != pred_mask.shape:
            return float("nan")
        nz = gt_mask.shape[2]
        vals = []
        for z in range(nz):
            g, p = gt_mask[:, :, z], pred_mask[:, :, z]
            inter = np.sum(g * p)
            denom = np.sum(g) + np.sum(p)
            if denom == 0:
                vals.append(1.0 if np.sum(g) == np.sum(p) else 0.0)
            else:
                vals.append(2.0 * inter / denom)
        return float(np.mean(vals)) if vals else float("nan")

    def calculate_cardinalities(self, gt_mask, pred_mask):
        tp = np.sum((gt_mask == 1) & (pred_mask == 1))
        fp = np.sum((gt_mask == 0) & (pred_mask == 1))
        fn = np.sum((gt_mask == 1) & (pred_mask == 0))
        tn = np.sum((gt_mask == 0) & (pred_mask == 0))
        return int(tp), int(fp), int(fn), int(tn)

    def calculate_sensitivity_specificity(self, tp, fp, fn, tn):
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(sens), float(spec)

    def calculate_precision(self, tp, fp):
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    def calculate_iou(self, gt_mask, pred_mask):
        inter = np.sum(gt_mask * pred_mask)
        union = np.sum((gt_mask | pred_mask).astype(np.uint8))
        if union == 0:
            return 1.0 if np.sum(gt_mask) == np.sum(pred_mask) else 0.0
        return float(inter / union)

    def get_surface_points(self, mask, spacing):
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure)
        boundary = mask & (~eroded)
        coords = np.array(np.where(boundary)).T
        return coords * np.array(spacing) if len(coords) > 0 else coords

    def calculate_hausdorff_distance_95(self, gt_mask, pred_mask, spacing):
        try:
            gt_s = self.get_surface_points(gt_mask, spacing)
            pr_s = self.get_surface_points(pred_mask, spacing)
            if len(gt_s) == 0 and len(pr_s) == 0:
                return 0.0
            if len(gt_s) == 0 or len(pr_s) == 0:
                return float("inf")
            d = cdist(gt_s, pr_s)
            if d.size == 0:
                return float("inf")
            d1 = d.min(axis=1)
            d2 = d.min(axis=0)
            all_d = np.concatenate([d1, d2])
            return float(np.percentile(all_d, 95))
        except Exception as e:
            print(f"Warning: HD95 failed: {e}")
            return float("nan")

    def calculate_assd(self, gt_mask, pred_mask, spacing):
        try:
            gt_s = self.get_surface_points(gt_mask, spacing)
            pr_s = self.get_surface_points(pred_mask, spacing)
            if len(gt_s) == 0 and len(pr_s) == 0:
                return 0.0
            if len(gt_s) == 0 or len(pr_s) == 0:
                return float("inf")
            d1 = cdist(gt_s, pr_s).min(axis=1).mean()
            d2 = cdist(pr_s, gt_s).min(axis=1).mean()
            return float((d1 + d2) / 2.0)
        except Exception as e:
            print(f"Warning: ASSD failed: {e}")
            return float("nan")

    def calculate_rve_percent(self, gt_mask, pred_mask, spacing):
        vox_vol = float(np.prod(spacing))
        gt_vol = np.sum(gt_mask) * vox_vol
        pr_vol = np.sum(pred_mask) * vox_vol
        if gt_vol == 0:
            return float("inf") if pr_vol > 0 else 0.0
        return float((pr_vol - gt_vol) / gt_vol * 100.0)

    # ---------------- Single Case ----------------
    def evaluate_single_case(self, pred_file: Path, gt_file: Path, fold: int, base_name: str):
        # Load data
        pred_data, pred_spacing = self.load_prediction(pred_file)
        gt_data, gt_spacing = self.load_multilabel_gt(gt_file)
        
        if pred_data is None or gt_data is None:
            print(f"  ERROR: Failed to load data for {base_name}")
            return None
        
        if gt_data.shape != pred_data.shape:
            print(f"  ERROR: Shape mismatch: {base_name} GT{gt_data.shape} vs PR{pred_data.shape}")
            return None

        spacing = gt_spacing
        result = {"Fold": fold, "Base_Name": base_name}

        # Per-hemisphere evaluation
        for hemi_idx in range(2):  # 0=Left, 1=Right
            hemi_name = self.hemisphere_names[hemi_idx]
            
            # Extract hemisphere masks
            g = gt_data[:, :, :, hemi_idx]
            p = pred_data[:, :, :, hemi_idx]

            # Calculate all 9 metrics
            dice_v = self.calculate_dice_score(g, p)
            dice_sl = self.calculate_dice_score_slicewise(g, p)
            tp, fp, fn, tn = self.calculate_cardinalities(g, p)
            sens, spec = self.calculate_sensitivity_specificity(tp, fp, fn, tn)
            prec = self.calculate_precision(tp, fp)
            iou = self.calculate_iou(g, p)
            hd95 = self.calculate_hausdorff_distance_95(g, p, spacing)
            assd = self.calculate_assd(g, p, spacing)
            rve = self.calculate_rve_percent(g, p, spacing)

            # Store results
            result.update({
                f"Hemisphere_{hemi_name}_DSC_Volume": dice_v,
                f"Hemisphere_{hemi_name}_DSC_Slicewise": dice_sl,
                f"Hemisphere_{hemi_name}_IoU": iou,
                f"Hemisphere_{hemi_name}_Sensitivity": sens,
                f"Hemisphere_{hemi_name}_Precision": prec,
                f"Hemisphere_{hemi_name}_Specificity": spec,
                f"Hemisphere_{hemi_name}_RVE_Percent": rve,
                f"Hemisphere_{hemi_name}_HD95_mm": hd95,
                f"Hemisphere_{hemi_name}_ASSD_mm": assd,
            })

        return result

    # ---------------- Discovery ----------------
    def find_fold_cases(self):
        fold_cases = {}
        for fold_num in range(5):
            vdir = self.results_root / f"fold_{fold_num}" / "validation"
            if not vdir.exists():
                continue
            pred_files = sorted(vdir.glob("*.nii"))
            fold_cases[fold_num] = []
            for pf in pred_files:
                base = pf.stem
                gt = self.gt_dir / f"{base}.nii"
                if gt.exists():
                    fold_cases[fold_num].append((pf, gt, base))
                else:
                    print(f"Warning: No matching GT file for {base}")
        return fold_cases

    def load_fold_summary(self, fold_num):
        sfile = self.results_root / f"fold_{fold_num}" / "validation" / "summary.json"
        if not sfile.exists():
            return None
        try:
            with open(sfile, "r") as f:
                summary = json.load(f)
            return {"fold": fold_num, "cases_count": len(summary.get("metric_per_case", []))}
        except Exception:
            return None

    # ---------------- Main runner ----------------
    def run_evaluation(self):
        print("Multi-Label Cross-Validation nnUNet Evaluation")
        print("=" * 70)
        print(f"Results root: {self.results_root}")
        print(f"Ground truth: {self.gt_dir}")
        print(f"Output dir  : {self.output_dir}\n")

        if not self.results_root.exists():
            print("ERROR: results root not found."); return False
        if not self.gt_dir.exists():
            print("ERROR: GT dir not found."); return False

        # Load fold summaries
        for f in range(5):
            s = self.load_fold_summary(f)
            if s: self.fold_summaries.append(s)

        fold_cases = self.find_fold_cases()
        total = sum(len(v) for v in fold_cases.values())
        if total == 0:
            print("ERROR: no validation cases found."); return False
        print(f"Found {total} validation cases across {len(fold_cases)} folds.\n")

        # Process all cases
        for fold, cases in fold_cases.items():
            print(f"Processing fold {fold} ({len(cases)} cases)")
            for pred_file, gt_file, base in cases:
                print(f"  Evaluating {base}...")
                self.processed_count += 1
                res = self.evaluate_single_case(pred_file, gt_file, fold, base)
                if res is not None:
                    self.results.append(res)
                    self.successful_count += 1
                else:
                    self.failed_files.append(f"{base} (fold {fold})")

        self.save_results_to_excel()
        self.print_evaluation_summary()
        return True

    # ---------------- Excel ----------------
    def save_results_to_excel(self):
        if not self.results:
            print("No results to save."); return

        df = pd.DataFrame(self.results)

        # Extract Subject / Visit from Base_Name like PerfTerr###-v#
        svh = df.get("Base_Name", pd.Series(dtype=str)).astype(str).str.extract(r"PerfTerr(\d+)-v(\d+)")
        df["Subject"] = np.where(svh[0].notna(), "sub-p" + svh[0].fillna("").str.zfill(3), None)
        df["Visit"] = np.where(svh[1].notna(), "v" + svh[1].fillna(""), None)

        base_cols = ["Fold", "Subject", "Visit", "Base_Name"]
        metric_cols = ['DSC_Volume','DSC_Slicewise','IoU','Sensitivity','Precision','Specificity','RVE_Percent','HD95_mm','ASSD_mm']

        # Build Per_Case_Details - reshape hemisphere data
        case_rows = []
        for hemi in ["Left", "Right"]:
            pref = f"Hemisphere_{hemi}_"
            present = [m for m in metric_cols if (pref+m) in df.columns]
            if not present: continue
            for _, r in df.iterrows():
                row = {k: r.get(k) for k in base_cols}
                row["Hemisphere"] = hemi
                for m in present:
                    row[m] = r.get(pref+m, np.nan)
                case_rows.append(row)
        per_case_details = pd.DataFrame(case_rows)

        # Safe stats (consistent): ddof=0, inf->NaN
        def safe_stats(series):
            vals = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                return np.nan, np.nan
            return float(vals.mean()), float(vals.std(ddof=0))

        # Per_Hemisphere_Summary
        phs_rows = []
        if not per_case_details.empty:
            for m in metric_cols:
                if m not in per_case_details.columns: continue
                for hemi, sub in per_case_details.groupby("Hemisphere"):
                    mean_v, std_v = safe_stats(sub[m])
                    phs_rows.append({"Hemisphere": hemi, "Metric": m, "Mean": mean_v, "Std": std_v})
        per_hemi_summary = pd.DataFrame(phs_rows)

        # Overall_Summary (combined across both hemispheres)
        overall_rows = []
        if not per_case_details.empty:
            for m in metric_cols:
                if m not in per_case_details.columns: continue
                mean_v, std_v = safe_stats(per_case_details[m])
                overall_rows.append({"Metric": m, "Mean": mean_v, "Std": std_v})
        overall_summary = pd.DataFrame(overall_rows)

        # Save to Excel
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx = self.output_dir / f"multi_label_cross_validation_evaluation_results_{ts}.xlsx"

        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            per_case_details.to_excel(writer, sheet_name="Per_Case_Details", index=False)
            per_hemi_summary.to_excel(writer, sheet_name="Per_Hemisphere_Summary", index=False)
            overall_summary.to_excel(writer, sheet_name="Overall_Summary", index=False)

            # Auto-adjust column widths
            for ws in writer.sheets.values():
                for col in ws.columns:
                    mx = 0
                    cells = list(col)
                    for c in cells:
                        v = "" if c.value is None else str(c.value)
                        mx = max(mx, len(v))
                    ws.column_dimensions[cells[0].column_letter].width = min(mx + 2, 50)

        print(f"Excel saved: {xlsx}")

    # ---------------- Console summary ----------------
    def print_evaluation_summary(self):
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Processed : {self.processed_count}")
        print(f"Succeeded : {self.successful_count}")
        print(f"Failed    : {len(self.failed_files)}")
        if self.failed_files:
            print("Failed files: " + ", ".join(self.failed_files))
        
        if self.results:
            df = pd.DataFrame(self.results)
            print(f"\nPERFORMANCE SUMMARY:")
            for hemi in ["Left", "Right"]:
                dice_col = f"Hemisphere_{hemi}_DSC_Volume"
                if dice_col in df.columns:
                    dice_vals = df[dice_col].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(dice_vals) > 0:
                        print(f"{hemi} Hemisphere - Dice: {dice_vals.mean():.4f} ± {dice_vals.std():.4f}")
        print("="*70)


def main():
    default_results_root = Path("/home/ubuntu/DLSegPerf/data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer_multilabel__nnUNetPlans__2d")
    default_output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/evaluation_results")

    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_results_root
    output_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output_dir

    ev = MultiLabelCrossValidationEvaluator(results_root, output_dir)
    ok = ev.run_evaluation()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())