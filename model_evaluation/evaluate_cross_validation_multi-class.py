#!/usr/bin/env python3
"""
Multi-class cross-validation evaluation for nnU-Net (compact, robust) — v2
- Keeps original data-loading behavior.
- Computes SAME 9 metrics as threshold script.
- Excel sheets: Per_Class_Details, Per_Hemisphere_Details, Per_Class_Summary,
  Per_Hemisphere_Summary, Overall_Summary (combined across both hemispheres ONLY).
- Consistent statistics: mean; std with ddof=0; inf handled as NaN.
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


class MultiClassCrossValidationEvaluator:
    def __init__(self, results_root: Path, output_dir: Path):
        self.results_root = Path(results_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DO NOT change GT location
        self.gt_dir = Path("/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories/labelsTr")

        self.class_names = {
            0: "Background",
            1: "Perfusion_Left",
            2: "Perfusion_Right",
            3: "Perfusion_Overlap",
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
            data = nii.get_fdata().astype(np.uint8)
            spacing = nii.header.get_zooms()[:3]
            return data, spacing
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    # ---------------- Masks ----------------
    def extract_class_mask(self, multi_class_mask, class_id):
        return (multi_class_mask == class_id).astype(np.uint8)

    def combine_hemisphere_mask(self, multi_class_mask, hemisphere="left"):
        # overlap (class 3) counts for both hemispheres
        if hemisphere.lower() == "left":
            return ((multi_class_mask == 1) | (multi_class_mask == 3)).astype(np.uint8)
        if hemisphere.lower() == "right":
            return ((multi_class_mask == 2) | (multi_class_mask == 3)).astype(np.uint8)
        raise ValueError("hemisphere must be 'left' or 'right'")

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
        pred_mask, pred_spacing = self.load_nifti_file(pred_file)
        gt_mask, gt_spacing = self.load_nifti_file(gt_file)
        if pred_mask is None or gt_mask is None:
            return None
        if gt_mask.shape != pred_mask.shape:
            print(f"  Shape mismatch: {base_name} GT{gt_mask.shape} vs PR{pred_mask.shape}")
            return None

        spacing = gt_spacing
        result = {"Fold": fold, "Base_Name": base_name}

        # Per-class (skip background)
        classes = np.unique(np.concatenate([gt_mask.reshape(-1), pred_mask.reshape(-1)])).astype(int)
        classes = [c for c in classes if c in (1, 2, 3)]
        for cid in classes:
            cname = self.class_names[cid]
            g = self.extract_class_mask(gt_mask, cid)
            p = self.extract_class_mask(pred_mask, cid)

            dice_v = self.calculate_dice_score(g, p)
            dice_sl = self.calculate_dice_score_slicewise(g, p)
            tp, fp, fn, tn = self.calculate_cardinalities(g, p)
            sens, spec = self.calculate_sensitivity_specificity(tp, fp, fn, tn)
            prec = self.calculate_precision(tp, fp)
            iou = self.calculate_iou(g, p)
            hd95 = self.calculate_hausdorff_distance_95(g, p, spacing)
            assd = self.calculate_assd(g, p, spacing)
            rve = self.calculate_rve_percent(g, p, spacing)

            result.update({
                f"{cname}_DSC_Volume": dice_v,
                f"{cname}_DSC_Slicewise": dice_sl,
                f"{cname}_IoU": iou,
                f"{cname}_Sensitivity": sens,
                f"{cname}_Precision": prec,
                f"{cname}_Specificity": spec,
                f"{cname}_RVE_Percent": rve,
                f"{cname}_HD95_mm": hd95,
                f"{cname}_ASSD_mm": assd,
            })

        # Per-hemisphere (overlap included in both)
        for hemi in ("left", "right"):
            g = self.combine_hemisphere_mask(gt_mask, hemi)
            p = self.combine_hemisphere_mask(pred_mask, hemi)

            dice_v = self.calculate_dice_score(g, p)
            dice_sl = self.calculate_dice_score_slicewise(g, p)
            tp, fp, fn, tn = self.calculate_cardinalities(g, p)
            sens, spec = self.calculate_sensitivity_specificity(tp, fp, fn, tn)
            prec = self.calculate_precision(tp, fp)
            iou = self.calculate_iou(g, p)
            hd95 = self.calculate_hausdorff_distance_95(g, p, spacing)
            assd = self.calculate_assd(g, p, spacing)
            rve = self.calculate_rve_percent(g, p, spacing)

            hk = f"Hemisphere_{hemi.title()}"
            result.update({
                f"{hk}_DSC_Volume": dice_v,
                f"{hk}_DSC_Slicewise": dice_sl,
                f"{hk}_IoU": iou,
                f"{hk}_Sensitivity": sens,
                f"{hk}_Precision": prec,
                f"{hk}_Specificity": spec,
                f"{hk}_RVE_Percent": rve,
                f"{hk}_HD95_mm": hd95,
                f"{hk}_ASSD_mm": assd,
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
        return fold_cases

    def load_fold_summary(self, fold_num):
        sfile = self.results_root / f"fold_{fold_num}" / "validation" / "summary.json"
        if not sfile.exists():
            return None
        try:
            import json
            with open(sfile, "r") as f:
                summary = json.load(f)
            return {"fold": fold_num, "cases_count": len(summary.get("metric_per_case", []))}
        except Exception:
            return None

    # ---------------- Main runner ----------------
    def run_evaluation(self):
        print("Multi-Class Cross-Validation nnUNet Evaluation (v2)")
        print("=" * 70)
        print(f"Results root: {self.results_root}")
        print(f"Ground truth: {self.gt_dir}")
        print(f"Output dir  : {self.output_dir}\n")

        if not self.results_root.exists():
            print("ERROR: results root not found."); return False
        if not self.gt_dir.exists():
            print("ERROR: GT dir not found."); return False

        for f in range(5):
            s = self.load_fold_summary(f)
            if s: self.fold_summaries.append(s)

        fold_cases = self.find_fold_cases()
        total = sum(len(v) for v in fold_cases.values())
        if total == 0:
            print("ERROR: no validation cases found."); return False
        print(f"Found {total} validation cases across {len(fold_cases)} folds.\n")

        for fold, cases in fold_cases.items():
            print(f"Processing fold {fold} ({len(cases)} cases)")
            for pred_file, gt_file, base in cases:
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

        # Extract Subject / Visit from Base_Name like PerfTerr###-v#-L/R or PerfTerr###-v#
        svh = df.get("Base_Name", pd.Series(dtype=str)).astype(str).str.extract(r"PerfTerr(\d+)-v(\d+)(?:-[LR])?")
        df["Subject"] = np.where(svh[0].notna(), "sub-p" + svh[0].fillna("").str.zfill(3), None)
        df["Visit"] = np.where(svh[1].notna(), "v" + svh[1].fillna(""), None)

        base_cols = ["Fold", "Subject", "Visit", "Base_Name"]
        metric_cols = ['DSC_Volume','DSC_Slicewise','IoU','Sensitivity','Precision','Specificity','RVE_Percent','HD95_mm','ASSD_mm']

        # Build Per_Class_Details
        class_rows = []
        for cname in ["Perfusion_Left","Perfusion_Right","Perfusion_Overlap"]:
            pref = f"{cname}_"
            present = [m for m in metric_cols if (pref+m) in df.columns]
            if not present: continue
            for _, r in df.iterrows():
                row = {k: r.get(k) for k in base_cols}
                row["Class"] = cname
                for m in present:
                    row[m] = r.get(pref+m, np.nan)
                class_rows.append(row)
        per_class_details = pd.DataFrame(class_rows)

        # Build Per_Hemisphere_Details
        hemi_rows = []
        for hemi in ["Left","Right"]:
            pref = f"Hemisphere_{hemi}_"
            present = [m for m in metric_cols if (pref+m) in df.columns]
            if not present: continue
            for _, r in df.iterrows():
                row = {k: r.get(k) for k in base_cols}
                row["Hemisphere"] = hemi
                for m in present:
                    row[m] = r.get(pref+m, np.nan)
                hemi_rows.append(row)
        per_hemi_details = pd.DataFrame(hemi_rows)

        # Safe stats (consistent across sheets): ddof=0, inf->NaN
        def safe_stats(series):
            vals = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                return np.nan, np.nan
            return float(vals.mean()), float(vals.std(ddof=0))

        # Per_Class_Summary
        pcs_rows = []
        if not per_class_details.empty:
            for m in metric_cols:
                if m not in per_class_details.columns: continue
                for cname, sub in per_class_details.groupby("Class"):
                    mean_v, std_v = safe_stats(sub[m])
                    pcs_rows.append({"Class": cname, "Metric": m, "Mean": mean_v, "Std": std_v})
        per_class_summary = pd.DataFrame(pcs_rows)

        # Per_Hemisphere_Summary
        phs_rows = []
        if not per_hemi_details.empty:
            for m in metric_cols:
                if m not in per_hemi_details.columns: continue
                for hemi, sub in per_hemi_details.groupby("Hemisphere"):
                    mean_v, std_v = safe_stats(sub[m])
                    phs_rows.append({"Hemisphere": hemi, "Metric": m, "Mean": mean_v, "Std": std_v})
        per_hemi_summary = pd.DataFrame(phs_rows)

        # Overall_Summary (COMBINED across both hemispheres ONLY)
        overall_rows = []
        if not per_hemi_details.empty:
            for m in metric_cols:
                if m not in per_hemi_details.columns: continue
                mean_v, std_v = safe_stats(per_hemi_details[m])
                overall_rows.append({"Metric": m, "Mean": mean_v, "Std": std_v})
        overall_summary = pd.DataFrame(overall_rows)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx = self.output_dir / f"multi_class_cross_validation_evaluation_results_{ts}.xlsx"

        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            per_class_details.to_excel(writer, sheet_name="Per_Class_Details", index=False)
            per_hemi_details.to_excel(writer, sheet_name="Per_Hemisphere_Details", index=False)
            per_class_summary.to_excel(writer, sheet_name="Per_Class_Summary", index=False)
            per_hemi_summary.to_excel(writer, sheet_name="Per_Hemisphere_Summary", index=False)
            overall_summary.to_excel(writer, sheet_name="Overall_Summary", index=False)

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
        print("="*70)


def main():
    default_results_root = Path("/home/ubuntu/DLSegPerf/data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d")
    default_output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/evaluation_results")

    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_results_root
    output_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output_dir

    ev = MultiClassCrossValidationEvaluator(results_root, output_dir)
    ok = ev.run_evaluation()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
