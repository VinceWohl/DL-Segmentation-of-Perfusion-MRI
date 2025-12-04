#!/usr/bin/env python3
"""
Enhanced Test Set Evaluation Script with Group-based Analysis
Evaluates predictions against ground truth with separate sheets per group (HC, ICAS, AVM).

Computes per datapoint:
- Dice Similarity Coefficient (volume-wise)
- Relative Volume Error (volume-wise)
- ASSD (slice-wise average)
- HD95 (volume-wise)

Output: Excel file with 6 sheets (3 groups x 2 sheets each)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy.spatial.distance import cdist

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("ERROR: nibabel not available. Install with: pip install nibabel")
    sys.exit(1)


class GroupBasedTestEvaluator:
    """Evaluator for test set predictions with group-based analysis"""

    def __init__(self, predictions_dir, gt_dir, group_file, output_dir, output_name):
        self.predictions_dir = Path(predictions_dir)
        self.gt_dir = Path(gt_dir)
        self.group_file = Path(group_file)
        self.output_dir = Path(output_dir)
        self.output_name = output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = []
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []

        # Load group information
        self.group_mapping = self.load_group_mapping()

        # Verify directories exist
        if not self.predictions_dir.exists():
            raise ValueError(f"Predictions directory does not exist: {predictions_dir}")
        if not self.gt_dir.exists():
            raise ValueError(f"Ground truth directory does not exist: {gt_dir}")
        if not self.group_file.exists():
            raise ValueError(f"Group file does not exist: {group_file}")

    # ---------- Group Management ----------
    def load_group_mapping(self):
        """Load group mapping from Excel file"""
        df = pd.read_excel(self.group_file)
        # Create dictionary mapping Case_ID to Group
        group_dict = dict(zip(df['Case_ID'], df['Group']))
        print(f"Loaded group mapping for {len(group_dict)} cases")
        print(f"Groups: {df['Group'].unique()}")
        print(f"Group distribution: {df['Group'].value_counts().to_dict()}")
        return group_dict

    # ---------- File Loading ----------
    def load_nifti_file(self, file_path):
        """Load NIfTI file and return binary mask and spacing"""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            mask = (data > 0).astype(np.uint8)
            spacing = nii.header.get_zooms()[:3]
            return mask, spacing
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    # ---------- Metric Calculations ----------
    def calculate_dice_score(self, gt_mask, pred_mask):
        """Calculate Dice Similarity Coefficient (volume-wise)"""
        intersection = np.sum(gt_mask * pred_mask)
        union = np.sum(gt_mask) + np.sum(pred_mask)
        if union == 0:
            return 1.0 if np.sum(gt_mask) == np.sum(pred_mask) else 0.0
        return (2.0 * intersection) / union

    def calculate_relative_volume_error(self, gt_mask, pred_mask, spacing):
        """Calculate Relative Volume Error in percent (volume-wise)"""
        voxel_vol = np.prod(spacing)
        gt_vol = np.sum(gt_mask) * voxel_vol
        pred_vol = np.sum(pred_mask) * voxel_vol
        if gt_vol == 0:
            return float('inf') if pred_vol > 0 else 0.0
        return (pred_vol - gt_vol) / gt_vol * 100.0

    def get_surface_points_2d(self, mask_2d, spacing_2d):
        """Get surface points for 2D slice"""
        structure = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask_2d, structure)
        boundary = mask_2d & ~eroded
        coords = np.array(np.where(boundary)).T
        if len(coords) > 0:
            return coords * np.array(spacing_2d)
        return coords

    def calculate_assd_slicewise(self, gt_mask, pred_mask, spacing):
        """Calculate Average Symmetric Surface Distance (slice-wise average)"""
        try:
            nz = gt_mask.shape[2]
            slice_assd_values = []

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
                        continue  # Skip slices with only one empty surface
                    else:
                        d1 = cdist(gt_surface_2d, pred_surface_2d).min(axis=1).mean()
                        d2 = cdist(pred_surface_2d, gt_surface_2d).min(axis=1).mean()
                        assd_slice = float((d1 + d2) / 2.0)

                    if not np.isinf(assd_slice) and not np.isnan(assd_slice):
                        slice_assd_values.append(assd_slice)

                except Exception as e:
                    continue  # Skip problematic slices

            # Return mean ASSD across all valid slices
            if len(slice_assd_values) > 0:
                return float(np.mean(slice_assd_values))
            else:
                return float('nan')

        except Exception as e:
            print(f"Warning: Slice-wise ASSD calculation failed: {e}")
            return float('nan')

    def get_surface_points_3d(self, mask, spacing):
        """Get surface points for 3D volume"""
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure)
        boundary = mask & ~eroded
        coords = np.array(np.where(boundary)).T
        if len(coords) > 0:
            return coords * np.array(spacing)
        return coords

    def calculate_hausdorff_distance_95(self, gt_mask, pred_mask, spacing):
        """Calculate 95th percentile Hausdorff Distance (volume-wise)"""
        try:
            gt_surface = self.get_surface_points_3d(gt_mask, spacing)
            pred_surface = self.get_surface_points_3d(pred_mask, spacing)

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

    # ---------- Evaluation ----------
    def evaluate_single_case(self, pred_file, gt_file, base_name):
        """Evaluate a single prediction case"""
        print(f"  Evaluating {base_name}...")

        # Load prediction and ground truth
        pred_mask, _ = self.load_nifti_file(pred_file)
        gt_mask, gt_spacing = self.load_nifti_file(gt_file)

        if gt_mask is None or pred_mask is None:
            print(f"    ERROR: Failed to load files")
            return None

        spacing = gt_spacing

        if gt_mask.shape != pred_mask.shape:
            print(f"    ERROR: Shape mismatch - GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            return None

        # Get group for this case
        group = self.group_mapping.get(base_name, 'Unknown')
        if group == 'Unknown':
            print(f"    WARNING: No group found for {base_name}, skipping")
            return None

        # Calculate metrics
        dice = self.calculate_dice_score(gt_mask, pred_mask)
        rve = self.calculate_relative_volume_error(gt_mask, pred_mask, spacing)
        assd = self.calculate_assd_slicewise(gt_mask, pred_mask, spacing)
        hd95 = self.calculate_hausdorff_distance_95(gt_mask, pred_mask, spacing)

        # Extract hemisphere from filename
        hemisphere = 'Left' if base_name.endswith('-L') else 'Right'

        print(f"    Group: {group}, DSC: {dice:.4f}, RVE: {rve:.2f}%, ASSD: {assd:.3f}mm, HD95: {hd95:.2f}mm")

        return {
            'Case_ID': base_name,
            'Group': group,
            'Hemisphere': hemisphere,
            'DSC': dice,
            'RVE_Percent': rve,
            'ASSD_mm': assd,
            'HD95_mm': hd95
        }

    def find_test_cases(self):
        """Find matching prediction and ground truth files"""
        test_cases = []

        # Find all prediction files
        pred_files = sorted(self.predictions_dir.glob("*.nii*"))

        for pred_file in pred_files:
            # Extract base name
            base_name = pred_file.stem
            if pred_file.suffix == '.gz':
                base_name = pred_file.with_suffix('').stem

            # Find corresponding ground truth file
            gt_file = self.gt_dir / f"{base_name}.nii"
            if not gt_file.exists():
                gt_file = self.gt_dir / f"{base_name}.nii.gz"

            if gt_file.exists():
                test_cases.append((pred_file, gt_file, base_name))
            else:
                print(f"Warning: No matching GT file for {base_name}")
                self.failed_files.append(base_name)

        return test_cases

    def run_evaluation(self):
        """Run complete evaluation"""
        print("\n" + "="*80)
        print("Group-Based Test Set Evaluation")
        print("="*80)
        print(f"Predictions: {self.predictions_dir}")
        print(f"Ground Truth: {self.gt_dir}")
        print(f"Group File: {self.group_file}")
        print(f"Output: {self.output_dir}")
        print("="*80 + "\n")

        # Find test cases
        print("Finding test cases...")
        test_cases = self.find_test_cases()
        print(f"Found {len(test_cases)} test cases\n")

        if len(test_cases) == 0:
            print("ERROR: No test cases found!")
            return

        # Evaluate each case
        print("Evaluating predictions...")
        for pred_file, gt_file, base_name in test_cases:
            self.processed_count += 1
            result = self.evaluate_single_case(pred_file, gt_file, base_name)

            if result is not None:
                self.results.append(result)
                self.successful_count += 1
            else:
                self.failed_files.append(base_name)

        print("\n" + "="*80)
        print("Evaluation Summary")
        print("="*80)
        print(f"Total cases processed: {self.processed_count}")
        print(f"Successful evaluations: {self.successful_count}")
        print(f"Failed evaluations: {len(self.failed_files)}")
        if self.failed_files:
            print(f"Failed files: {', '.join(self.failed_files)}")
        print("="*80 + "\n")

        # Save results to Excel
        if self.results:
            self.save_results_to_excel()
        else:
            print("ERROR: No successful evaluations to save!")

    def save_results_to_excel(self):
        """Save evaluation results to Excel file with group-based sheets"""
        print("Saving results to Excel...")

        # Create DataFrame
        df = pd.DataFrame(self.results)

        # Sort by Group and Case_ID
        df = df.sort_values(['Group', 'Case_ID']).reset_index(drop=True)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{self.output_name}_{timestamp}.xlsx"

        # Get unique groups
        groups = ['HC', 'ICAS', 'AVM']  # Fixed order

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for group in groups:
                group_df = df[df['Group'] == group].copy()

                if len(group_df) == 0:
                    print(f"  Warning: No data for group {group}")
                    continue

                # Sheet 1: Group predictions (detailed per case)
                sheet_name_pred = f"{group}_test_predictions"
                group_df_pred = group_df[['Case_ID', 'Hemisphere', 'DSC', 'RVE_Percent', 'ASSD_mm', 'HD95_mm']].copy()
                group_df_pred.to_excel(writer, sheet_name=sheet_name_pred, index=False)

                # Sheet 2: Group summary statistics
                sheet_name_summary = f"{group}_test_summary"
                summary_data = {
                    'Metric': ['DSC', 'RVE_Percent', 'ASSD_mm', 'HD95_mm'],
                    'Mean': [
                        group_df['DSC'].mean(),
                        group_df['RVE_Percent'].mean(),
                        group_df['ASSD_mm'].mean(),
                        group_df['HD95_mm'].mean()
                    ],
                    'Std': [
                        group_df['DSC'].std(),
                        group_df['RVE_Percent'].std(),
                        group_df['ASSD_mm'].std(),
                        group_df['HD95_mm'].std()
                    ],
                    'Median': [
                        group_df['DSC'].median(),
                        group_df['RVE_Percent'].median(),
                        group_df['ASSD_mm'].median(),
                        group_df['HD95_mm'].median()
                    ],
                    'Min': [
                        group_df['DSC'].min(),
                        group_df['RVE_Percent'].min(),
                        group_df['ASSD_mm'].min(),
                        group_df['HD95_mm'].min()
                    ],
                    'Max': [
                        group_df['DSC'].max(),
                        group_df['RVE_Percent'].max(),
                        group_df['ASSD_mm'].max(),
                        group_df['HD95_mm'].max()
                    ],
                    'N': [
                        len(group_df),
                        len(group_df),
                        len(group_df),
                        len(group_df)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name=sheet_name_summary, index=False)

                print(f"  {group}: {len(group_df)} cases - DSC: {group_df['DSC'].mean():.4f} +/- {group_df['DSC'].std():.4f}")

        print(f"\nResults saved to: {output_file}")

        # Print overall summary
        print(f"\nOverall Summary Across All Groups:")
        for group in groups:
            group_df = df[df['Group'] == group]
            if len(group_df) > 0:
                print(f"  {group} (n={len(group_df)}):")
                print(f"    DSC: {group_df['DSC'].mean():.4f} +/- {group_df['DSC'].std():.4f}")
                print(f"    RVE: {group_df['RVE_Percent'].mean():.2f} +/- {group_df['RVE_Percent'].std():.2f}%")
                print(f"    ASSD: {group_df['ASSD_mm'].mean():.3f} +/- {group_df['ASSD_mm'].std():.3f} mm")
                print(f"    HD95: {group_df['HD95_mm'].mean():.2f} +/- {group_df['HD95_mm'].std():.2f} mm")


def main():
    """Main execution"""
    # Paths
    base_dir = Path("/home/ubuntu/DLSegPerf/data/nnUNet_raw/Dataset001_PerfusionTerritories")
    gt_dir = base_dir / "labelsTs"
    group_file = base_dir / "Group_distripution.xlsx"
    output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/test_evaluation/results_final_dataset")

    # Evaluation configurations
    evaluations = [
        {
            'name': 'nnUNet Predictions',
            'predictions_dir': "/home/ubuntu/DLSegPerf/data/nnUNet_results/Dataset001_PerfusionTerritories/nnUNetTrainer__nnUNetPlans__2d/fold_all/predictions",
            'output_name': 'test_results_nnUNet'
        },
        {
            'name': 'Thresholding Baseline',
            'predictions_dir': base_dir / "thresholded_labelsTs",
            'output_name': 'test_results_thresholding'
        }
    ]

    # Run evaluations
    for config in evaluations:
        print("\n" + "="*80)
        print(f"Running Evaluation: {config['name']}")
        print("="*80)

        evaluator = GroupBasedTestEvaluator(
            predictions_dir=config['predictions_dir'],
            gt_dir=gt_dir,
            group_file=group_file,
            output_dir=output_dir,
            output_name=config['output_name']
        )
        evaluator.run_evaluation()

    print("\n" + "="*80)
    print("All Evaluations Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
