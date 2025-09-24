#!/usr/bin/env python3
"""
Threshold Segmentation Evaluation Script

This script computes the Dice Similarity Coefficient (DSC) between ground truth
test labels and threshold-based segmentation results.

Input:
- Ground truth labels in data/labelsTs/
- Threshold segmentation results in data/thresholded_labelsTs/

Output: DSC scores for each case and overall statistics

Author: Generated for threshold segmentation evaluation
"""

import sys
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class ThresholdEvaluator:
    def __init__(self, gt_dir, pred_dir):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)

        self.results = []
        self.case_scores = {}

    def compute_dice_coefficient(self, y_true, y_pred):
        """Compute Dice Similarity Coefficient between binary masks"""
        # Ensure binary masks
        y_true = (y_true > 0).astype(np.uint8)
        y_pred = (y_pred > 0).astype(np.uint8)

        # Compute intersection and union
        intersection = np.sum(y_true * y_pred)
        total = np.sum(y_true) + np.sum(y_pred)

        # Handle empty masks
        if total == 0:
            return 1.0 if intersection == 0 else 0.0

        # Compute Dice coefficient
        dice = (2.0 * intersection) / total
        return dice

    def load_nifti_data(self, file_path):
        """Load NIfTI file and return data array"""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def evaluate_case(self, case_name):
        """Evaluate a single case (both hemispheres)"""
        print(f"Evaluating {case_name}...")

        case_results = {}

        # Evaluate both hemispheres
        for hemisphere in ['L', 'R']:
            gt_file = self.gt_dir / f"{case_name}-{hemisphere}.nii"
            pred_file = self.pred_dir / f"{case_name}-{hemisphere}.nii"

            if not gt_file.exists():
                print(f"  Warning: Ground truth file not found: {gt_file}")
                continue
            if not pred_file.exists():
                print(f"  Warning: Prediction file not found: {pred_file}")
                continue

            # Load data
            gt_data = self.load_nifti_data(gt_file)
            pred_data = self.load_nifti_data(pred_file)

            if gt_data is None or pred_data is None:
                print(f"  Failed to load data for {hemisphere} hemisphere")
                continue

            # Compute DSC
            dice_score = self.compute_dice_coefficient(gt_data, pred_data)
            case_results[hemisphere] = dice_score

            print(f"  {hemisphere} hemisphere DSC: {dice_score:.4f}")

        # Compute case average
        if case_results:
            case_avg = np.mean(list(case_results.values()))
            case_results['average'] = case_avg
            print(f"  Case average DSC: {case_avg:.4f}")

        return case_results

    def run_evaluation(self):
        """Run evaluation on all test cases"""
        print("Threshold Segmentation Evaluation")
        print("=" * 50)
        print(f"Ground truth directory: {self.gt_dir}")
        print(f"Prediction directory: {self.pred_dir}")
        print()

        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False

        if not self.gt_dir.exists():
            print(f"ERROR: Ground truth directory does not exist: {self.gt_dir}")
            return False

        if not self.pred_dir.exists():
            print(f"ERROR: Prediction directory does not exist: {self.pred_dir}")
            return False

        # Find all ground truth files
        gt_files = list(self.gt_dir.glob("*.nii"))
        if not gt_files:
            print(f"ERROR: No .nii files found in {self.gt_dir}")
            return False

        # Extract unique case names
        case_names = set()
        for gt_file in gt_files:
            # Extract case name (remove hemisphere suffix)
            case_name = gt_file.stem.rsplit('-', 1)[0]  # Remove -L or -R
            case_names.add(case_name)

        print(f"Found {len(case_names)} test cases to evaluate")
        print()

        # Evaluate each case
        all_hemisphere_scores = []
        all_case_scores = []

        for case_name in sorted(case_names):
            case_results = self.evaluate_case(case_name)

            if case_results:
                self.case_scores[case_name] = case_results

                # Collect hemisphere scores
                for hemisphere in ['L', 'R']:
                    if hemisphere in case_results:
                        all_hemisphere_scores.append(case_results[hemisphere])

                # Collect case average scores
                if 'average' in case_results:
                    all_case_scores.append(case_results['average'])

            print()

        # Compute overall statistics
        print("=" * 50)
        print("EVALUATION SUMMARY:")
        print(f"Cases evaluated: {len(self.case_scores)}")
        print(f"Total hemisphere evaluations: {len(all_hemisphere_scores)}")

        if all_hemisphere_scores:
            print("\nHemisphere-wise Statistics:")
            print(f"  Mean DSC: {np.mean(all_hemisphere_scores):.4f}")
            print(f"  Std DSC:  {np.std(all_hemisphere_scores):.4f}")
            print(f"  Min DSC:  {np.min(all_hemisphere_scores):.4f}")
            print(f"  Max DSC:  {np.max(all_hemisphere_scores):.4f}")

        if all_case_scores:
            print("\nCase-wise Statistics:")
            print(f"  Mean DSC: {np.mean(all_case_scores):.4f}")
            print(f"  Std DSC:  {np.std(all_case_scores):.4f}")
            print(f"  Min DSC:  {np.min(all_case_scores):.4f}")
            print(f"  Max DSC:  {np.max(all_case_scores):.4f}")

        # Detailed results per case
        print("\nDetailed Results:")
        print("-" * 40)
        for case_name in sorted(self.case_scores.keys()):
            results = self.case_scores[case_name]
            print(f"{case_name}:")
            if 'L' in results:
                print(f"  Left:  {results['L']:.4f}")
            if 'R' in results:
                print(f"  Right: {results['R']:.4f}")
            if 'average' in results:
                print(f"  Avg:   {results['average']:.4f}")

        return True


def main():
    """Main function to run threshold segmentation evaluation"""
    script_dir = Path(__file__).parent

    # Define directories
    gt_dir = script_dir / "data" / "labelsTs"
    pred_dir = script_dir / "data" / "thresholded_labelsTs"

    print("Threshold Segmentation Evaluation Script")
    print("=" * 50)

    # Initialize evaluator
    evaluator = ThresholdEvaluator(gt_dir, pred_dir)

    # Run evaluation
    success = evaluator.run_evaluation()

    if success:
        print("\nEvaluation completed successfully!")
    else:
        print("Evaluation failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)