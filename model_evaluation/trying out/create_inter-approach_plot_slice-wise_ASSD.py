#!/usr/bin/env python3
"""
Create Inter-Approach Plot: Slice-wise ASSD Analysis Comparing Segmentation Approaches

Compares different segmentation approaches (Thresholding, Single-class, Single-class_halfdps,
Multi-class, Multi-label) using slice-wise ASSD for CBF input configuration.

Includes statistical testing using Wilcoxon signed-rank test.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
from scipy import stats
from datetime import datetime
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

class SliceWiseASSDApproachComparison:
    """Analyze slice-wise ASSD performance across different segmentation approaches"""

    def __init__(self, base_path, output_dir):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Segmentation approaches to analyze
        self.approach_patterns = {
            'Thresholding': 'thresholded_masks',
            'Single-class': 'nnUNet_results_Single-class_CBF',
            'Single-class_halfdps': 'nnUNet_results_Single-class_halfdps_CBF',
            'Multi-class': 'nnUNet_results_Multi-class_CBF',
            'Multi-label': 'nnUNet_results_Multi-label_CBF'
        }

        # Colors for each approach
        self.approach_colors = {
            'Thresholding': '#1f77b4',      # Blue
            'Single-class': '#ff7f0e',     # Orange
            'Single-class_halfdps': '#2ca02c', # Green
            'Multi-class': '#d62728',       # Red
            'Multi-label': '#9467bd'        # Purple
        }

        # Ground truth path
        self.gt_path = self.base_path / "nnUNet_raw_250827-PerfTerr-25" / "Dataset001_PerfusionTerritories" / "labelsTr"

    def compute_assd_2d(self, pred_slice, gt_slice, spacing):
        """Compute Average Symmetric Surface Distance for a 2D slice"""
        try:
            # Convert to binary if not already
            pred_binary = (pred_slice > 0.5).astype(np.uint8)
            gt_binary = (gt_slice > 0.5).astype(np.uint8)

            # Check if either mask is empty
            if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
                return 0.0  # Perfect match for empty masks
            elif np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
                return 50.0  # Large distance for empty vs non-empty

            # Get surface points (boundary pixels)
            pred_surface = pred_binary - binary_erosion(pred_binary)
            gt_surface = gt_binary - binary_erosion(gt_binary)

            # Get coordinates of surface points
            pred_coords = np.column_stack(np.where(pred_surface))
            gt_coords = np.column_stack(np.where(gt_surface))

            if len(pred_coords) == 0 or len(gt_coords) == 0:
                return 50.0

            # Apply spacing to get real-world coordinates
            pred_coords_real = pred_coords * spacing
            gt_coords_real = gt_coords * spacing

            # Compute directed Hausdorff distances
            dist_pred_to_gt = directed_hausdorff(pred_coords_real, gt_coords_real)[0]
            dist_gt_to_pred = directed_hausdorff(gt_coords_real, pred_coords_real)[0]

            # Average Symmetric Surface Distance
            assd = (dist_pred_to_gt + dist_gt_to_pred) / 2.0

            return assd

        except Exception as e:
            print(f"Error computing ASSD: {e}")
            return np.nan

    def find_validation_files(self):
        """Find all approach validation prediction and ground truth files across all folds"""
        validation_data = {}

        for approach_name, approach_pattern in self.approach_patterns.items():
            print(f"\nSearching for {approach_name} validation files...")
            validation_data[approach_name] = {'files': [], 'gt_files': []}

            approach_dir = self.base_path / approach_pattern

            if approach_name == 'Thresholding':
                # Direct files in thresholded_masks directory
                if approach_dir.exists():
                    print(f"Found thresholding directory: {approach_dir}")
                    pred_files = list(approach_dir.glob("*.nii"))
                    validation_data[approach_name]['files'].extend(pred_files)

                    # Ground truth files are in the shared GT directory
                    if self.gt_path.exists():
                        gt_files = list(self.gt_path.glob("*.nii"))
                        validation_data[approach_name]['gt_files'].extend(gt_files)
            else:
                # nnUNet approaches - process all folds (0-4)
                for fold in range(5):
                    if approach_name == 'Multi-label':
                        config_dir = approach_dir / "Dataset001_PerfusionTerritories" / "nnUNetTrainer_multilabel__nnUNetPlans__2d" / f"fold_{fold}" / "validation"
                    else:
                        config_dir = approach_dir / "Dataset001_PerfusionTerritories" / "nnUNetTrainer__nnUNetPlans__2d" / f"fold_{fold}" / "validation"

                    if config_dir.exists():
                        print(f"Found validation directory: {config_dir}")

                        # Find prediction files (*.nii files)
                        pred_files = list(config_dir.glob("*.nii"))
                        validation_data[approach_name]['files'].extend(pred_files)

                        # Ground truth files are in the shared GT directory (same for all folds)
                        if fold == 0 and self.gt_path.exists():  # Only add GT files once
                            gt_files = list(self.gt_path.glob("*.nii"))
                            validation_data[approach_name]['gt_files'].extend(gt_files)

                    else:
                        print(f"Validation directory not found: {config_dir}")

            print(f"Found {len(validation_data[approach_name]['files'])} prediction files")
            print(f"Found {len(validation_data[approach_name]['gt_files'])} ground truth files")

        return validation_data

    def load_and_match_files(self, validation_data):
        """Load NIfTI files and match predictions with ground truth"""
        matched_data = {}

        for approach_name, data in validation_data.items():
            if len(data['files']) == 0:
                print(f"No files found for {approach_name}")
                continue

            print(f"\nProcessing {approach_name}...")
            matched_data[approach_name] = []

            # Create mapping of case names to files
            pred_files = {}
            for pred_file in data['files']:
                # Extract case identifier from filename
                case_id = self.extract_case_id(pred_file.name, approach_name)
                if case_id not in pred_files:
                    pred_files[case_id] = []
                pred_files[case_id].append(pred_file)

            gt_files = {}
            for gt_file in data['gt_files']:
                case_id = self.extract_case_id_gt(gt_file.name)
                gt_files[case_id] = gt_file

            # Match predictions with ground truth
            for case_id, pred_file_list in pred_files.items():
                # For multi-label, case_id is like "PerfTerr001-v1", need to match with GT files "PerfTerr001-v1-L" and "PerfTerr001-v1-R"
                # For other approaches, case_id is like "PerfTerr001-v1-L", need to match with GT file "PerfTerr001-v1-L"

                if approach_name in ['Multi-label', 'Multi-class']:
                    # Multi-label and Multi-class have single files, need to match with both L and R GT files
                    base_case = case_id  # PerfTerr001-v1
                    gt_left = base_case + '-L'
                    gt_right = base_case + '-R'

                    if gt_left in gt_files and gt_right in gt_files:
                        for pred_file in pred_file_list:
                            # Add both hemispheres for this prediction file
                            matched_data[approach_name].append({
                                'case_id': case_id + '-L',  # Mark as left hemisphere
                                'pred_file': pred_file,
                                'gt_file': gt_files[gt_left]
                            })
                            matched_data[approach_name].append({
                                'case_id': case_id + '-R',  # Mark as right hemisphere
                                'pred_file': pred_file,
                                'gt_file': gt_files[gt_right]
                            })
                    else:
                        print(f"No matching GT found for {approach_name} prediction: {case_id}")
                else:
                    # Other approaches have separate files per hemisphere
                    if case_id in gt_files:
                        gt_file = gt_files[case_id]
                        for pred_file in pred_file_list:
                            matched_data[approach_name].append({
                                'case_id': case_id,
                                'pred_file': pred_file,
                                'gt_file': gt_file
                            })
                    else:
                        print(f"No matching GT found for prediction: {case_id}")

            print(f"Matched {len(matched_data[approach_name])} file pairs for {approach_name}")

        return matched_data

    def extract_case_id(self, filename, approach_name):
        """Extract case identifier from filename"""
        # Remove common suffixes
        case_id = filename.replace('.nii.gz', '').replace('.nii', '')

        if approach_name == 'Multi-label':
            # Multi-label format: PerfTerr001-v1.nii -> PerfTerr001-v1
            return case_id
        else:
            # Other formats: PerfTerr001-v1-L.nii -> PerfTerr001-v1-L
            return case_id

    def extract_case_id_gt(self, filename):
        """Extract case identifier from ground truth filename"""
        # Remove common suffixes
        case_id = filename.replace('.nii.gz', '').replace('.nii', '')
        return case_id

    def compute_slice_wise_assd(self, matched_data):
        """Compute ASSD for each slice in each matched file pair"""
        all_assd_data = []

        for approach_name, file_pairs in matched_data.items():
            if len(file_pairs) == 0:
                continue

            print(f"\nComputing slice-wise ASSD for {approach_name}...")

            for pair in file_pairs:
                try:
                    # Load NIfTI files
                    pred_nii = nib.load(pair['pred_file'])
                    gt_nii = nib.load(pair['gt_file'])

                    pred_data = pred_nii.get_fdata()
                    gt_data = gt_nii.get_fdata()

                    # Get spacing information
                    spacing = pred_nii.header.get_zooms()[:2]  # Only in-plane spacing for 2D

                    # Handle different output formats
                    if approach_name == 'Multi-label':
                        # Multi-label: single file with 4D output (H, W, D, channels)
                        # The case_id now includes hemisphere info (e.g., "PerfTerr001-v1-L")
                        if '-L' in pair['case_id']:
                            hemisphere = 'Left'
                            hemisphere_idx = 0  # Left hemisphere is channel 0
                        elif '-R' in pair['case_id']:
                            hemisphere = 'Right'
                            hemisphere_idx = 1  # Right hemisphere is channel 1
                        else:
                            continue

                        # For multi-label predictions, extract the appropriate channel
                        if pred_data.ndim == 4:  # 4D predictions (H, W, D, channels)
                            if pred_data.shape[3] > hemisphere_idx:
                                pred_hemisphere = pred_data[:, :, :, hemisphere_idx]
                            else:
                                continue
                        else:  # 3D predictions - use entire volume (binary prediction)
                            pred_hemisphere = pred_data

                        # GT data is already hemisphere-specific (single hemisphere file)
                        gt_hemisphere = gt_data.astype(np.uint8)

                        self.process_slices(pred_hemisphere, gt_hemisphere, spacing,
                                          pair['case_id'], hemisphere, approach_name, all_assd_data)

                    elif approach_name == 'Multi-class':
                        # Multi-class: single file with 3D multi-class output (values 0,1,2,3)
                        # Class 0: background, Class 1: perfusion_left, Class 2: perfusion_right, Class 3: perfusion_overlap
                        # Left hemisphere = Class 1 + Class 3, Right hemisphere = Class 2 + Class 3
                        if '-L' in pair['case_id']:
                            hemisphere = 'Left'
                            # Left hemisphere includes both perfusion_left (1) and perfusion_overlap (3)
                            pred_hemisphere = ((pred_data == 1) | (pred_data == 3)).astype(np.float64)
                        elif '-R' in pair['case_id']:
                            hemisphere = 'Right'
                            # Right hemisphere includes both perfusion_right (2) and perfusion_overlap (3)
                            pred_hemisphere = ((pred_data == 2) | (pred_data == 3)).astype(np.float64)
                        else:
                            continue

                        # GT data is already hemisphere-specific (single hemisphere file)
                        gt_hemisphere = gt_data.astype(np.uint8)

                        self.process_slices(pred_hemisphere, gt_hemisphere, spacing,
                                          pair['case_id'], hemisphere, approach_name, all_assd_data)
                    else:
                        # Single-class, Multi-class, Thresholding: process by hemisphere from filename
                        if '-L' in pair['case_id']:
                            hemisphere = 'Left'
                        elif '-R' in pair['case_id']:
                            hemisphere = 'Right'
                        else:
                            continue

                        # GT data is already hemisphere-specific (single hemisphere file)
                        gt_hemisphere = gt_data.astype(np.uint8)

                        self.process_slices(pred_data, gt_hemisphere, spacing,
                                          pair['case_id'], hemisphere, approach_name, all_assd_data)

                except Exception as e:
                    print(f"Error processing {pair['case_id']}: {e}")
                    continue

        return pd.DataFrame(all_assd_data)

    def process_slices(self, pred_data, gt_data, spacing, case_id, hemisphere, approach_name, all_assd_data):
        """Process individual slices and compute ASSD"""
        # Process each slice
        for slice_idx in range(pred_data.shape[2]):  # Assuming axial slices
            pred_slice = pred_data[:, :, slice_idx]
            gt_slice = gt_data[:, :, slice_idx]

            # Check if slice has any content in GT
            if np.sum(gt_slice) == 0:
                continue  # Skip empty ground truth slices

            # Compute ASSD for this slice
            assd = self.compute_assd_2d(pred_slice, gt_slice, spacing)

            if not np.isnan(assd) and not np.isinf(assd):
                all_assd_data.append({
                    'Approach': approach_name,
                    'Hemisphere': hemisphere,
                    'Case': case_id,
                    'Slice': slice_idx,
                    'ASSD': assd
                })

    def create_box_plot(self, assd_df):
        """Create box plot of slice-wise ASSD values comparing segmentation approaches"""
        if assd_df.empty:
            print("No ASSD data available for plotting")
            return

        print(f"\nCreating box plot with {len(assd_df)} slice-wise ASSD measurements...")
        print(f"Data summary by approach:")
        print(assd_df.groupby(['Approach', 'Hemisphere'])['ASSD'].agg(['count', 'mean', 'std']).round(3))

        # Set style
        plt.style.use('default')

        # Create the plot with exact same layout as inter-input plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grouped data structure: hemisphere first, then approach
        all_approaches = ['Thresholding', 'Single-class', 'Single-class_halfdps', 'Multi-class', 'Multi-label']
        hemispheres = ['Left', 'Right']

        # Prepare data for grouped plotting
        plot_positions = []
        plot_data_list = []
        plot_colors = []
        plot_labels = []

        position = 0
        hemisphere_positions = {}
        hemisphere_centers = {}

        for hemi_idx, hemisphere in enumerate(hemispheres):
            hemisphere_positions[hemisphere] = []
            start_pos = position

            for approach_idx, approach in enumerate(all_approaches):
                approach_data = assd_df[(assd_df['Hemisphere'] == hemisphere) & (assd_df['Approach'] == approach)]

                if not approach_data.empty:
                    plot_data_list.append(approach_data['ASSD'].values)
                else:
                    # Add empty data for missing approaches
                    plot_data_list.append([])

                plot_colors.append(self.approach_colors[approach])
                plot_labels.append(f'{hemisphere}_{approach}')
                plot_positions.append(position)
                hemisphere_positions[hemisphere].append(position)
                position += 0.7  # Spacing within hemisphere

            # Calculate hemisphere center for labeling
            hemisphere_centers[hemisphere] = (start_pos + position - 0.7) / 2

            # Add gap between hemispheres
            if hemi_idx < len(hemispheres) - 1:
                position += 0.3  # Gap between hemispheres

        # Create box plots manually with proper positioning
        box_parts = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=True,
            patch_artist=True,
            widths=0.5
        )

        # Color the boxes according to approach
        for patch, color in zip(box_parts['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Customize the plot
        ax.set_title('Segmentation Approaches: Slice-wise ASSD by Approach and Hemisphere\n'
                     'Cross-validation Results (CBF Input)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Hemisphere', fontsize=14, fontweight='bold')
        ax.set_ylabel('ASSD (mm)', fontsize=14, fontweight='bold')

        # Set custom x-axis labels for hemisphere groups
        ax.set_xticks([hemisphere_centers['Left'], hemisphere_centers['Right']])
        ax.set_xticklabels(['Left', 'Right'])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Add legend for approaches (back to original position)
        legend_handles = []
        for approach in all_approaches:
            handle = plt.Rectangle((0,0),1,1, facecolor=self.approach_colors[approach], alpha=0.7, edgecolor='black')
            legend_handles.append(handle)

        ax.legend(legend_handles, all_approaches, title='Segmentation Approach',
                 loc='upper right', fontsize=10, title_fontsize=11)

        # Add grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add median and IQR values below each box
        self.add_median_iqr_labels(ax, assd_df, hemisphere_positions)

        # Add sample size annotations
        self.add_sample_size_labels(ax, assd_df, hemisphere_positions)

        # Adjust y-axis limits to accommodate labels below (with much more space)
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]

        # Extend downward significantly for both median and sample size labels
        new_y_min = current_ylim[0] - y_range * 0.25

        ax.set_ylim(new_y_min, current_ylim[1])

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"slice_wise_assd_approach_comparison_{timestamp}.png"

        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"\nPlot saved to: {plot_path}")

        # Perform statistical testing
        stats_path = self.perform_statistical_testing(assd_df, timestamp)

        return plot_path, stats_path

    def add_median_iqr_labels(self, ax, assd_df, hemisphere_positions):
        """Add median [IQR] labels below each box plot to avoid legend overlap"""

        # Calculate positions for each box
        all_approaches = ['Thresholding', 'Single-class', 'Single-class_halfdps', 'Multi-class', 'Multi-label']
        hemispheres = ['Left', 'Right']

        for hemi_idx, hemisphere in enumerate(hemispheres):
            for approach_idx, approach in enumerate(all_approaches):
                if approach_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][approach_idx]

                    approach_data = assd_df[(assd_df['Approach'] == approach) & (assd_df['Hemisphere'] == hemisphere)]

                    if not approach_data.empty and len(approach_data) > 0:
                        values = approach_data['ASSD'].dropna()

                        if len(values) > 0:
                            median = values.median()
                            q1 = values.quantile(0.25)
                            q3 = values.quantile(0.75)

                            # Position well below the plot area to avoid lower whiskers
                            y_min = ax.get_ylim()[0]
                            y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                            y_pos = y_min - y_offset

                            # Format label: median [Q1-Q3] as in the reference plot
                            label = f'{median:.1f} [{q1:.1f}-{q3:.1f}]'

                            # Create box with color matching the approach
                            bbox_props = dict(boxstyle="round,pad=0.2",
                                            facecolor='white',
                                            alpha=0.9,
                                            edgecolor=self.approach_colors[approach],
                                            linewidth=1.0)

                            ax.text(box_position, y_pos, label,
                                   ha='center', va='top', fontsize=8,
                                   color=self.approach_colors[approach], weight='bold',
                                   bbox=bbox_props)

    def add_sample_size_labels(self, ax, assd_df, hemisphere_positions):
        """Add sample size (n=) labels below each box plot with black styling"""

        # Calculate positions for each box
        all_approaches = ['Thresholding', 'Single-class', 'Single-class_halfdps', 'Multi-class', 'Multi-label']
        hemispheres = ['Left', 'Right']

        for hemi_idx, hemisphere in enumerate(hemispheres):
            for approach_idx, approach in enumerate(all_approaches):
                if approach_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][approach_idx]

                    approach_data = assd_df[(assd_df['Approach'] == approach) & (assd_df['Hemisphere'] == hemisphere)]

                    if not approach_data.empty:
                        n = len(approach_data)

                        # Position below the median labels, further down
                        y_min = ax.get_ylim()[0]
                        y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
                        y_pos = y_min - y_offset

                        # Format label: n=XXX
                        label = f'n={n}'

                        # Create box with black outline for sample size labels
                        bbox_props = dict(boxstyle="round,pad=0.2",
                                        facecolor='white',
                                        alpha=0.9,
                                        edgecolor='black',
                                        linewidth=1.0)

                        ax.text(box_position, y_pos, label,
                               ha='center', va='top', fontsize=8,
                               color='black', weight='bold',
                               bbox=bbox_props)

    def perform_statistical_testing(self, assd_df, timestamp):
        """Perform Wilcoxon signed-rank test comparing segmentation approaches (slice-wise)"""

        print("\nPerforming Statistical Testing")
        print("=" * 50)
        print("Using Wilcoxon signed-rank test for pairwise comparisons")
        print("Slice-wise ASSD comparison between approaches")
        print("=" * 50)

        all_stats_results = []

        # Process each hemisphere separately
        for hemisphere in ['Left', 'Right']:
            print(f"\n{hemisphere} Hemisphere Analysis:")
            print("-" * 30)

            # Get available approaches with data for this hemisphere
            hemi_data = assd_df[assd_df['Hemisphere'] == hemisphere]
            available_approaches = hemi_data['Approach'].unique().tolist()

            if len(available_approaches) < 2:
                print(f"  Insufficient data for comparisons (need e2 approaches)")
                continue

            print(f"  Available approaches: {', '.join(available_approaches)}")

            # Perform all pairwise comparisons
            comparison_results = []

            for approach1, approach2 in combinations(available_approaches, 2):
                data1_df = hemi_data[hemi_data['Approach'] == approach1]
                data2_df = hemi_data[hemi_data['Approach'] == approach2]

                # Find paired slices (same case ID and slice number between approaches)
                # Merge dataframes on Case and Slice to get slice-wise pairs
                merged_df = data1_df.merge(data2_df, on=['Case', 'Slice'], suffixes=('_1', '_2'))

                if len(merged_df) < 10:  # Minimum sample size for Wilcoxon test
                    print(f"    {approach1} vs {approach2}: Insufficient paired slices (n={len(merged_df)})")
                    continue

                # Get paired slice-wise data directly
                paired_data1 = merged_df['ASSD_1'].values
                paired_data2 = merged_df['ASSD_2'].values

                try:
                    print(f"    Paired slices: {len(paired_data1)} (from {len(data1_df)} and {len(data2_df)} total slices)")

                    # Wilcoxon signed-rank test for paired slice data
                    statistic, p_value = stats.wilcoxon(paired_data1, paired_data2, alternative='two-sided')

                    # Calculate effect size (r = Z / sqrt(N))
                    n = len(paired_data1)
                    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0  # Cap extreme values
                    effect_size = z_score / np.sqrt(n)

                    # Calculate medians and difference (should now match plot values)
                    median1 = np.median(paired_data1)
                    median2 = np.median(paired_data2)
                    median_diff = median1 - median2

                    # Determine significance
                    significance = ""
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"

                    comparison_results.append({
                        'Hemisphere': hemisphere,
                        'Approach1': approach1,
                        'Approach2': approach2,
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

                    print(f"    {approach1} vs {approach2}:")
                    print(f"      Paired slices: {n}")
                    print(f"      Medians: {median1:.3f} vs {median2:.3f} (diff: {median_diff:+.3f})")
                    print(f"      Wilcoxon W: {statistic:.2f}, p-value: {p_value:.6f} {significance}")
                    print(f"      Effect size (r): {effect_size:.3f}")

                except Exception as e:
                    print(f"    {approach1} vs {approach2}: Error - {e}")

            all_stats_results.extend(comparison_results)

        # Save statistical results to Excel
        if all_stats_results:
            stats_df = pd.DataFrame(all_stats_results)

            # Sort by hemisphere and p-value
            stats_df = stats_df.sort_values(['Hemisphere', 'P_Value'])

            # Add interpretation column
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
                group['P_Value_Bonferroni'] = np.minimum(group['P_Value_Bonferroni'], 1.0)  # Cap at 1.0

                # Update significance with Bonferroni correction
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

            # Save to Excel with timestamp
            stats_path = self.output_dir / f"slice_wise_assd_approach_statistical_comparison_{timestamp}.xlsx"

            # Create comprehensive Excel file with multiple sheets
            with pd.ExcelWriter(stats_path, engine='openpyxl') as writer:
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

            print(f"\n{'='*60}")
            print(f"STATISTICAL RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Total pairwise comparisons: {len(stats_df)}")
            print(f"Significant at p < 0.05: {sum(stats_df['P_Value'] < 0.05)} ({100*sum(stats_df['P_Value'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Significant at p < 0.01: {sum(stats_df['P_Value'] < 0.01)} ({100*sum(stats_df['P_Value'] < 0.01)/len(stats_df):.1f}%)")
            print(f"Significant with Bonferroni correction (p < 0.05): {sum(stats_df['P_Value_Bonferroni'] < 0.05)} ({100*sum(stats_df['P_Value_Bonferroni'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Average paired slices per comparison: {stats_df['N_Paired_Slices'].mean():.1f}")

            print(f"\nStatistical results saved to: {stats_path}")

            return stats_path
        else:
            print("No statistical comparisons could be performed.")
            return None

    def run_analysis(self):
        """Run the complete slice-wise ASSD analysis"""

        print("Starting slice-wise ASSD analysis for approach comparison...")
        print("="*80)

        # Find all validation files
        validation_data = self.find_validation_files()

        if not validation_data:
            print("No validation files found!")
            return

        # Load and match files
        matched_data = self.load_and_match_files(validation_data)

        # Compute slice-wise ASSD
        assd_df = self.compute_slice_wise_assd(matched_data)

        if not assd_df.empty:
            # Create box plot and perform statistical testing
            plot_path, stats_path = self.create_box_plot(assd_df)
            print(f"\nSlice-wise ASSD approach comparison analysis completed!")
            print(f"Total slices analyzed: {len(assd_df)}")
            print(f"Plot saved to: {plot_path}")
            if stats_path:
                print(f"Statistical analysis saved to: {stats_path}")

            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = self.output_dir / f"slice_wise_assd_approach_comparison_data_{timestamp}.xlsx"
            assd_df.to_excel(data_path, index=False)
            print(f"Raw data saved to: {data_path}")

            return assd_df, plot_path, stats_path
        else:
            print("No ASSD data could be computed. Check file paths and data format.")
            return None

def main():
    """Main function to run slice-wise ASSD analysis"""

    # Set paths
    base_path = Path("/home/ubuntu/DLSegPerf/data/TrainingsResults-PerfTerr")
    output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/trying out")

    # Initialize analyzer
    analyzer = SliceWiseASSDApproachComparison(base_path, output_dir)

    # Run analysis
    results = analyzer.run_analysis()

    if results:
        assd_df, plot_path, stats_results = results
        print(f"\nApproach comparison analysis completed successfully!")
        print(f"Results saved in: {output_dir}")
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    main()