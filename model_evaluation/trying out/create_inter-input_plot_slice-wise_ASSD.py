#!/usr/bin/env python3
"""
Inter-Input Configuration Slice-wise ASSD Box Plot Generator

Creates box plots comparing slice-wise Average Symmetric Surface Distance (ASSD)
across different input configurations for single-class segmentation approach only.

Each data point represents one slice (not one volume), providing much higher sample sizes.
Computes ASSD from raw prediction and ground truth NIfTI files in TrainingsResults-PerfTerr.

Input configurations compared: CBF, CBF+T1w, CBF+FLAIR, CBF+T1w+FLAIR
Hemispheres: Left and Right (analyzed separately)
Approach: Single-Class only

Output: Box plot showing slice-wise ASSD distribution per input configuration and hemisphere.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
from scipy import stats
from itertools import combinations
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class SlicewiseASSDInputPlotter:
    """Generate slice-wise ASSD box plots comparing input configurations for single-class segmentation"""

    def __init__(self, results_dir):
        """Initialize with TrainingsResults-PerfTerr directory path"""
        self.results_dir = Path(results_dir)
        self.output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/trying out")
        self.output_dir.mkdir(exist_ok=True)

        # Define input configuration patterns for single-class only
        self.config_patterns = {
            'CBF': 'nnUNet_results_Single-class_CBF',
            'CBF+T1w': 'nnUNet_results_Single-class_CBF_T1w',
            'CBF+FLAIR': 'nnUNet_results_Single-class_CBF_FLAIR',
            'CBF+T1w+FLAIR': 'nnUNet_results_Single-class_CBF_T1w_FLAIR'
        }

        # Ground truth path
        self.gt_path = self.results_dir / "nnUNet_raw_250827-PerfTerr-25" / "Dataset001_PerfusionTerritories" / "labelsTr"

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
                return np.inf  # Infinite distance for empty vs non-empty

            # Get surface points (boundary pixels)
            pred_surface = pred_binary - binary_erosion(pred_binary)
            gt_surface = gt_binary - binary_erosion(gt_binary)

            # Get coordinates of surface points
            pred_coords = np.column_stack(np.where(pred_surface))
            gt_coords = np.column_stack(np.where(gt_surface))

            if len(pred_coords) == 0 or len(gt_coords) == 0:
                return np.inf

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
        """Find all single-class validation prediction and ground truth files across all folds"""
        validation_data = {}

        for config_name, config_pattern in self.config_patterns.items():
            print(f"\nSearching for {config_name} validation files...")
            validation_data[config_name] = {'files': [], 'gt_files': []}

            # Process all folds (0-4)
            for fold in range(5):
                config_dir = self.results_dir / config_pattern / "Dataset001_PerfusionTerritories" / "nnUNetTrainer__nnUNetPlans__2d" / f"fold_{fold}" / "validation"

                if config_dir.exists():
                    print(f"Found validation directory: {config_dir}")

                    # Find prediction files (*.nii files)
                    pred_files = list(config_dir.glob("*.nii"))
                    validation_data[config_name]['files'].extend(pred_files)

                    # Ground truth files are in the shared GT directory (same for all folds)
                    if fold == 0 and self.gt_path.exists():  # Only add GT files once
                        gt_files = list(self.gt_path.glob("*.nii"))
                        validation_data[config_name]['gt_files'].extend(gt_files)

                else:
                    print(f"Validation directory not found: {config_dir}")

            print(f"Found {len(validation_data[config_name]['files'])} prediction files across all folds")
            print(f"Found {len(validation_data[config_name]['gt_files'])} ground truth files")

        return validation_data

    def load_and_match_files(self, validation_data):
        """Load NIfTI files and match predictions with ground truth"""
        matched_data = {}

        for config_name, data in validation_data.items():
            if len(data['files']) == 0:
                print(f"No files found for {config_name}")
                continue

            print(f"\nProcessing {config_name}...")
            matched_data[config_name] = []

            # Create mapping of case names to files
            pred_files = {}
            for pred_file in data['files']:
                # Extract case identifier from filename
                case_id = self.extract_case_id(pred_file.name)
                pred_files[case_id] = pred_file

            gt_files = {}
            for gt_file in data['gt_files']:
                case_id = self.extract_case_id(gt_file.name)
                gt_files[case_id] = gt_file

            # Match predictions with ground truth
            for case_id, pred_file in pred_files.items():
                if case_id in gt_files:
                    gt_file = gt_files[case_id]
                    matched_data[config_name].append({
                        'case_id': case_id,
                        'pred_file': pred_file,
                        'gt_file': gt_file
                    })
                else:
                    print(f"No matching GT found for prediction: {case_id}")

            print(f"Matched {len(matched_data[config_name])} file pairs for {config_name}")

        return matched_data

    def extract_case_id(self, filename):
        """Extract case identifier from filename"""
        # Remove common suffixes and prefixes
        case_id = filename.replace('.nii.gz', '').replace('.nii', '')
        case_id = case_id.replace('_gt', '').replace('gt_', '')

        # Handle the specific format: PerfTerr001-v1-L.nii -> PerfTerr001-v1-L
        if case_id.startswith('PerfTerr'):
            return case_id

        # Look for patterns like PerfTerr001-v1-L, etc.
        import re
        match = re.search(r'(PerfTerr\d+-v\d+-[LR])', case_id)
        if match:
            return match.group(1)

        return case_id

    def compute_slice_wise_assd(self, matched_data):
        """Compute ASSD for each slice in each matched file pair"""
        all_assd_data = []

        for config_name, file_pairs in matched_data.items():
            if len(file_pairs) == 0:
                continue

            print(f"\nComputing slice-wise ASSD for {config_name}...")

            for pair in file_pairs:
                try:
                    # Load NIfTI files
                    pred_nii = nib.load(pair['pred_file'])
                    gt_nii = nib.load(pair['gt_file'])

                    pred_data = pred_nii.get_fdata()
                    gt_data = gt_nii.get_fdata()

                    # Get spacing information
                    spacing = pred_nii.header.get_zooms()[:2]  # Only in-plane spacing for 2D

                    # Ensure same shape
                    if pred_data.shape != gt_data.shape:
                        print(f"Shape mismatch for {pair['case_id']}: {pred_data.shape} vs {gt_data.shape}")
                        continue

                    # Process each slice
                    for slice_idx in range(pred_data.shape[2]):  # Assuming axial slices
                        pred_slice = pred_data[:, :, slice_idx]
                        gt_slice = gt_data[:, :, slice_idx]

                        # Check if slice has any content
                        if np.sum(gt_slice) == 0:
                            continue  # Skip empty ground truth slices

                        # Determine hemisphere from filename (PerfTerr001-v1-L or PerfTerr001-v1-R)
                        if '-L' in pair['case_id']:
                            hemisphere = 'Left'
                        elif '-R' in pair['case_id']:
                            hemisphere = 'Right'
                        else:
                            hemisphere = 'Unknown'

                        # Compute ASSD for this slice
                        assd = self.compute_assd_2d(pred_slice, gt_slice, spacing)

                        if not np.isnan(assd) and not np.isinf(assd):
                            all_assd_data.append({
                                'Input_Configuration': config_name,
                                'Hemisphere': hemisphere,
                                'Case': pair['case_id'],
                                'Slice': slice_idx,
                                'ASSD': assd
                            })

                except Exception as e:
                    print(f"Error processing {pair['case_id']}: {e}")
                    continue

        return pd.DataFrame(all_assd_data)

    def determine_hemisphere(self, gt_slice):
        """Determine hemisphere based on slice content"""
        # Simple heuristic: check which side of the slice has more content
        width = gt_slice.shape[1]
        left_side = gt_slice[:, :width//2]
        right_side = gt_slice[:, width//2:]

        left_sum = np.sum(left_side)
        right_sum = np.sum(right_side)

        if left_sum > right_sum:
            return 'Left'
        elif right_sum > left_sum:
            return 'Right'
        else:
            # If equal, default to Left
            return 'Left'

    def create_box_plot(self, assd_df):
        """Create box plot of slice-wise ASSD values comparing input configurations"""
        if assd_df.empty:
            print("No ASSD data available for plotting")
            return

        print(f"\nCreating box plot with {len(assd_df)} slice-wise ASSD measurements...")
        print(f"Data summary by input configuration:")
        print(assd_df.groupby(['Input_Configuration', 'Hemisphere'])['ASSD'].agg(['count', 'mean', 'std']).round(3))

        # Color palette for input configurations (match inter-input plot exactly)
        input_colors = {
            'CBF': '#1f77b4',           # Blue
            'CBF+T1w': '#ff7f0e',       # Orange
            'CBF+FLAIR': '#2ca02c',     # Green
            'CBF+T1w+FLAIR': '#d62728'  # Red
        }

        # Set style
        plt.style.use('default')

        # Create the plot with exact same layout as inter-input plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grouped data structure: hemisphere first, then input config
        all_configs = ['CBF', 'CBF+T1w', 'CBF+FLAIR', 'CBF+T1w+FLAIR']
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

            for config_idx, config in enumerate(all_configs):
                config_data = assd_df[(assd_df['Hemisphere'] == hemisphere) & (assd_df['Input_Configuration'] == config)]

                if not config_data.empty:
                    plot_data_list.append(config_data['ASSD'].values)
                else:
                    # Add empty data for missing configurations
                    plot_data_list.append([])

                plot_colors.append(input_colors[config])
                plot_labels.append(f'{hemisphere}_{config}')
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

        # Color the boxes according to input configuration
        for patch, color in zip(box_parts['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Customize the plot (match inter-input plot style exactly)
        ax.set_title('Single-Class Segmentation: Slice-wise ASSD by Input Configuration and Hemisphere\n'
                     'Cross-validation Results',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Hemisphere', fontsize=14, fontweight='bold')
        ax.set_ylabel('ASSD (mm)', fontsize=14, fontweight='bold')

        # Set custom x-axis labels for hemisphere groups
        ax.set_xticks([hemisphere_centers['Left'], hemisphere_centers['Right']])
        ax.set_xticklabels(['Left', 'Right'])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Add legend for input configurations (back to original position)
        legend_handles = []
        for config in all_configs:
            handle = plt.Rectangle((0,0),1,1, facecolor=input_colors[config], alpha=0.7, edgecolor='black')
            legend_handles.append(handle)

        ax.legend(legend_handles, all_configs, title='Input Configuration',
                 loc='upper right', fontsize=10, title_fontsize=11)

        # Add grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add median and IQR values above each box
        self.add_median_iqr_labels(ax, assd_df, hemisphere_positions, input_colors)

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
        plot_path = self.output_dir / f"slice_wise_assd_input_comparison_singleclass_{timestamp}.png"

        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"\nPlot saved to: {plot_path}")

        # Perform statistical testing
        stats_path = self.perform_statistical_testing(assd_df, timestamp)

        return plot_path, stats_path

    def add_median_iqr_labels(self, ax, assd_df, hemisphere_positions, input_colors):
        """Add median [IQR] labels below each box plot to avoid legend overlap"""

        # Calculate positions for each box
        all_configs = ['CBF', 'CBF+T1w', 'CBF+FLAIR', 'CBF+T1w+FLAIR']
        hemispheres = ['Left', 'Right']

        for hemi_idx, hemisphere in enumerate(hemispheres):
            for config_idx, config in enumerate(all_configs):
                if config_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][config_idx]

                    config_data = assd_df[(assd_df['Input_Configuration'] == config) & (assd_df['Hemisphere'] == hemisphere)]

                    if not config_data.empty and len(config_data) > 0:
                        values = config_data['ASSD'].dropna()

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

                            # Create box with color matching the input configuration
                            bbox_props = dict(boxstyle="round,pad=0.2",
                                            facecolor='white',
                                            alpha=0.9,
                                            edgecolor=input_colors[config],
                                            linewidth=1.0)

                            ax.text(box_position, y_pos, label,
                                   ha='center', va='top', fontsize=8,
                                   color=input_colors[config], weight='bold',
                                   bbox=bbox_props)

    def add_sample_size_labels(self, ax, assd_df, hemisphere_positions):
        """Add sample size (n=) labels below each box plot with color-matched styling"""

        # Calculate positions for each box
        all_configs = ['CBF', 'CBF+T1w', 'CBF+FLAIR', 'CBF+T1w+FLAIR']
        hemispheres = ['Left', 'Right']

        # Input colors for styling
        input_colors = {
            'CBF': '#1f77b4',           # Blue
            'CBF+T1w': '#ff7f0e',       # Orange
            'CBF+FLAIR': '#2ca02c',     # Green
            'CBF+T1w+FLAIR': '#d62728'  # Red
        }

        for hemi_idx, hemisphere in enumerate(hemispheres):
            for config_idx, config in enumerate(all_configs):
                if config_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][config_idx]

                    config_data = assd_df[(assd_df['Input_Configuration'] == config) & (assd_df['Hemisphere'] == hemisphere)]

                    if not config_data.empty:
                        n = len(config_data)

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
        """Perform Wilcoxon signed-rank test comparing input configurations"""

        print("\nPerforming Statistical Testing")
        print("=" * 50)
        print("Using Wilcoxon signed-rank test for pairwise comparisons")
        print("Same cases, different input configurations")
        print("=" * 50)

        all_stats_results = []

        # Process each hemisphere separately
        for hemisphere in ['Left', 'Right']:
            print(f"\n{hemisphere} Hemisphere Analysis:")
            print("-" * 30)

            # Get available configurations with data for this hemisphere
            hemi_data = assd_df[assd_df['Hemisphere'] == hemisphere]
            available_configs = hemi_data['Input_Configuration'].unique().tolist()

            if len(available_configs) < 2:
                print(f"  Insufficient data for comparisons (need ≥2 configurations)")
                continue

            print(f"  Available configurations: {', '.join(available_configs)}")

            # Perform all pairwise comparisons
            comparison_results = []

            for config1, config2 in combinations(available_configs, 2):
                data1_df = hemi_data[hemi_data['Input_Configuration'] == config1]
                data2_df = hemi_data[hemi_data['Input_Configuration'] == config2]

                # Find paired cases (same case ID between configurations)
                common_cases = set(data1_df['Case']) & set(data2_df['Case'])

                if len(common_cases) < 5:  # Minimum sample size for Wilcoxon test
                    print(f"    {config1} vs {config2}: Insufficient paired cases (n={len(common_cases)})")
                    continue

                # Get paired data by averaging slices within each case
                paired_data1 = []
                paired_data2 = []

                for case_id in common_cases:
                    case1_assd = data1_df[data1_df['Case'] == case_id]['ASSD'].mean()
                    case2_assd = data2_df[data2_df['Case'] == case_id]['ASSD'].mean()
                    paired_data1.append(case1_assd)
                    paired_data2.append(case2_assd)

                try:
                    print(f"    Paired cases: {len(paired_data1)} (from {len(data1_df)} and {len(data2_df)} total slices)")

                    # Wilcoxon signed-rank test for paired data
                    statistic, p_value = stats.wilcoxon(paired_data1, paired_data2, alternative='two-sided')

                    # Calculate effect size (r = Z / sqrt(N))
                    n = len(paired_data1)
                    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0  # Cap extreme values
                    effect_size = z_score / np.sqrt(n)

                    # Calculate medians and difference
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
                        'Config1': config1,
                        'Config2': config2,
                        'Median1': median1,
                        'Median2': median2,
                        'Median_Diff': median_diff,
                        'Statistic': statistic,
                        'P_Value': p_value,
                        'Effect_Size': effect_size,
                        'Significance': significance,
                        'N_Paired': n,
                        'N1_Total_Slices': len(data1_df),
                        'N2_Total_Slices': len(data2_df)
                    })

                    print(f"    {config1} vs {config2}:")
                    print(f"      Paired cases: {n}")
                    print(f"      Medians: {median1:.3f} vs {median2:.3f} (diff: {median_diff:+.3f})")
                    print(f"      Wilcoxon W: {statistic:.2f}, p-value: {p_value:.6f} {significance}")
                    print(f"      Effect size (r): {effect_size:.3f}")

                except Exception as e:
                    print(f"    {config1} vs {config2}: Error - {e}")

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
            stats_path = self.output_dir / f"slice_wise_assd_statistical_comparison_{timestamp}.xlsx"

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
                    'N_Paired': 'mean'
                }).round(4)

                summary_by_hemisphere.columns = ['Total_Comparisons', 'Significant_p05', 'Significant_p01',
                                               'Bonferroni_Significant', 'Mean_Effect_Size', 'Std_Effect_Size',
                                               'Avg_Paired_Cases']
                summary_by_hemisphere.to_excel(writer, sheet_name='Summary_by_Hemisphere')

            print(f"\n{'='*60}")
            print(f"STATISTICAL RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Total pairwise comparisons: {len(stats_df)}")
            print(f"Significant at p < 0.05: {sum(stats_df['P_Value'] < 0.05)} ({100*sum(stats_df['P_Value'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Significant at p < 0.01: {sum(stats_df['P_Value'] < 0.01)} ({100*sum(stats_df['P_Value'] < 0.01)/len(stats_df):.1f}%)")
            print(f"Significant with Bonferroni correction (p < 0.05): {sum(stats_df['P_Value_Bonferroni'] < 0.05)} ({100*sum(stats_df['P_Value_Bonferroni'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Average paired cases per comparison: {stats_df['N_Paired'].mean():.1f}")

            print(f"\nStatistical results saved to: {stats_path}")

            return stats_path
        else:
            print("No statistical comparisons could be performed.")
            return None

def main():
    """Main function to generate slice-wise ASSD input configuration comparison plots"""
    print("Slice-wise ASSD Input Configuration Comparison for Single-Class Segmentation")
    print("=" * 75)

    # Initialize plotter
    results_dir = Path("/home/ubuntu/DLSegPerf/data/TrainingsResults-PerfTerr")
    plotter = SlicewiseASSDInputPlotter(results_dir)

    # Find validation files
    validation_data = plotter.find_validation_files()

    # Load and match files
    matched_data = plotter.load_and_match_files(validation_data)

    # Compute slice-wise ASSD
    assd_df = plotter.compute_slice_wise_assd(matched_data)

    if not assd_df.empty:
        # Create box plot and perform statistical testing
        plot_path, stats_path = plotter.create_box_plot(assd_df)
        print(f"\nSlice-wise ASSD input configuration analysis completed!")
        print(f"Total slices analyzed: {len(assd_df)}")
        print(f"Plot saved to: {plot_path}")
        if stats_path:
            print(f"Statistical analysis saved to: {stats_path}")
    else:
        print("No ASSD data could be computed. Check file paths and data format.")

if __name__ == "__main__":
    main()