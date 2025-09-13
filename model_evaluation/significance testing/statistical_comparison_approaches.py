#!/usr/bin/env python3
"""
Statistical Comparison of Segmentation Approaches using Wilcoxon Signed-Rank Test

Performs comprehensive statistical testing comparing different segmentation approaches:
- threshold segmentation
- single-class
- single-class halfdps
- multi-class
- multi-label

Uses Wilcoxon signed-rank test for paired comparisons (same cases, different approaches).
Hemispheres are analyzed separately with Bonferroni correction for multiple comparisons.

Output: Excel file with comprehensive statistical analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from scipy import stats
from itertools import combinations
warnings.filterwarnings('ignore')

class StatisticalComparator:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(__file__).parent

        # Input channel configurations (match actual file naming)
        self.input_configs = {
            'CBF': ['CBF_results_'],
            'CBF+T1w': ['CBF_T1w_results_'],
            'CBF+FLAIR': ['CBF_FLAIR_results_'],
            'CBF+T1w+FLAIR': ['CBF_T1w_FLAIR_results_']
        }

        # Segmentation approaches and their patterns
        self.approaches = {
            'Threshold': 'thresholdseg_results_',
            'Single-Class': 'crossval_singleclass_',
            'Single-Class HalfDPS': 'crossval_singleclass_halfdps_',
            'Multi-Class': 'crossval_multiclass_',
            'Multi-Label': 'crossval_multilabel_'
        }

    def find_excel_files(self):
        """Find and categorize all Excel files by input config and approach"""
        files = {}

        # Initialize structure for each input configuration
        for config_name in self.input_configs.keys():
            files[config_name] = {}

            for approach_name, approach_pattern in self.approaches.items():
                matches = []

                if approach_name == 'Threshold':
                    # Threshold segmentation is input-agnostic (CBF only)
                    if config_name == 'CBF':
                        search_pattern = f"{approach_pattern}*.xlsx"
                        found_files = list(self.results_dir.glob(search_pattern))
                        matches.extend(found_files)
                else:
                    # Look for files matching the input config patterns
                    for config_pattern in self.input_configs[config_name]:
                        search_pattern = f"{approach_pattern}{config_pattern}*.xlsx"
                        found_files = list(self.results_dir.glob(search_pattern))

                        # For CBF-only, ensure we don't match multi-modal files
                        if config_name == 'CBF':
                            found_files = [f for f in found_files
                                         if 'T1w' not in f.name and 'FLAIR' not in f.name]

                        matches.extend(found_files)

                if matches:
                    # Use the most recent file if multiple exist
                    files[config_name][approach_name] = sorted(matches)[-1]
                    print(f"Found {config_name} {approach_name}: {matches[-1].name}")

        return files

    def load_hemisphere_data(self, file_path, approach):
        """Load hemisphere-specific data from Excel file"""
        try:
            if approach in ['Single-Class', 'Single-Class HalfDPS', 'Multi-Label']:
                # Load Per_Case_Details sheet with Hemisphere column
                df = pd.read_excel(file_path, sheet_name='Per_Case_Details')

                # Extract data for each hemisphere, preserving case information
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi][['Subject', 'Visit', 'DSC_Volume']].copy()
                    hemi_data['Case_ID'] = hemi_data['Subject'] + '_' + hemi_data['Visit']
                    data[hemi] = hemi_data

                return data

            elif approach == 'Multi-Class':
                # Load Per_Hemisphere_Details sheet
                df = pd.read_excel(file_path, sheet_name='Per_Hemisphere_Details')

                # Extract data for each hemisphere, preserving case information
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi][['Subject', 'Visit', 'DSC_Volume']].copy()
                    hemi_data['Case_ID'] = hemi_data['Subject'] + '_' + hemi_data['Visit']
                    data[hemi] = hemi_data

                return data

            elif approach == 'Threshold':
                # Load Per_Case_Results sheet
                df = pd.read_excel(file_path, sheet_name='Per_Case_Results')

                # Extract data for each hemisphere, preserving case information
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi][['Subject', 'Visit', 'DSC_Volume']].copy()
                    hemi_data['Case_ID'] = hemi_data['Subject'] + '_' + hemi_data['Visit']
                    data[hemi] = hemi_data

                return data

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {'Left': pd.DataFrame(), 'Right': pd.DataFrame()}

    def align_paired_data(self, data1, data2):
        """Align data from two approaches based on Case_ID for paired comparison"""

        # Merge on Case_ID to get paired data
        merged = pd.merge(data1, data2, on='Case_ID', suffixes=('_approach1', '_approach2'))

        if len(merged) == 0:
            return np.array([]), np.array([])

        # Extract paired DSC values
        values1 = merged['DSC_Volume_approach1'].values
        values2 = merged['DSC_Volume_approach2'].values

        print(f"    Paired cases: {len(merged)} (from {len(data1)} and {len(data2)} total)")

        return values1, values2

    def perform_comprehensive_testing(self):
        """Perform comprehensive Wilcoxon signed-rank tests across all configurations"""

        print("Comprehensive Statistical Comparison of Segmentation Approaches")
        print("=" * 70)
        print("Using Wilcoxon signed-rank test for paired comparisons")
        print("Same cases, different approaches - hemispheres analyzed separately")
        print("=" * 70)

        # Find all Excel files
        files_dict = self.find_excel_files()

        if not any(files_dict.values()):
            print("No Excel files found for statistical testing!")
            return None

        # Collect all statistical results
        all_stats_results = []

        # Process each input configuration
        for config_name, approach_files in files_dict.items():
            if not any(approach_files.values()):
                print(f"No files found for {config_name}")
                continue

            print(f"\n{'='*60}")
            print(f"STATISTICAL ANALYSIS: {config_name}")
            print(f"{'='*60}")

            # Load data for all approaches in this configuration
            approach_data = {}
            approach_order = ['Threshold', 'Single-Class', 'Single-Class HalfDPS', 'Multi-Class', 'Multi-Label']

            for approach_name in approach_order:
                if approach_name in approach_files and approach_files[approach_name]:
                    file_path = approach_files[approach_name]
                    hemi_data = self.load_hemisphere_data(file_path, approach_name)
                    approach_data[approach_name] = hemi_data

            # Perform pairwise comparisons for each hemisphere
            for hemisphere in ['Left', 'Right']:
                print(f"\n{hemisphere} Hemisphere Analysis:")
                print("-" * 30)

                # Get available approaches with data for this hemisphere
                available_approaches = []
                hemisphere_datasets = {}

                for approach_name in approach_order:
                    if (approach_name in approach_data and
                        hemisphere in approach_data[approach_name] and
                        len(approach_data[approach_name][hemisphere]) > 0):

                        # Special handling for Threshold (CBF only)
                        if approach_name == 'Threshold' and config_name != 'CBF':
                            continue

                        available_approaches.append(approach_name)
                        hemisphere_datasets[approach_name] = approach_data[approach_name][hemisphere]

                if len(available_approaches) < 2:
                    print(f"  Insufficient data for comparisons (need ≥2 approaches)")
                    continue

                print(f"  Available approaches: {', '.join(available_approaches)}")
                print(f"  Sample sizes: {', '.join([f'{app}(n={len(hemisphere_datasets[app])})' for app in available_approaches])}")

                # Perform all pairwise comparisons
                comparison_results = []

                for approach1, approach2 in combinations(available_approaches, 2):
                    data1 = hemisphere_datasets[approach1]
                    data2 = hemisphere_datasets[approach2]

                    # Align paired data based on Case_ID
                    values1, values2 = self.align_paired_data(data1, data2)

                    if len(values1) < 3:  # Need at least 3 pairs for Wilcoxon test
                        print(f"    {approach1} vs {approach2}: Insufficient paired data (n={len(values1)})")
                        continue

                    # Wilcoxon signed-rank test for paired data
                    try:
                        # Calculate differences
                        differences = values1 - values2
                        non_zero_diffs = differences[differences != 0]

                        if len(non_zero_diffs) < 3:
                            print(f"    {approach1} vs {approach2}: Too many tied pairs (n_non_zero={len(non_zero_diffs)})")
                            continue

                        # Perform Wilcoxon signed-rank test
                        statistic, p_value = stats.wilcoxon(values1, values2, alternative='two-sided')

                        # Calculate effect size (r = Z / sqrt(N))
                        # For Wilcoxon, we can use the rank-biserial correlation
                        n_pairs = len(values1)
                        z_score = stats.norm.ppf(1 - p_value/2) * (1 if np.median(differences) > 0 else -1)
                        effect_size = z_score / np.sqrt(n_pairs)

                        # Calculate medians and median difference
                        median1 = np.median(values1)
                        median2 = np.median(values2)
                        median_diff = np.median(differences)

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
                            'Config': config_name,
                            'Hemisphere': hemisphere,
                            'Approach1': approach1,
                            'Approach2': approach2,
                            'N_Pairs': n_pairs,
                            'Median1': median1,
                            'Median2': median2,
                            'Median_Diff': median_diff,
                            'Mean_Diff': np.mean(differences),
                            'Std_Diff': np.std(differences),
                            'Statistic': statistic,
                            'P_Value': p_value,
                            'Effect_Size': effect_size,
                            'Significance': significance
                        })

                        print(f"    {approach1} vs {approach2}:")
                        print(f"      Paired cases: {n_pairs}")
                        print(f"      Medians: {median1:.4f} vs {median2:.4f}")
                        print(f"      Median difference: {median_diff:+.4f}")
                        print(f"      Wilcoxon W: {statistic:.2f}, p-value: {p_value:.6f} {significance}")
                        print(f"      Effect size (r): {effect_size:.3f}")

                    except Exception as e:
                        print(f"    {approach1} vs {approach2}: Error - {e}")

                all_stats_results.extend(comparison_results)

        # Process and save results
        if all_stats_results:
            return self.save_statistical_results(all_stats_results)
        else:
            print("No statistical comparisons could be performed.")
            return None

    def save_statistical_results(self, all_stats_results):
        """Save statistical results to Excel with comprehensive analysis"""

        stats_df = pd.DataFrame(all_stats_results)

        # Sort by configuration, hemisphere, and p-value
        stats_df = stats_df.sort_values(['Config', 'Hemisphere', 'P_Value'])

        # Add interpretation columns
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

        # Apply Bonferroni correction within each config-hemisphere combination
        corrected_results = []
        for (config, hemisphere), group in stats_df.groupby(['Config', 'Hemisphere']):
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = self.output_dir / f"statistical_comparison_{timestamp}.xlsx"

        # Create comprehensive Excel file with multiple sheets
        with pd.ExcelWriter(stats_path, engine='openpyxl') as writer:
            # Main results sheet
            stats_df.to_excel(writer, sheet_name='All_Comparisons', index=False)

            # Significant results only (uncorrected)
            significant_df = stats_df[stats_df['P_Value'] < 0.05]
            if not significant_df.empty:
                significant_df.to_excel(writer, sheet_name='Significant_Uncorrected', index=False)

            # Significant results with Bonferroni correction
            significant_bonf_df = stats_df[stats_df['P_Value_Bonferroni'] < 0.05]
            if not significant_bonf_df.empty:
                significant_bonf_df.to_excel(writer, sheet_name='Significant_Bonferroni', index=False)

            # Summary by configuration
            summary_by_config = stats_df.groupby(['Config', 'Hemisphere']).agg({
                'P_Value': ['count', lambda x: sum(x < 0.05), lambda x: sum(x < 0.01)],
                'P_Value_Bonferroni': [lambda x: sum(x < 0.05), lambda x: sum(x < 0.01)],
                'Effect_Size': ['mean', 'std'],
                'N_Pairs': 'mean'
            }).round(4)
            summary_by_config.columns = [
                'Total_Comparisons', 'Significant_p05', 'Significant_p01',
                'Bonferroni_p05', 'Bonferroni_p01',
                'Mean_Effect_Size', 'Std_Effect_Size', 'Mean_N_Pairs'
            ]
            summary_by_config.to_excel(writer, sheet_name='Summary_by_Config')

            # Effect size summary
            effect_summary = stats_df.groupby(['Config', 'Hemisphere', 'Effect_Size_Interpretation']).size().unstack(fill_value=0)
            effect_summary.to_excel(writer, sheet_name='Effect_Size_Summary')

        print(f"\n{'='*60}")
        print(f"STATISTICAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total pairwise comparisons: {len(stats_df)}")
        print(f"Significant at p < 0.05: {sum(stats_df['P_Value'] < 0.05)} ({100*sum(stats_df['P_Value'] < 0.05)/len(stats_df):.1f}%)")
        print(f"Significant at p < 0.01: {sum(stats_df['P_Value'] < 0.01)} ({100*sum(stats_df['P_Value'] < 0.01)/len(stats_df):.1f}%)")
        print(f"Significant with Bonferroni correction (p < 0.05): {sum(stats_df['P_Value_Bonferroni'] < 0.05)} ({100*sum(stats_df['P_Value_Bonferroni'] < 0.05)/len(stats_df):.1f}%)")
        print(f"Average paired cases per comparison: {stats_df['N_Pairs'].mean():.1f}")
        print(f"Effect sizes: Large={sum(stats_df['Effect_Size_Interpretation'] == 'Large')}, " +
              f"Medium={sum(stats_df['Effect_Size_Interpretation'] == 'Medium')}, " +
              f"Small={sum(stats_df['Effect_Size_Interpretation'] == 'Small')}, " +
              f"Negligible={sum(stats_df['Effect_Size_Interpretation'] == 'Negligible')}")
        print(f"\nResults saved to: {stats_path}")

        return stats_df


def main():
    # Set up paths
    results_dir = Path(__file__).parent.parent / "evaluation_results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Create statistical comparator and run analysis
    comparator = StatisticalComparator(results_dir)
    stats_results = comparator.perform_comprehensive_testing()

    if stats_results is not None:
        print("\nStatistical comparison completed successfully!")
    else:
        print("\nStatistical comparison failed - no results generated.")


if __name__ == "__main__":
    main()