#!/usr/bin/env python3
"""
Statistical Comparison of Input Configurations using Wilcoxon Signed-Rank Test

Performs comprehensive statistical testing comparing different input configurations:
- CBF
- CBF+T1w
- CBF+FLAIR
- CBF+T1w+FLAIR

Uses Wilcoxon signed-rank test for paired comparisons (same cases, different configurations).
Segmentation approaches are kept constant while comparing input configurations.
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
import openpyxl

warnings.filterwarnings('ignore')

class InputConfigStatisticalComparator:
    """Statistical comparison of input configurations using Wilcoxon signed-rank test"""

    def __init__(self, results_dir):
        """Initialize with results directory path"""
        self.results_dir = Path(results_dir)
        self.output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/significance testing")
        self.output_dir.mkdir(exist_ok=True)

        # Define mapping for input configurations
        self.config_patterns = {
            'CBF': ['CBF_results', 'thresholdseg_results'],
            'CBF+T1w': ['CBF_T1w_results'],
            'CBF+FLAIR': ['CBF_FLAIR_results'],
            'CBF+T1w+FLAIR': ['CBF_T1w_FLAIR_results']
        }

        # Define approach patterns
        self.approach_patterns = {
            'Threshold': 'thresholdseg_results',
            'Single-Class': 'crossval_singleclass_CBF_results',
            'Single-Class HalfDPS': 'crossval_singleclass_halfdps_CBF_results',
            'Multi-Class': 'crossval_multiclass_CBF_results',
            'Multi-Label': 'crossval_multilabel_CBF_results'
        }

    def find_excel_files(self):
        """Find all Excel files and organize by approach and configuration"""
        files_dict = {}

        # Initialize structure: approach -> config -> file_path
        for approach in self.approach_patterns.keys():
            files_dict[approach] = {}
            for config in self.config_patterns.keys():
                files_dict[approach][config] = None

        # Search for Excel files
        excel_files = list(self.results_dir.rglob("*.xlsx"))

        for file_path in excel_files:
            filename = file_path.name
            print(f"Checking file: {filename}")

            # Determine approach
            approach_found = None
            if 'thresholdseg_results' in filename:
                approach_found = 'Threshold'
            elif 'crossval_singleclass_halfdps' in filename:
                approach_found = 'Single-Class HalfDPS'
            elif 'crossval_singleclass' in filename:
                approach_found = 'Single-Class'
            elif 'crossval_multiclass' in filename:
                approach_found = 'Multi-Class'
            elif 'crossval_multilabel' in filename:
                approach_found = 'Multi-Label'

            if not approach_found:
                continue

            # Determine configuration
            config_found = None
            if 'CBF_T1w_FLAIR_results' in filename:
                config_found = 'CBF+T1w+FLAIR'
            elif 'CBF_T1w_results' in filename:
                config_found = 'CBF+T1w'
            elif 'CBF_FLAIR_results' in filename:
                config_found = 'CBF+FLAIR'
            elif ('CBF_results' in filename) or ('thresholdseg_results' in filename):
                config_found = 'CBF'

            if config_found and approach_found:
                files_dict[approach_found][config_found] = file_path
                print(f"Found {approach_found} {config_found}: {filename}")

        return files_dict

    def load_hemisphere_data(self, file_path, approach_name):
        """Load hemisphere-specific data from Excel file"""
        try:
            df = pd.read_excel(file_path)

            hemisphere_data = {}

            if approach_name == 'Threshold':
                # Threshold data structure (has DSC_Left_Hemisphere, DSC_Right_Hemisphere columns)
                for hemisphere in ['Left', 'Right']:
                    col_name = f'DSC_{hemisphere}_Hemisphere'
                    if col_name in df.columns:
                        # Create Subject+Visit identifier for pairing
                        if 'Subject' in df.columns and 'Visit' in df.columns:
                            df['Subject_Visit'] = df['Subject'].astype(str) + '_v' + df['Visit'].astype(str)
                        else:
                            df['Subject_Visit'] = df.index.astype(str)

                        valid_data = df[df[col_name].notna()]
                        hemisphere_data[hemisphere] = valid_data[[col_name, 'Subject_Visit']].rename(
                            columns={col_name: 'DSC'}
                        )
                    else:
                        hemisphere_data[hemisphere] = pd.DataFrame(columns=['DSC', 'Subject_Visit'])

            else:
                # Deep learning approaches structure (has Hemisphere column and DSC_Volume)
                if 'Hemisphere' in df.columns and 'DSC_Volume' in df.columns:
                    # Create Subject+Visit identifier for pairing
                    if 'Subject' in df.columns and 'Visit' in df.columns:
                        df['Subject_Visit'] = df['Subject'].astype(str) + '_v' + df['Visit'].astype(str)
                    else:
                        df['Subject_Visit'] = df.index.astype(str)

                    for hemisphere in ['Left', 'Right']:
                        hemi_data = df[df['Hemisphere'] == hemisphere].copy()
                        if len(hemi_data) > 0:
                            valid_data = hemi_data[hemi_data['DSC_Volume'].notna()]
                            hemisphere_data[hemisphere] = valid_data[['DSC_Volume', 'Subject_Visit']].rename(
                                columns={'DSC_Volume': 'DSC'}
                            )
                        else:
                            hemisphere_data[hemisphere] = pd.DataFrame(columns=['DSC', 'Subject_Visit'])
                else:
                    # Fallback: try the threshold format for all approaches
                    for hemisphere in ['Left', 'Right']:
                        col_name = f'DSC_{hemisphere}_Hemisphere'
                        if col_name in df.columns:
                            if 'Subject' in df.columns and 'Visit' in df.columns:
                                df['Subject_Visit'] = df['Subject'].astype(str) + '_v' + df['Visit'].astype(str)
                            else:
                                df['Subject_Visit'] = df.index.astype(str)

                            valid_data = df[df[col_name].notna()]
                            hemisphere_data[hemisphere] = valid_data[[col_name, 'Subject_Visit']].rename(
                                columns={col_name: 'DSC'}
                            )
                        else:
                            hemisphere_data[hemisphere] = pd.DataFrame(columns=['DSC', 'Subject_Visit'])

            return hemisphere_data

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {'Left': pd.DataFrame(columns=['DSC', 'Subject_Visit']),
                   'Right': pd.DataFrame(columns=['DSC', 'Subject_Visit'])}

    def find_paired_cases(self, data1, data2):
        """Find cases that exist in both datasets and return paired data"""
        if data1.empty or data2.empty:
            return np.array([]), np.array([]), []

        # Find common Subject_Visit identifiers
        common_cases = set(data1['Subject_Visit']) & set(data2['Subject_Visit'])

        if not common_cases:
            return np.array([]), np.array([]), []

        # Sort for consistent ordering
        common_cases = sorted(list(common_cases))

        # Extract paired data
        paired_data1 = []
        paired_data2 = []

        for case_id in common_cases:
            val1 = data1[data1['Subject_Visit'] == case_id]['DSC'].iloc[0]
            val2 = data2[data2['Subject_Visit'] == case_id]['DSC'].iloc[0]
            paired_data1.append(val1)
            paired_data2.append(val2)

        return np.array(paired_data1), np.array(paired_data2), common_cases

    def perform_statistical_testing(self):
        """Perform comprehensive statistical testing comparing input configurations"""

        print("Comprehensive Statistical Comparison of Input Configurations")
        print("=" * 70)
        print("Using Wilcoxon signed-rank test for paired comparisons")
        print("Same cases, different input configurations - hemispheres analyzed separately")
        print("=" * 70)

        # Find all Excel files
        files_dict = self.find_excel_files()

        # Collect all statistical results
        all_stats_results = []

        # Process each approach (keeping approach constant, varying configuration)
        for approach_name, config_files in files_dict.items():
            # Check if we have files for this approach
            available_configs = [config for config, file_path in config_files.items() if file_path is not None]

            if len(available_configs) < 2:
                print(f"Insufficient configurations for {approach_name} (need ≥2 configs)")
                continue

            print(f"\n{'='*60}")
            print(f"STATISTICAL ANALYSIS: {approach_name}")
            print(f"{'='*60}")

            # Load data for all configurations in this approach
            config_data = {}
            for config_name in available_configs:
                file_path = config_files[config_name]
                if file_path:
                    hemi_data = self.load_hemisphere_data(file_path, approach_name)
                    config_data[config_name] = hemi_data

            # Perform pairwise comparisons for each hemisphere
            for hemisphere in ['Left', 'Right']:
                print(f"\n{hemisphere} Hemisphere Analysis:")
                print("-" * 30)

                # Get available configurations with data for this hemisphere
                available_configs_hemi = []
                hemisphere_datasets = {}

                for config_name in available_configs:
                    if (config_name in config_data and
                        hemisphere in config_data[config_name] and
                        len(config_data[config_name][hemisphere]) > 0):

                        available_configs_hemi.append(config_name)
                        hemisphere_datasets[config_name] = config_data[config_name][hemisphere]

                if len(available_configs_hemi) < 2:
                    print(f"  Insufficient data for comparisons (need ≥2 configurations)")
                    continue

                print(f"  Available configurations: {', '.join(available_configs_hemi)}")
                print(f"  Sample sizes: {', '.join([f'{config}(n={len(hemisphere_datasets[config])})' for config in available_configs_hemi])}")

                # Perform all pairwise comparisons
                comparison_results = []

                for config1, config2 in combinations(available_configs_hemi, 2):
                    data1_df = hemisphere_datasets[config1]
                    data2_df = hemisphere_datasets[config2]

                    # Find paired cases
                    data1_paired, data2_paired, common_cases = self.find_paired_cases(data1_df, data2_df)

                    if len(data1_paired) < 5:  # Minimum sample size for Wilcoxon test
                        print(f"    {config1} vs {config2}: Insufficient paired cases (n={len(data1_paired)})")
                        continue

                    try:
                        print(f"    Paired cases: {len(data1_paired)} (from {len(data1_df)} and {len(data2_df)} total)")

                        # Wilcoxon signed-rank test for paired data
                        statistic, p_value = stats.wilcoxon(data1_paired, data2_paired, alternative='two-sided')

                        # Calculate effect size (r = Z / sqrt(N))
                        n = len(data1_paired)
                        z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0  # Cap extreme values
                        effect_size = z_score / np.sqrt(n)

                        # Calculate medians and difference
                        median1 = np.median(data1_paired)
                        median2 = np.median(data2_paired)
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
                            'Approach': approach_name,
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
                            'N1_Total': len(data1_df),
                            'N2_Total': len(data2_df)
                        })

                        print(f"    {config1} vs {config2}:")
                        print(f"      Paired cases: {n}")
                        print(f"      Medians: {median1:.4f} vs {median2:.4f} (diff: {median_diff:+.4f})")
                        print(f"      Wilcoxon W: {statistic:.2f}, p-value: {p_value:.6f} {significance}")
                        print(f"      Effect size (r): {effect_size:.3f}")

                    except Exception as e:
                        print(f"    {config1} vs {config2}: Error - {e}")

                all_stats_results.extend(comparison_results)

        # Save statistical results to Excel
        if all_stats_results:
            stats_df = pd.DataFrame(all_stats_results)

            # Sort by approach, hemisphere, and p-value
            stats_df = stats_df.sort_values(['Approach', 'Hemisphere', 'P_Value'])

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

            # Apply Bonferroni correction within each approach-hemisphere combination
            corrected_results = []
            for (approach, hemisphere), group in stats_df.groupby(['Approach', 'Hemisphere']):
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
            stats_path = self.output_dir / f"statistical_comparison_config_{timestamp}.xlsx"

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

                # Summary by approach
                summary_by_approach = stats_df.groupby(['Approach', 'Hemisphere']).agg({
                    'P_Value': ['count', lambda x: sum(x < 0.05), lambda x: sum(x < 0.01)],
                    'P_Value_Bonferroni': [lambda x: sum(x < 0.05)],
                    'Effect_Size': ['mean', 'std'],
                    'N_Paired': 'mean'
                }).round(4)

                summary_by_approach.columns = ['Total_Comparisons', 'Significant_p05', 'Significant_p01',
                                             'Bonferroni_Significant', 'Mean_Effect_Size', 'Std_Effect_Size',
                                             'Avg_Paired_Cases']
                summary_by_approach.to_excel(writer, sheet_name='Summary_by_Approach')

                # Effect size summary
                effect_size_summary = stats_df['Effect_Size_Interpretation'].value_counts().to_frame('Count')
                effect_size_summary['Percentage'] = (effect_size_summary['Count'] / len(stats_df) * 100).round(1)
                effect_size_summary.to_excel(writer, sheet_name='Effect_Size_Summary')

            print(f"\n{'='*60}")
            print(f"STATISTICAL RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Total pairwise comparisons: {len(stats_df)}")
            print(f"Significant at p < 0.05: {sum(stats_df['P_Value'] < 0.05)} ({100*sum(stats_df['P_Value'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Significant at p < 0.01: {sum(stats_df['P_Value'] < 0.01)} ({100*sum(stats_df['P_Value'] < 0.01)/len(stats_df):.1f}%)")
            print(f"Significant with Bonferroni correction (p < 0.05): {sum(stats_df['P_Value_Bonferroni'] < 0.05)} ({100*sum(stats_df['P_Value_Bonferroni'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Average paired cases per comparison: {stats_df['N_Paired'].mean():.1f}")

            # Effect size distribution
            effect_counts = stats_df['Effect_Size_Interpretation'].value_counts()
            print(f"Effect sizes: Large={effect_counts.get('Large', 0)}, Medium={effect_counts.get('Medium', 0)}, Small={effect_counts.get('Small', 0)}, Negligible={effect_counts.get('Negligible', 0)}")

            print(f"\nResults saved to: {stats_path}")
            print("Statistical comparison completed successfully!")

            return stats_df
        else:
            print("No statistical comparisons could be performed.")
            return None

def main():
    """Main function to run statistical comparison"""
    results_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation")

    # Initialize and run statistical comparison
    comparator = InputConfigStatisticalComparator(results_dir)
    stats_results = comparator.perform_statistical_testing()

    if stats_results is not None:
        print(f"\nStatistical comparison completed with {len(stats_results)} comparisons.")
    else:
        print("\nNo statistical comparisons could be performed.")

if __name__ == "__main__":
    main()