#!/usr/bin/env python3
"""
Statistical Comparison of Left vs Right Hemisphere Performance using Wilcoxon Signed-Rank Test

Performs comprehensive statistical testing comparing left vs right hemisphere performance
for all deep learning segmentation approaches (excluding threshold segmentation).

Uses Wilcoxon signed-rank test for paired comparisons (same cases, different hemispheres).
Approaches and input configurations are analyzed separately with Bonferroni correction
for multiple comparisons.

Output: Excel file with comprehensive hemisphere comparison results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from scipy import stats
import openpyxl

warnings.filterwarnings('ignore')

class HemisphereStatisticalComparator:
    """Statistical comparison of left vs right hemisphere performance using Wilcoxon signed-rank test"""

    def __init__(self, results_dir):
        """Initialize with results directory path"""
        self.results_dir = Path(results_dir)
        self.output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/significance testing")
        self.output_dir.mkdir(exist_ok=True)

        # Define approach and configuration patterns (excluding threshold, multi-class, and single-class halfdps)
        self.approach_patterns = {
            'Single-Class': 'crossval_singleclass',
            'Multi-Label': 'crossval_multilabel'
        }

        self.config_patterns = {
            'CBF': '_CBF_results',
            'CBF+T1w': '_CBF_T1w_results',
            'CBF+FLAIR': '_CBF_FLAIR_results',
            'CBF+T1w+FLAIR': '_CBF_T1w_FLAIR_results'
        }

    def find_excel_files(self):
        """Find all Excel files and organize by approach and configuration"""
        files_dict = {}

        # Initialize structure: approach -> config -> file_path
        for approach in self.approach_patterns.keys():
            files_dict[approach] = {}
            for config in self.config_patterns.keys():
                files_dict[approach][config] = None

        # Search for Excel files (excluding threshold files)
        excel_files = list(self.results_dir.rglob("*.xlsx"))

        for file_path in excel_files:
            filename = file_path.name

            # Skip threshold, multi-class, and single-class halfdps files
            if 'thresholdseg' in filename or 'crossval_multiclass' in filename or 'crossval_singleclass_halfdps' in filename:
                continue

            print(f"Checking file: {filename}")

            # Determine approach
            approach_found = None
            if 'crossval_singleclass' in filename:
                approach_found = 'Single-Class'
            elif 'crossval_multilabel' in filename:
                approach_found = 'Multi-Label'

            if not approach_found:
                continue

            # Determine configuration
            config_found = None
            if '_CBF_T1w_FLAIR_results' in filename:
                config_found = 'CBF+T1w+FLAIR'
            elif '_CBF_T1w_results' in filename:
                config_found = 'CBF+T1w'
            elif '_CBF_FLAIR_results' in filename:
                config_found = 'CBF+FLAIR'
            elif '_CBF_results' in filename:
                config_found = 'CBF'

            if config_found and approach_found:
                files_dict[approach_found][config_found] = file_path
                print(f"Found {approach_found} {config_found}: {filename}")

        return files_dict

    def load_hemisphere_data(self, file_path):
        """Load hemisphere data from Excel file and prepare for left vs right comparison"""
        try:
            df = pd.read_excel(file_path)

            # Check for DSC_Volume column
            if 'DSC_Volume' not in df.columns:
                print(f"Missing DSC_Volume column in {file_path}")
                return pd.DataFrame()

            # Create Subject+Visit identifier for pairing
            if 'Subject' in df.columns and 'Visit' in df.columns:
                df['Subject_Visit'] = df['Subject'].astype(str) + '_v' + df['Visit'].astype(str)
            else:
                df['Subject_Visit'] = df.index.astype(str)

            # Check for Hemisphere column (standard format for Single-Class, Single-Class HalfDPS, Multi-Label)
            if 'Hemisphere' not in df.columns:
                print(f"No Hemisphere column found in {file_path}")
                return pd.DataFrame()

            # Filter to only Left/Right hemispheres
            df = df[df['Hemisphere'].isin(['Left', 'Right'])]
            if len(df) == 0:
                print(f"No Left/Right hemisphere data found in {file_path}")
                return pd.DataFrame()

            # Pivot to get left and right hemisphere data side by side
            hemisphere_pivot = df.pivot_table(
                index='Subject_Visit',
                columns='Hemisphere',
                values='DSC_Volume',
                aggfunc='first'
            ).reset_index()

            # Only keep cases that have both left and right hemisphere data
            if 'Left' in hemisphere_pivot.columns and 'Right' in hemisphere_pivot.columns:
                complete_cases = hemisphere_pivot.dropna(subset=['Left', 'Right'])
                return complete_cases
            else:
                print(f"Missing Left/Right hemisphere data in {file_path}")
                print(f"Available columns: {hemisphere_pivot.columns.tolist()}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def perform_statistical_testing(self):
        """Perform comprehensive statistical testing comparing left vs right hemispheres"""

        print("Comprehensive Statistical Comparison of Left vs Right Hemisphere Performance")
        print("=" * 80)
        print("Using Wilcoxon signed-rank test for paired comparisons")
        print("Same cases, comparing left vs right hemisphere performance")
        print("Excluding threshold, multi-class, and single-class halfdps results")
        print("=" * 80)

        # Find all Excel files
        files_dict = self.find_excel_files()

        # Collect all statistical results
        all_stats_results = []

        # Process each approach and configuration combination
        for approach_name, config_files in files_dict.items():
            for config_name, file_path in config_files.items():
                if file_path is None:
                    continue

                print(f"\n{'='*60}")
                print(f"STATISTICAL ANALYSIS: {approach_name} - {config_name}")
                print(f"{'='*60}")

                # Load hemisphere data
                hemisphere_data = self.load_hemisphere_data(file_path)

                if hemisphere_data.empty or len(hemisphere_data) < 5:
                    print(f"Insufficient data for analysis (n={len(hemisphere_data)})")
                    continue

                print(f"Paired cases available: {len(hemisphere_data)}")

                # Extract left and right hemisphere data
                left_data = hemisphere_data['Left'].values
                right_data = hemisphere_data['Right'].values

                try:
                    # Wilcoxon signed-rank test for paired data
                    statistic, p_value = stats.wilcoxon(left_data, right_data, alternative='two-sided')

                    # Calculate effect size (r = Z / sqrt(N))
                    n = len(left_data)
                    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0  # Cap extreme values
                    effect_size = z_score / np.sqrt(n)

                    # Calculate medians and difference
                    median_left = np.median(left_data)
                    median_right = np.median(right_data)
                    median_diff = median_left - median_right

                    # Calculate additional descriptive statistics
                    mean_left = np.mean(left_data)
                    mean_right = np.mean(right_data)
                    std_left = np.std(left_data)
                    std_right = np.std(right_data)

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

                    # Determine which hemisphere performs better
                    better_hemisphere = "Left" if median_left > median_right else "Right"
                    if abs(median_diff) < 0.001:
                        better_hemisphere = "Equal"

                    all_stats_results.append({
                        'Approach': approach_name,
                        'Configuration': config_name,
                        'N_Cases': n,
                        'Left_Median': median_left,
                        'Right_Median': median_right,
                        'Median_Diff_Left_vs_Right': median_diff,
                        'Left_Mean': mean_left,
                        'Right_Mean': mean_right,
                        'Left_Std': std_left,
                        'Right_Std': std_right,
                        'Statistic': statistic,
                        'P_Value': p_value,
                        'Effect_Size': effect_size,
                        'Significance': significance,
                        'Better_Hemisphere': better_hemisphere,
                        'Abs_Diff': abs(median_diff)
                    })

                    print(f"Left hemisphere:  Median={median_left:.4f}, Mean={mean_left:.4f} ± {std_left:.4f}")
                    print(f"Right hemisphere: Median={median_right:.4f}, Mean={mean_right:.4f} ± {std_right:.4f}")
                    print(f"Difference (L-R): {median_diff:+.4f}")
                    print(f"Wilcoxon W: {statistic:.2f}, p-value: {p_value:.6f} {significance}")
                    print(f"Effect size (r): {effect_size:.3f}")
                    print(f"Better hemisphere: {better_hemisphere}")

                except Exception as e:
                    print(f"Error in statistical test: {e}")

        # Save statistical results to Excel
        if all_stats_results:
            stats_df = pd.DataFrame(all_stats_results)

            # Sort by approach, configuration, and p-value
            stats_df = stats_df.sort_values(['Approach', 'Configuration', 'P_Value'])

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

            # Apply Bonferroni correction within each approach
            corrected_results = []
            for approach, group in stats_df.groupby('Approach'):
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
            stats_path = self.output_dir / f"statistical_comparison_hemis_{timestamp}.xlsx"

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
                summary_by_approach = stats_df.groupby('Approach').agg({
                    'P_Value': ['count', lambda x: sum(x < 0.05), lambda x: sum(x < 0.01)],
                    'P_Value_Bonferroni': [lambda x: sum(x < 0.05)],
                    'Effect_Size': ['mean', 'std'],
                    'Abs_Diff': 'mean',
                    'N_Cases': 'mean'
                }).round(4)

                summary_by_approach.columns = ['Total_Configs', 'Significant_p05', 'Significant_p01',
                                             'Bonferroni_Significant', 'Mean_Effect_Size', 'Std_Effect_Size',
                                             'Mean_Abs_Diff', 'Avg_Cases']
                summary_by_approach.to_excel(writer, sheet_name='Summary_by_Approach')

                # Hemisphere preference summary
                hemisphere_preference = stats_df['Better_Hemisphere'].value_counts().to_frame('Count')
                hemisphere_preference['Percentage'] = (hemisphere_preference['Count'] / len(stats_df) * 100).round(1)
                hemisphere_preference.to_excel(writer, sheet_name='Hemisphere_Preference')

                # Effect size summary
                effect_size_summary = stats_df['Effect_Size_Interpretation'].value_counts().to_frame('Count')
                effect_size_summary['Percentage'] = (effect_size_summary['Count'] / len(stats_df) * 100).round(1)
                effect_size_summary.to_excel(writer, sheet_name='Effect_Size_Summary')

                # Configuration summary
                config_summary = stats_df.groupby('Configuration').agg({
                    'P_Value': ['count', lambda x: sum(x < 0.05)],
                    'Median_Diff_Left_vs_Right': 'mean',
                    'Abs_Diff': 'mean'
                }).round(4)
                config_summary.columns = ['Total_Approaches', 'Significant_p05', 'Mean_Left_vs_Right_Diff', 'Mean_Abs_Diff']
                config_summary.to_excel(writer, sheet_name='Summary_by_Configuration')

            print(f"\n{'='*60}")
            print(f"STATISTICAL RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Total approach-configuration combinations: {len(stats_df)}")
            print(f"Significant at p < 0.05: {sum(stats_df['P_Value'] < 0.05)} ({100*sum(stats_df['P_Value'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Significant at p < 0.01: {sum(stats_df['P_Value'] < 0.01)} ({100*sum(stats_df['P_Value'] < 0.01)/len(stats_df):.1f}%)")
            print(f"Significant with Bonferroni correction (p < 0.05): {sum(stats_df['P_Value_Bonferroni'] < 0.05)} ({100*sum(stats_df['P_Value_Bonferroni'] < 0.05)/len(stats_df):.1f}%)")
            print(f"Average cases per comparison: {stats_df['N_Cases'].mean():.1f}")

            # Hemisphere preference
            hemisphere_counts = stats_df['Better_Hemisphere'].value_counts()
            print(f"Hemisphere preference: Left={hemisphere_counts.get('Left', 0)}, Right={hemisphere_counts.get('Right', 0)}, Equal={hemisphere_counts.get('Equal', 0)}")

            # Effect size distribution
            effect_counts = stats_df['Effect_Size_Interpretation'].value_counts()
            print(f"Effect sizes: Large={effect_counts.get('Large', 0)}, Medium={effect_counts.get('Medium', 0)}, Small={effect_counts.get('Small', 0)}, Negligible={effect_counts.get('Negligible', 0)}")

            # Average difference
            mean_abs_diff = stats_df['Abs_Diff'].mean()
            print(f"Average absolute difference between hemispheres: {mean_abs_diff:.4f}")

            print(f"\nResults saved to: {stats_path}")
            print("Hemisphere comparison completed successfully!")

            return stats_df
        else:
            print("No statistical comparisons could be performed.")
            return None

def main():
    """Main function to run hemisphere statistical comparison"""
    results_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/evaluation_results")

    # Initialize and run statistical comparison
    comparator = HemisphereStatisticalComparator(results_dir)
    stats_results = comparator.perform_statistical_testing()

    if stats_results is not None:
        print(f"\nHemisphere comparison completed with {len(stats_results)} comparisons.")
    else:
        print("\nNo hemisphere comparisons could be performed.")

if __name__ == "__main__":
    main()