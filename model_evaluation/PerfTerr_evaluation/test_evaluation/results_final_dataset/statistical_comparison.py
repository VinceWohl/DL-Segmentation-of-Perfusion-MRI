#!/usr/bin/env python3
"""
Statistical Comparison Script for Test Results
Performs paired Wilcoxon signed-rank testing with Bonferroni correction
comparing three approaches: nnUNet_CBF, nnUNet_CBF_T1w, and thresholding

Output: Excel file with 4 sheets (DSC, RVE, ASSD, HD95) showing significant comparisons
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from itertools import combinations


class StatisticalComparator:
    """Statistical comparison of test results across approaches"""

    def __init__(self, results_dir, output_dir):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Approaches to compare
        self.approaches = ['nnUNet_CBF', 'nnUNet_CBF_T1w', 'thresholding']

        # Metrics to analyze
        self.metrics = ['DSC', 'RVE_Percent', 'ASSD_mm', 'HD95_mm']
        self.metric_names = {'DSC': 'DSC', 'RVE_Percent': 'RVE', 'ASSD_mm': 'ASSD', 'HD95_mm': 'HD95'}

        # Load all results
        self.results_data = {}
        self.load_all_results()

    def load_all_results(self):
        """Load Per_Case results from all approaches"""
        print("Loading results from all approaches...")

        for approach in self.approaches:
            # Find most recent results file for this approach
            # Filter to exact match to avoid "nnUNet_CBF" matching "nnUNet_CBF_T1w"
            all_files = list(self.results_dir.glob("test_results_*.xlsx"))
            files = sorted([
                f for f in all_files
                if f.stem.startswith(f"test_results_{approach}_") and
                   f.stem.replace(f"test_results_{approach}_", "")[0].isdigit()
            ])

            if len(files) == 0:
                print(f"  ERROR: No results file found for {approach}")
                continue

            # Use most recent file
            results_file = files[-1]
            print(f"  Loading {approach}: {results_file.name}")

            # Load all Per_Case sheets
            approach_data = {}
            for group in ['HC', 'ICAS', 'AVM']:
                sheet_name = f"Per_Case_{group}"
                try:
                    df = pd.read_excel(results_file, sheet_name=sheet_name)
                    approach_data[group] = df
                    print(f"    {group}: {len(df)} cases")
                except Exception as e:
                    print(f"    Warning: Could not load {sheet_name}: {e}")

            self.results_data[approach] = approach_data

        print(f"\nLoaded results for {len(self.results_data)} approaches\n")

    def calculate_effect_size(self, data1, data2):
        """Calculate effect size (r = Z / sqrt(N))"""
        try:
            diff = data1 - data2
            n = len(diff)
            stat, p_value = stats.wilcoxon(diff, alternative='two-sided')
            # Calculate Z-score from p-value
            z = stats.norm.ppf(1 - p_value/2)
            r = abs(z) / np.sqrt(n)
            return r
        except:
            return np.nan

    def interpret_effect_size(self, r):
        """Interpret effect size according to Cohen's conventions"""
        if np.isnan(r):
            return "N/A"
        elif r < 0.1:
            return "Negligible"
        elif r < 0.3:
            return "Small"
        elif r < 0.5:
            return "Medium"
        else:
            return "Large"

    def perform_comparisons(self):
        """Perform pairwise comparisons for all metrics and groups"""
        print("Performing statistical comparisons...")

        all_comparisons = []

        # For each group
        for group in ['HC', 'ICAS', 'AVM']:
            print(f"\nAnalyzing {group} group...")

            # Check if all approaches have data for this group
            available_approaches = []
            for approach in self.approaches:
                if approach in self.results_data and group in self.results_data[approach]:
                    available_approaches.append(approach)

            if len(available_approaches) < 2:
                print(f"  Skipping {group}: Not enough approaches with data")
                continue

            # For each metric
            for metric in self.metrics:
                # Get data for all approaches
                approach_data = {}
                for approach in available_approaches:
                    df = self.results_data[approach][group]
                    if metric in df.columns:
                        approach_data[approach] = df.set_index('Case_ID')[metric]

                # Perform pairwise comparisons
                for approach1, approach2 in combinations(available_approaches, 2):
                    if approach1 not in approach_data or approach2 not in approach_data:
                        continue

                    # Find common cases
                    common_cases = approach_data[approach1].index.intersection(approach_data[approach2].index)

                    if len(common_cases) < 3:
                        print(f"  {metric} {approach1} vs {approach2}: Not enough paired samples (n={len(common_cases)})")
                        continue

                    # Get paired data
                    data1 = approach_data[approach1].loc[common_cases].values
                    data2 = approach_data[approach2].loc[common_cases].values

                    # Remove NaN/Inf values
                    valid_mask = np.isfinite(data1) & np.isfinite(data2)
                    data1_clean = data1[valid_mask]
                    data2_clean = data2[valid_mask]

                    if len(data1_clean) < 3:
                        print(f"  {metric} {approach1} vs {approach2}: Not enough valid samples after cleaning (n={len(data1_clean)})")
                        continue

                    # Perform Wilcoxon signed-rank test
                    try:
                        stat, p_value = stats.wilcoxon(data1_clean, data2_clean, alternative='two-sided')
                    except Exception as e:
                        print(f"  {metric} {approach1} vs {approach2}: Wilcoxon test failed: {e}")
                        continue

                    # Calculate effect size
                    effect_size = self.calculate_effect_size(data1_clean, data2_clean)
                    effect_interp = self.interpret_effect_size(effect_size)

                    # Store results
                    comparison = {
                        'Group': group,
                        'Metric': self.metric_names[metric],
                        'Approach1': approach1,
                        'Approach2': approach2,
                        'Median1': np.median(data1_clean),
                        'Median2': np.median(data2_clean),
                        'Mean1': np.mean(data1_clean),
                        'Mean2': np.mean(data2_clean),
                        'Median_Diff': np.median(data1_clean) - np.median(data2_clean),
                        'Statistic': stat,
                        'P_Value': p_value,
                        'Effect_Size': effect_size,
                        'Effect_Size_Interpretation': effect_interp,
                        'N_Paired': len(data1_clean)
                    }

                    all_comparisons.append(comparison)
                    print(f"  {metric} {approach1} vs {approach2}: p={p_value:.6f}, n={len(data1_clean)}")

        return pd.DataFrame(all_comparisons)

    def apply_bonferroni_correction(self, df):
        """Apply Bonferroni correction and mark significance"""
        # Calculate number of tests per metric
        n_tests_per_metric = {}
        for metric in df['Metric'].unique():
            n_tests_per_metric[metric] = len(df[df['Metric'] == metric])

        # Apply correction
        df['N_Tests'] = df['Metric'].map(n_tests_per_metric)
        df['P_Value_Bonferroni'] = df['P_Value'] * df['N_Tests']
        df['P_Value_Bonferroni'] = df['P_Value_Bonferroni'].clip(upper=1.0)

        # Mark significance
        def get_significance(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "ns"

        df['Significance'] = df['P_Value'].apply(get_significance)
        df['Significance_Bonferroni'] = df['P_Value_Bonferroni'].apply(get_significance)

        return df

    def save_results(self, df):
        """Save results to Excel with one sheet per metric"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"statistical_comparison_{timestamp}.xlsx"

        print(f"\nSaving results to: {output_file}")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for metric in ['DSC', 'RVE', 'ASSD', 'HD95']:
                # Filter for this metric and only significant comparisons after Bonferroni
                metric_df = df[df['Metric'] == metric].copy()
                significant_df = metric_df[metric_df['Significance_Bonferroni'] != 'ns'].copy()

                if len(significant_df) == 0:
                    print(f"  {metric}: No significant comparisons after Bonferroni correction")
                    # Still save all comparisons for reference
                    metric_df_sorted = metric_df.sort_values('P_Value_Bonferroni')
                else:
                    print(f"  {metric}: {len(significant_df)} significant comparisons (from {len(metric_df)} total)")
                    significant_df = significant_df.sort_values('P_Value_Bonferroni')

                # Select columns to display
                columns = [
                    'Group', 'Approach1', 'Approach2',
                    'Median1', 'Median2', 'Median_Diff',
                    'Mean1', 'Mean2',
                    'Statistic', 'P_Value', 'P_Value_Bonferroni',
                    'Significance', 'Significance_Bonferroni',
                    'Effect_Size', 'Effect_Size_Interpretation',
                    'N_Paired', 'N_Tests'
                ]

                # Save significant comparisons (or all if none significant)
                if len(significant_df) > 0:
                    significant_df[columns].to_excel(writer, sheet_name=metric, index=False)
                else:
                    metric_df[columns].sort_values('P_Value_Bonferroni').to_excel(writer, sheet_name=metric, index=False)

        print(f"\nResults saved successfully!")
        return output_file

    def print_summary(self, df):
        """Print summary of significant findings"""
        print("\n" + "="*80)
        print("SUMMARY OF SIGNIFICANT FINDINGS (After Bonferroni Correction)")
        print("="*80)

        for metric in ['DSC', 'RVE', 'ASSD', 'HD95']:
            metric_df = df[df['Metric'] == metric]
            significant = metric_df[metric_df['Significance_Bonferroni'] != 'ns']

            print(f"\n{metric}:")
            if len(significant) == 0:
                print(f"  No significant differences found")
            else:
                for _, row in significant.iterrows():
                    direction = ">" if row['Median1'] > row['Median2'] else "<"
                    print(f"  [{row['Group']}] {row['Approach1']} {direction} {row['Approach2']}: "
                          f"p={row['P_Value_Bonferroni']:.4f} {row['Significance_Bonferroni']}, "
                          f"effect={row['Effect_Size']:.3f} ({row['Effect_Size_Interpretation']})")

        print("\n" + "="*80)

    def run_analysis(self):
        """Run complete statistical analysis"""
        print("\n" + "="*80)
        print("Statistical Comparison Analysis")
        print("="*80)

        # Perform comparisons
        results_df = self.perform_comparisons()

        if len(results_df) == 0:
            print("\nERROR: No comparisons could be performed!")
            return

        # Apply Bonferroni correction
        results_df = self.apply_bonferroni_correction(results_df)

        # Save results
        output_file = self.save_results(results_df)

        # Print summary
        self.print_summary(results_df)

        print(f"\nAnalysis complete! Results saved to: {output_file}")


def main():
    """Main execution"""
    results_dir = Path(__file__).parent  # Look for results in script directory
    output_dir = Path(__file__).parent  # Save results in script directory

    comparator = StatisticalComparator(results_dir, output_dir)
    comparator.run_analysis()


if __name__ == "__main__":
    main()
