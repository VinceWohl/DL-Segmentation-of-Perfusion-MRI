#!/usr/bin/env python3
"""
Statistical Comparison Script for Test Results
Performs paired Wilcoxon signed-rank testing with Bonferroni correction
comparing three approaches: nnUNet_CBF, nnUNet_CBF_T1w, and thresholding

Output: Excel file with 4 sheets (DSC, RVE, ASSD, HD95) showing ALL comparisons
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from itertools import combinations

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be skipped.")


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
                # Filter for this metric and show ALL comparisons (not just significant)
                metric_df = df[df['Metric'] == metric].copy()
                significant_df = metric_df[metric_df['Significance_Bonferroni'] != 'ns'].copy()

                # Report statistics but save all comparisons
                if len(significant_df) == 0:
                    print(f"  {metric}: No significant comparisons after Bonferroni correction (showing all {len(metric_df)} comparisons)")
                else:
                    print(f"  {metric}: {len(significant_df)} significant comparisons (showing all {len(metric_df)} comparisons)")

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

                # Save ALL comparisons sorted by Bonferroni-corrected p-value
                metric_df_sorted = metric_df.sort_values('P_Value_Bonferroni')
                metric_df_sorted[columns].to_excel(writer, sheet_name=metric, index=False)

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

    def create_boxplots(self, timestamp):
        """Create boxplot visualizations for each group"""
        if not PLOTTING_AVAILABLE:
            print("\nSkipping boxplot creation (matplotlib not available)")
            return

        print("\nCreating boxplot visualizations...")

        # Approach mapping for display
        approach_labels = {
            'nnUNet_CBF': 'nnUNet w/\nPerf.',
            'nnUNet_CBF_T1w': 'nnUNet w/\nPerf.\n+MP-RAGE',
            'thresholding': 'Thresholding'
        }

        approach_colors = {
            'thresholding': '#d62728',  # Red
            'nnUNet_CBF': '#1f77b4',  # Blue
            'nnUNet_CBF_T1w': '#ff7f0e'  # Orange
        }

        approach_order = ['thresholding', 'nnUNet_CBF', 'nnUNet_CBF_T1w']

        # Store statistical results for significance brackets
        self.stats_results = self.perform_comparisons()
        self.stats_results = self.apply_bonferroni_correction(self.stats_results)

        # Create plots for each group
        for group in ['HC', 'ICAS', 'AVM']:
            # Check if all approaches have data for this group
            group_has_data = all(
                approach in self.results_data and group in self.results_data[approach]
                for approach in approach_order
            )

            if not group_has_data:
                print(f"  Skipping {group}: Missing data for some approaches")
                continue

            # Create 2x2 subplot figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

            # Plot each metric - adjust DSC ylim for AVM to prevent legend overlap
            dsc_ylim = (0.7, 1.0) if group != 'AVM' else (0.72, 1.0)
            self._plot_metric_subplot(ax1, group, 'DSC', '(A) Dice Similarity Coefficient',
                                     'Dice per volume', approach_order, approach_colors,
                                     approach_labels, ylim=dsc_ylim, sig_position='above')

            self._plot_metric_subplot(ax2, group, 'RVE_Percent', '(B) Relative Volume Error',
                                     'RVE (%) per volume', approach_order, approach_colors,
                                     approach_labels, ylim=None, sig_position='below')

            self._plot_metric_subplot(ax3, group, 'ASSD_mm', '(C) Average Symmetric Surface Distance',
                                     'ASSD (mm) per slice', approach_order, approach_colors,
                                     approach_labels, ylim=None, sig_position='below')

            self._plot_metric_subplot(ax4, group, 'HD95_mm', '(D) 95th Percentile Hausdorff Distance',
                                     'HD95 (mm) per volume', approach_order, approach_colors,
                                     approach_labels, ylim=None, sig_position='below')

            # Overall title - positioned higher to avoid overlap
            fig.suptitle(f'{group} Test Set Evaluation: Segmentation Approach / Input Configuration',
                        fontsize=24, fontweight='bold', y=0.995)

            # Use tight_layout with rect to leave room for title
            plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4, w_pad=2.5)
            fig.subplots_adjust(hspace=0.33)  # Adjust vertical spacing between subplots

            # Save figure - bbox_inches='tight' will automatically include all annotations
            plot_file = self.output_dir / f"{group}_box-plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
            plt.close()

            print(f"  Saved: {plot_file.name}")

    def _add_significance_brackets(self, ax, group, metric_name, approach_order, approach_positions, position='above'):
        """Add significance brackets for significant comparisons"""
        if not hasattr(self, 'stats_results') or self.stats_results is None or self.stats_results.empty:
            return

        # Filter for this group and metric
        group_stats = self.stats_results[(self.stats_results['Group'] == group) &
                                        (self.stats_results['Metric'] == metric_name)].copy()

        # Filter for Bonferroni-significant results only
        significant_stats = group_stats[group_stats['P_Value_Bonferroni'] < 0.05].copy()

        if significant_stats.empty:
            return

        # Map approach names to positions
        approach_to_pos = {approach: approach_positions[i]
                          for i, approach in enumerate(approach_order)}

        # Get y-axis limits
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Prepare significant pairs
        significant_pairs = []
        for _, row in significant_stats.iterrows():
            approach1 = row['Approach1']
            approach2 = row['Approach2']
            p_bonf = row['P_Value_Bonferroni']

            if approach1 in approach_to_pos and approach2 in approach_to_pos:
                pos1 = approach_to_pos[approach1]
                pos2 = approach_to_pos[approach2]

                # Determine significance symbol
                if p_bonf < 0.001:
                    symbol = '***'
                elif p_bonf < 0.01:
                    symbol = '**'
                elif p_bonf < 0.05:
                    symbol = '*'
                else:
                    continue

                significant_pairs.append({
                    'pos1': min(pos1, pos2),
                    'pos2': max(pos1, pos2),
                    'symbol': symbol,
                    'span': abs(pos2 - pos1)
                })

        if not significant_pairs:
            return

        # Sort by span
        significant_pairs.sort(key=lambda x: x['span'])

        # Assign bracket heights
        bracket_heights = []
        bracket_height_increment = y_range * 0.045
        # Adjust base offset - group and position specific
        # For HC below: annotations at 0.01 (median) and -0.06 (n), so brackets start at ~0.15 for more clearance
        # For other groups below: annotations at 0.08 (median) and 0.01 (n), so brackets start at ~0.13
        # For above: annotations vary by group
        if position == 'below' and group == 'HC':
            bracket_base_offset = y_range * 0.15  # More space for HC
        elif position == 'below':
            bracket_base_offset = y_range * 0.13
        elif position == 'above' and group == 'ICAS':
            bracket_base_offset = y_range * 0.07  # ICAS DSC: user-specified value
        else:  # above for other groups
            bracket_base_offset = y_range * 0.10

        for pair in significant_pairs:
            level = 0
            while True:
                if position == 'above':
                    # For ICAS DSC, position brackets BELOW y_max to stay within 1.10 limit
                    if group == 'ICAS' and metric_name == 'DSC':
                        height = y_max - bracket_base_offset - (level * bracket_height_increment)
                    else:
                        height = y_max + bracket_base_offset + (level * bracket_height_increment)
                else:  # below
                    height = y_min - bracket_base_offset - (level * bracket_height_increment)

                # Check for overlaps
                overlaps = False
                for existing_pair, existing_height in bracket_heights:
                    if not (pair['pos2'] < existing_pair['pos1'] or pair['pos1'] > existing_pair['pos2']):
                        if abs(height - existing_height) < bracket_height_increment * 0.8:
                            overlaps = True
                            break

                if not overlaps:
                    bracket_heights.append((pair, height))
                    break

                level += 1
                if level > 15:
                    break

        # Draw brackets
        for pair, height in bracket_heights:
            pos1 = pair['pos1']
            pos2 = pair['pos2']
            symbol = pair['symbol']

            # Draw horizontal line
            ax.plot([pos1, pos2], [height, height], 'k-', linewidth=1.5)

            # Draw vertical ticks
            tick_height = y_range * 0.01
            if position == 'above':
                ax.plot([pos1, pos1], [height, height - tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height - tick_height], 'k-', linewidth=1.5)
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height - y_range * 0.012, symbol, ha='center', va='bottom',
                       fontsize=16, fontweight='bold')
            else:  # below
                ax.plot([pos1, pos1], [height, height + tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height + tick_height], 'k-', linewidth=1.5)
                mid_x = (pos1 + pos2) / 2
                ax.text(mid_x, height - y_range * 0.010, symbol, ha='center', va='top',
                       fontsize=16, fontweight='bold')

        # Adjust y-axis limits to accommodate brackets and annotations
        # This creates space OUTSIDE the data range for annotations
        if bracket_heights:
            if position == 'above':
                max_bracket_height = max(h for _, h in bracket_heights)
                new_y_max = max_bracket_height + y_range * 0.05
                # For ICAS DSC, keep the fixed y_max=1.10 to avoid extending beyond
                if group == 'ICAS' and metric_name == 'DSC':
                    ax.set_ylim(y_min, 1.10)  # Keep fixed y_max
                else:
                    ax.set_ylim(y_min, new_y_max)
            else:  # below
                min_bracket_height = min(h for _, h in bracket_heights)
                # ICAS needs extra space below - annotations are at y_min - 0.25*y_range
                if group == 'ICAS':
                    # Extend below to include annotations at y_min - 0.25*y_range
                    # Need to extend down by at least 0.35 from original y_min
                    new_y_min = min(min_bracket_height - y_range * 0.05, y_min - y_range * 0.60)
                else:
                    new_y_min = min_bracket_height - y_range * 0.05
                ax.set_ylim(new_y_min, y_max)
        else:
            # No brackets, but still need space for annotations
            if position == 'below':
                # Extend below to accommodate median + sample size boxes
                # ICAS annotations are at y_min - 0.25*y_range, need significant extension
                if group == 'ICAS':
                    new_y_min = y_min - y_range * 0.60  # Extend plot area well below for annotations
                else:
                    new_y_min = y_min - y_range * 0.25
                ax.set_ylim(new_y_min, y_max)
            else:  # above
                # Extend above to accommodate median + sample size boxes
                new_y_max = y_max + y_range * 0.15
                ax.set_ylim(y_min, new_y_max)

    def _plot_metric_subplot(self, ax, group, metric_col, title, ylabel,
                            approach_order, approach_colors, approach_labels, ylim=None, sig_position='above'):
        """Plot a single metric as a boxplot subplot"""
        # Collect data for each approach
        plot_data_list = []
        plot_positions = []
        plot_colors = []

        for i, approach in enumerate(approach_order):
            df = self.results_data[approach][group]
            if metric_col in df.columns:
                values = df[metric_col].replace([np.inf, -np.inf], np.nan).dropna().values
                plot_data_list.append(values)
            else:
                plot_data_list.append([])

            plot_positions.append(i)
            plot_colors.append(approach_colors.get(approach, '#95a5a6'))

        # Create boxplot
        bp = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=False,
            patch_artist=True,
            widths=0.7,
            medianprops=dict(color='black', linewidth=2)
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Subplot title and labels
        ax.set_title(title, fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach / Input Configuration', fontsize=18, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')

        # X-axis labels
        ax.set_xticks(plot_positions)
        ax.set_xticklabels([approach_labels.get(a, a) for a in approach_order])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Grid
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Y-axis limits - adjust for AVM DSC to prevent legend overlap
        if ylim is not None:
            ax.set_ylim(ylim)
        elif group == 'AVM' and metric_col == 'DSC':
            # Extend y-axis range for AVM DSC to prevent legend overlap with whiskers
            current_ylim = ax.get_ylim()
            ax.set_ylim(current_ylim[0] - 0.02, current_ylim[1])

        # PRE-EXTEND y-axis for ICAS plots with specific limits
        if group == 'ICAS':
            current_ylim = ax.get_ylim()
            y_max_current = current_ylim[1]
            y_min_current = current_ylim[0]

            if sig_position == 'above' and metric_col == 'DSC':
                # (A) DSC: Set y_min=0.72, y_max=1.10
                ax.set_ylim(0.72, 1.10)
            elif sig_position == 'below':
                # (B, C, D): Set specific y_min values for each metric
                if metric_col == 'RVE_Percent':
                    new_y_min = -35  # User-specified y_min for RVE
                elif metric_col == 'ASSD_mm':
                    new_y_min = -1   # User-specified y_min for ASSD
                elif metric_col == 'HD95_mm':
                    new_y_min = -10  # User-specified y_min for HD95
                else:
                    new_y_min = y_min_current - (y_max_current - y_min_current) * 0.25
                ax.set_ylim(new_y_min, y_max_current)

        # Add annotations (median, IQR, n)
        y_max = ax.get_ylim()[1]
        y_min = ax.get_ylim()[0]
        y_range = y_max - y_min

        # Individual positioning per group and metric
        for i, approach in enumerate(approach_order):
            df = self.results_data[approach][group]
            if metric_col in df.columns:
                values = df[metric_col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > 0:
                    median = values.median()
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1
                    n = len(values)

                    # Median [IQR] - 3 decimal places
                    label = f'{median:.3f} [{iqr:.3f}]'
                    color = approach_colors[approach]

                    # Match reference positioning - annotations INSIDE plot area but OUTSIDE data range
                    if sig_position == 'above':  # DSC plots - annotations above whiskers
                        # Group-specific adjustments
                        if group == 'HC':
                            # HC (A) DSC: Move median boxes higher
                            median_offset = -0.02  # set to -0.02 as requested
                            sample_offset = 0.06  # keep same
                        elif group == 'ICAS':
                            # ICAS (A) DSC: User-specified values
                            median_offset = 0.20  # user-specified value
                            sample_offset = 0.26  # user-specified value
                        else:
                            # Default for AVM and other groups
                            median_offset = 0.08
                            sample_offset = 0.17

                        ax.text(plot_positions[i], y_max - y_range * median_offset, label,
                               ha='center', va='bottom', fontsize=13,
                               color=color, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       alpha=0.8, edgecolor=color, linewidth=0.8))

                        ax.text(plot_positions[i], y_max - y_range * sample_offset, f'n={n}',
                               ha='center', va='bottom', fontsize=13, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    else:  # RVE, ASSD, HD95 - annotations below whiskers
                        # Group-specific adjustments for B, C, D
                        if group == 'HC':
                            # HC (B, C, D): Keep median position, move sample size boxes minimally higher
                            median_offset = 0.01  # keep same position
                            sample_offset = -0.07  # minimally higher (was -0.08, less negative = higher on y-axis)
                        elif group == 'ICAS':
                            # ICAS (B, C, D): Move median lower (+), sample size lower (+)
                            median_offset = 0.12  # Position lower (was 0.14) - smaller = lower on y-axis
                            sample_offset = 0.06  # Position lower (was 0.08) - smaller = lower on y-axis
                        else:
                            # Default for AVM and other groups
                            median_offset = 0.08
                            sample_offset = 0.01

                        ax.text(plot_positions[i], y_min + y_range * median_offset, label,
                               ha='center', va='top', fontsize=13,
                               color=color, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       alpha=0.8, edgecolor=color, linewidth=0.8))

                        ax.text(plot_positions[i], y_min + y_range * sample_offset, f'n={n}',
                               ha='center', va='top', fontsize=13, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Add significance brackets
        metric_name_map = {'DSC': 'DSC', 'RVE_Percent': 'RVE', 'ASSD_mm': 'ASSD', 'HD95_mm': 'HD95'}
        metric_name = metric_name_map.get(metric_col, metric_col)
        self._add_significance_brackets(ax, group, metric_name, approach_order, plot_positions, sig_position)

        # Add legend box - always bottom-right for DSC, top-right for others
        if sig_position == 'above':  # DSC
            legend_x = 0.98
            legend_y = 0.02
            legend_ha = 'right'
            legend_va = 'bottom'
        else:  # RVE, ASSD, HD95
            legend_x = 0.98
            legend_y = 0.98
            legend_ha = 'right'
            legend_va = 'top'

        ax.text(legend_x, legend_y, 'Median [IQR]\nn = sample size\n* p<0.05, ** p<0.01, *** p<0.001',
               transform=ax.transAxes, fontsize=15,
               verticalalignment=legend_va, horizontalalignment=legend_ha,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.9, edgecolor='gray', linewidth=1.5))

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

        # Create boxplots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.create_boxplots(timestamp)

        print(f"\nAnalysis complete! Results saved to: {output_file}")


def main():
    """Main execution"""
    results_dir = Path(__file__).parent  # Look for results in script directory
    output_dir = Path(__file__).parent  # Save results in script directory

    comparator = StatisticalComparator(results_dir, output_dir)
    comparator.run_analysis()


if __name__ == "__main__":
    main()
