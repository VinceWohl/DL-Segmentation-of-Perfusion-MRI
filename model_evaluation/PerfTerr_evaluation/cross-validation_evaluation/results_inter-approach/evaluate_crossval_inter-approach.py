#!/usr/bin/env python3
"""
Cross-Validation Inter-Approach Comparison (From Pre-computed Results)
Loads pre-computed metrics from Excel files and creates box plots with statistical comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
from itertools import combinations

class CrossValInterApproachEvaluator:
    def __init__(self, results_dir, output_dir):
        """Initialize with directories for pre-computed results and output"""
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Excel files mapping
        self.excel_files = {
            'Thresholding': 'thresholdseg_results_20250905_154303.xlsx',
            'Single-class': 'crossval_singleclass_CBF_results_20250908_143400.xlsx',
            'Single-class halfdps': 'crossval_singleclass_halfdps_CBF_results_20250908_153350.xlsx',
            'Multi-class': 'crossval_multiclass_CBF_results_20250909_160000.xlsx',
            'Multi-label': 'crossval_multilabel_CBF_results_20250911_060204.xlsx'
        }

        # Approach colors
        self.approach_colors = {
            'Thresholding': '#e74c3c',
            'Single-class': '#3498db',
            'Single-class halfdps': '#2ecc71',
            'Multi-class': '#f39c12',
            'Multi-label': '#9b59b6'
        }

        self.data = {}
        self.statistical_results = {}

    def load_data(self):
        """Load all pre-computed results from Excel files"""
        print("\nLoading Pre-computed Results...")
        print("="*80)

        for approach, filename in self.excel_files.items():
            filepath = self.results_dir / filename
            if not filepath.exists():
                print(f"  WARNING: {filename} not found")
                continue

            # Load correct sheet based on approach
            if approach == 'Thresholding':
                sheet_name = 'Per_Case_Results'
            elif approach == 'Multi-class':
                # For Multi-class, use Per_Hemisphere_Details sheet
                sheet_name = 'Per_Hemisphere_Details'
            else:
                sheet_name = 'Per_Case_Details'

            df = pd.read_excel(filepath, sheet_name=sheet_name)

            # Filter out severe outliers from Thresholding (DSC < 0.5)
            if approach == 'Thresholding' and 'DSC_Volume' in df.columns:
                n_before = len(df)
                df = df[df['DSC_Volume'] >= 0.5].copy()
                n_after = len(df)
                if n_before > n_after:
                    print(f"  Filtered {n_before - n_after} severe outlier(s) (DSC < 0.5) from {approach}")

            # Store data (combining left and right hemispheres for analysis)
            self.data[approach] = df
            print(f"  Loaded {approach}: {len(df)} rows from sheet '{sheet_name}'")

        print(f"\n  Total approaches loaded: {len(self.data)}")

    def perform_statistical_testing(self):
        """Perform pairwise Wilcoxon tests with Bonferroni correction"""
        print("\nPerforming Statistical Testing...")
        print("="*80)

        metrics = ['DSC_Volume', 'RVE_Percent', 'ASSD_mm', 'HD95_mm']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for metric in metrics:
            print(f"\n  Metric: {metric}")
            print("  " + "-"*40)

            metric_results = []
            approaches = list(self.data.keys())

            # Pairwise comparisons
            for approach1, approach2 in combinations(approaches, 2):
                try:
                    df1 = self.data[approach1]
                    df2 = self.data[approach2]

                    # Get metric values
                    values1 = df1[metric].dropna()
                    values2 = df2[metric].dropna()

                    if len(values1) == 0 or len(values2) == 0:
                        continue

                    # Mann-Whitney U test (unpaired)
                    statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')

                    # Calculate effect size
                    n1, n2 = len(values1), len(values2)
                    z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 5.0
                    effect_size = z_score / np.sqrt(n1 + n2)

                    # Medians
                    median1 = np.median(values1)
                    median2 = np.median(values2)
                    median_diff = median1 - median2

                    # Uncorrected significance
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"

                    metric_results.append({
                        'Hemisphere': 'Combined',  # Match reference format
                        'Config1': approach1,
                        'Config2': approach2,
                        'Median1': median1,
                        'Median2': median2,
                        'Median_Diff': median_diff,
                        'Statistic': statistic,
                        'P_Value': p_value,
                        'Effect_Size': effect_size,
                        'Significance': significance,
                        'N_Paired_Slices': min(n1, n2),  # For unpaired, use min as reference
                        'N1_Total_Slices': n1,
                        'N2_Total_Slices': n2
                    })

                    print(f"    {approach1} vs {approach2}: p={p_value:.4f} {significance} (n1={n1}, n2={n2})")

                except Exception as e:
                    print(f"    Error comparing {approach1} vs {approach2}: {e}")

            # Save results with Bonferroni correction
            if metric_results:
                stats_df = self._save_statistical_results(metric_results, metric, timestamp)
                self.statistical_results[metric] = stats_df
            else:
                print(f"    No statistical comparisons performed for {metric}")

    def _save_statistical_results(self, results, metric_name, timestamp):
        """Save statistical results to Excel with Bonferroni correction"""
        stats_df = pd.DataFrame(results)

        # Apply Bonferroni correction
        n_comparisons = len(stats_df)
        stats_df['P_Value_Bonferroni'] = stats_df['P_Value'] * n_comparisons
        stats_df['P_Value_Bonferroni'] = stats_df['P_Value_Bonferroni'].clip(upper=1.0)

        # Bonferroni-corrected significance
        def get_significance_bonf(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "ns"

        stats_df['Significance_Bonferroni'] = stats_df['P_Value_Bonferroni'].apply(get_significance_bonf)

        # Sort by Bonferroni-corrected p-value
        stats_df = stats_df.sort_values('P_Value_Bonferroni')

        # Add effect size interpretation
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

        # Reorder columns to match reference format exactly
        column_order = [
            'Hemisphere', 'Config1', 'Config2', 'Median1', 'Median2', 'Median_Diff',
            'Statistic', 'P_Value', 'Effect_Size', 'Significance',
            'N_Paired_Slices', 'N1_Total_Slices', 'N2_Total_Slices',
            'Effect_Size_Interpretation', 'P_Value_Bonferroni', 'Significance_Bonferroni'
        ]
        stats_df = stats_df[column_order]

        # Save to Excel
        metric_short = metric_name.split('_')[0]
        output_file = self.output_dir / f"crossval_statistical_comparison_{metric_short}_{timestamp}.xlsx"

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            stats_df.to_excel(writer, sheet_name='Statistical_Tests', index=False)

        print(f"    Saved: {output_file.name}")
        return stats_df

    def create_combined_box_plot(self):
        """Create 4-subplot box plot with all metrics"""
        print("\nCreating Combined Box Plot...")
        print("="*80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Cross-Validation Evaluation: Segmentation Approaches (solely Perfusion Maps as Input)',
                     fontsize=24, fontweight='bold', y=0.995)

        # Define metrics
        metrics = [
            ('DSC_Volume', 'Dice Similarity Coefficient', 'Dice per volume', '(A)', None, 'above'),
            ('RVE_Percent', 'Relative Volume Error', 'RVE (%) per volume', '(B)', None, 'below'),
            ('ASSD_mm', 'Average Symmetric Surface Distance', 'ASSD (mm) per slice', '(C)', None, 'below'),
            ('HD95_mm', '95th Percentile Hausdorff Distance', 'HD95 (mm) per volume', '(D)', None, 'below')
        ]

        approach_order = list(self.data.keys())

        for idx, (metric, title, ylabel, label, ylim, annotation_position) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            self._plot_metric_subplot(ax, metric, title, ylabel, label, ylim,
                                     approach_order, annotation_position)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=5, w_pad=3)
        fig.subplots_adjust(hspace=0.35, wspace=0.20)

        # Save
        plot_file = self.output_dir / f"crossval_combined_boxplot_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  Saved: {plot_file.name}")

    def _plot_metric_subplot(self, ax, metric, title, ylabel, label, ylim,
                            approach_order, annotation_position):
        """Plot a single metric subplot"""

        # Collect data for each approach
        plot_data = []
        plot_labels = []
        plot_colors = []

        for approach in approach_order:
            if approach in self.data:
                values = self.data[approach][metric].dropna()
                plot_data.append(values)
                plot_labels.append(approach)
                plot_colors.append(self.approach_colors[approach])

        if not plot_data:
            return

        # Create boxplot
        positions = list(range(1, len(plot_data) + 1))
        bp = ax.boxplot(plot_data, positions=positions, notch=False, patch_artist=True,
                       widths=0.7, medianprops=dict(color='black', linewidth=2),
                       showfliers=True,
                       flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

        # Color boxes
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Title and labels
        ax.set_title(f'{label} {title}', fontsize=22, fontweight='bold', pad=15, loc='left')
        ax.set_xlabel('Segmentation Approach', fontsize=18, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_labels, rotation=15, ha='right')
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Store initial y-limits
        y_max_initial = ax.get_ylim()[1]
        y_min_initial = ax.get_ylim()[0]
        y_range_initial = y_max_initial - y_min_initial

        # Extend y-axis for annotations
        if annotation_position == 'above':
            new_y_max = y_max_initial + y_range_initial * 0.15
            ax.set_ylim(y_min_initial, new_y_max)
        else:
            new_y_min = y_min_initial - y_range_initial * 0.25
            ax.set_ylim(new_y_min, y_max_initial)

        # Add annotations
        for i, (approach, values) in enumerate(zip(approach_order, plot_data)):
            if len(values) > 0:
                median = values.median()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                n = len(values)

                if metric == 'DSC_Volume' :
                    label_text = f'{median:.3f} [{iqr:.3f}]'
                else:
                    label_text = f'{median:.1f} [{iqr:.1f}]'

                color = self.approach_colors[approach]

                # Multiply sample size by 14 only for ASSD (slice-wise analysis)
                n_display = n * 14 if metric == 'ASSD_mm' else n

                if annotation_position == 'above':
                    # Sample size lower position (moved very close to boxplot top)
                    ax.text(positions[i], y_max_initial - y_range_initial * 0.005, f'n={n_display}',
                           ha='center', va='bottom', fontsize=13, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    # Median higher position (moved up further)
                    ax.text(positions[i], y_max_initial + y_range_initial * 0.08, label_text,
                           ha='center', va='bottom', fontsize=13,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.8, edgecolor=color, linewidth=0.8))
                else:
                    # Median boxes below
                    ax.text(positions[i], y_min_initial - y_range_initial * 0.02, label_text,
                           ha='center', va='top', fontsize=13,
                           color=color, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor=color, linewidth=0.8))
                    # Sample size below median
                    ax.text(positions[i], y_min_initial - y_range_initial * 0.12, f'n={n_display}',
                           ha='center', va='top', fontsize=13, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor='black', linewidth=0.8))

        # Add significance brackets
        if metric in self.statistical_results:
            box_positions_dict = {approach: positions[i] for i, approach in enumerate(approach_order)}
            self._add_significance_brackets(ax, self.statistical_results[metric],
                                          box_positions_dict, approach_order, annotation_position)

        # Add legend
        if annotation_position == 'above':
            ax.text(0.98, 0.02, 'Median [IQR]\nn = sample size\n* p<0.05, ** p<0.01, *** p<0.001',
                   transform=ax.transAxes, fontsize=15,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            alpha=0.9, edgecolor='gray', linewidth=1.5))
        else:
            ax.text(0.98, 0.98, 'Median [IQR]\nn = sample size\n* p<0.05, ** p<0.01, *** p<0.001',
                   transform=ax.transAxes, fontsize=15,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            alpha=0.9, edgecolor='gray', linewidth=1.5))

    def _add_significance_brackets(self, ax, stats_df, box_positions_dict, approach_order,
                                  annotation_position):
        """Add significance brackets using Bonferroni-corrected p-values"""
        sig_results = stats_df[stats_df['Significance_Bonferroni'].isin(['*', '**', '***'])].copy()

        if len(sig_results) == 0:
            return

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Prepare significant pairs
        significant_pairs = []
        for _, row in sig_results.iterrows():
            config1 = row['Config1']
            config2 = row['Config2']

            if config1 in box_positions_dict and config2 in box_positions_dict:
                pos1 = box_positions_dict[config1]
                pos2 = box_positions_dict[config2]

                significant_pairs.append({
                    'pos1': min(pos1, pos2),
                    'pos2': max(pos1, pos2),
                    'symbol': row['Significance_Bonferroni'],
                    'span': abs(pos2 - pos1),
                    'p_value': row['P_Value_Bonferroni']
                })

        if not significant_pairs:
            return

        # Sort by span
        significant_pairs.sort(key=lambda x: x['span'])

        # Assign bracket heights
        bracket_heights = []
        bracket_height_increment = y_range * 0.045
        bracket_base_offset = y_range * 0.008

        for pair in significant_pairs:
            level = 0
            while True:
                if annotation_position == 'above':
                    height = y_max + bracket_base_offset + (level * bracket_height_increment)
                else:
                    height = y_min - bracket_base_offset - (level * bracket_height_increment)

                # Check overlap
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

            # Horizontal line
            ax.plot([pos1, pos2], [height, height], 'k-', linewidth=1.5)

            # Vertical ticks
            tick_height = y_range * 0.01
            if annotation_position == 'above':
                ax.plot([pos1, pos1], [height, height - tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height - tick_height], 'k-', linewidth=1.5)
            else:
                ax.plot([pos1, pos1], [height, height + tick_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [height, height + tick_height], 'k-', linewidth=1.5)

            # Asterisk (positioned exactly at bracket line)
            mid_x = (pos1 + pos2) / 2
            if annotation_position == 'above':
                # Position asterisk center exactly at bracket line
                ax.text(mid_x, height + tick_height * 0.7, symbol, ha='center', va='center',
                       fontsize=16, fontweight='bold')
            else:
                ax.text(mid_x, height - y_range * 0.005, symbol, ha='center', va='top',
                       fontsize=16, fontweight='bold')

        # Adjust y-axis
        if bracket_heights:
            if annotation_position == 'above':
                max_bracket_height = max(h for _, h in bracket_heights)
                new_y_max = max_bracket_height + y_range * 0.050
                ax.set_ylim(y_min, new_y_max)
            else:
                min_bracket_height = min(h for _, h in bracket_heights)
                new_y_min = min_bracket_height - y_range * 0.050
                ax.set_ylim(new_y_min, y_max)

def main():
    """Main execution"""
    print("="*80)
    print("Cross-Validation Inter-Approach Comparison")
    print("(Loading from Pre-computed Results)")
    print("="*80)

    # Setup paths
    results_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/cross-validation_evaluation/crossval_evaluation_results")
    output_dir = Path("/home/ubuntu/DLSegPerf/model_evaluation/cross-validation_evaluation/results_inter-approach")

    # Create evaluator
    evaluator = CrossValInterApproachEvaluator(results_dir, output_dir)

    # Load data
    evaluator.load_data()

    # Statistical testing
    evaluator.perform_statistical_testing()

    # Create plots
    evaluator.create_combined_box_plot()

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
