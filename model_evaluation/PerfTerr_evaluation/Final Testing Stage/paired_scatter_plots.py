#!/usr/bin/env python3
"""
Paired Scatter Plots for Ipsilateral vs Contralateral Hemispheres
Creates scatter plots comparing ipsilateral and contralateral hemisphere metrics
for ICAS and AVM patient groups across different segmentation approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime

# Set Times New Roman (or Liberation Serif as compatible alternative) as default font
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Define paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path('/home/ubuntu/DLSegPerf/data/test_results')
GROUP_FILE = DATA_DIR / 'Group_distripution.xlsx'

# Segmentation approach (using only nnUNet_CBF)
APPROACH = 'nnUNet_CBF'

# Metrics to plot (metric_col, y_label, subplot_title)
METRICS = [
    ('DSC', 'Dice per volume', '(A) Dice Similarity Coefficient'),
    ('RVE_Percent', 'RVE (%) per volume', '(B) Relative Volume Error'),
    ('ASSD_mm', 'ASSD (mm) per slice', '(C) Average Symmetric Surface Distance'),
    ('HD95_mm', 'HD95 (mm) per volume', '(D) 95th Percentile Hausdorff Distance')
]

# Color for the approach
COLOR = '#1f77b4'  # Blue
MEAN_COLOR = '#d62728'  # Red (matching thresholding box plots)

def load_data():
    """Load test results and group distribution data."""
    # Load group distribution (hemisphere pairing information)
    group_df = pd.read_excel(GROUP_FILE)

    # Load test results for nnUNet_CBF - load ICAS and AVM sheets
    result_file = SCRIPT_DIR / 'test_results_nnUNet_CBF_20251217_084409.xlsx'

    if not result_file.exists():
        raise FileNotFoundError(f"Could not find results file: {result_file}")

    # Load ICAS and AVM sheets separately
    icas_df = pd.read_excel(result_file, sheet_name='Per_Case_ICAS')
    avm_df = pd.read_excel(result_file, sheet_name='Per_Case_AVM')

    print(f"Loaded ICAS: {len(icas_df)} cases, AVM: {len(avm_df)} cases from {result_file.name}")

    results = {'ICAS': icas_df, 'AVM': avm_df}

    return group_df, results

def create_paired_data(group_df, results_df, target_group):
    """
    Create paired data structure for ipsilateral vs contralateral comparisons.

    Args:
        group_df: DataFrame with Case_ID, Group, Side columns
        results_df: DataFrame with test results (Case_ID, Hemisphere, DSC, RVE_Percent, ASSD_mm, HD95_mm)
        target_group: 'ICAS' or 'AVM'

    Returns:
        Dictionary with paired data for each metric
    """
    # Filter for target group
    group_cases = group_df[group_df['Group'] == target_group].copy()

    # Extract subject and visit from Case_ID (e.g., "ICASII-PerfTerr039-v1-L" -> "PerfTerr039-v1")
    group_cases['Subject_Visit'] = group_cases['Case_ID'].str.extract(r'(PerfTerr\d+-v\d+)')[0]

    # Find pairs (subjects that have both Ipsi and Contra sides)
    pairs = []
    for subject_visit in group_cases['Subject_Visit'].unique():
        subj_data = group_cases[group_cases['Subject_Visit'] == subject_visit]

        # Check if both Ipsi and Contra exist (handle typo "Conrta" as well)
        has_ipsi = (subj_data['Side'] == 'Ipsi').any()
        has_contra = (subj_data['Side'].isin(['Contra', 'Conrta'])).any()

        if has_ipsi and has_contra:
            ipsi_case = subj_data[subj_data['Side'] == 'Ipsi']['Case_ID'].values[0]
            contra_case = subj_data[subj_data['Side'].isin(['Contra', 'Conrta'])]['Case_ID'].values[0]
            pairs.append({'Subject_Visit': subject_visit, 'Ipsi': ipsi_case, 'Contra': contra_case})

    pairs_df = pd.DataFrame(pairs)
    print(f"\n{target_group}: Found {len(pairs_df)} paired subjects")
    print(pairs_df)

    # Create paired data for each metric
    paired_data = {metric[0]: [] for metric in METRICS}

    for _, pair in pairs_df.iterrows():
        # Get ipsilateral data
        ipsi_data = results_df[results_df['Case_ID'] == pair['Ipsi']]
        # Get contralateral data
        contra_data = results_df[results_df['Case_ID'] == pair['Contra']]

        if len(ipsi_data) == 0 or len(contra_data) == 0:
            print(f"Warning: Missing data for pair {pair['Subject_Visit']}")
            print(f"  Ipsi: {pair['Ipsi']} - found {len(ipsi_data)} rows")
            print(f"  Contra: {pair['Contra']} - found {len(contra_data)} rows")
            continue

        # Extract metrics
        for metric_col, _, _ in METRICS:
            ipsi_val = ipsi_data[metric_col].values[0]
            contra_val = contra_data[metric_col].values[0]
            paired_data[metric_col].append({
                'Subject_Visit': pair['Subject_Visit'],
                'Ipsi': ipsi_val,
                'Contra': contra_val
            })

    return paired_data

def plot_paired_scatter(paired_data, group_name, output_file):
    """
    Create a figure with 4 subplots (one per metric) showing paired scatter plots.
    Each subplot has ipsilateral points on the left (x=0) and contralateral points on the right (x=1),
    with lines connecting each pair.

    Args:
        paired_data: Dictionary with paired data for each metric
        group_name: 'ICAS' or 'AVM'
        output_file: Path to save the figure
    """
    # Create figure with reduced height
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    n_pairs = 0  # Track number of pairs for title

    for idx, (metric_col, y_label, subplot_title) in enumerate(METRICS):
        ax = axes[idx]

        # Get data for this metric
        data = paired_data.get(metric_col, [])
        if not data:
            print(f"Warning: No data for metric {metric_col}")
            continue

        df = pd.DataFrame(data)
        n_pairs = len(df)

        print(f"  {subplot_title}: {n_pairs} pairs")

        # Plot connecting lines between paired points
        for i, row in df.iterrows():
            # Horizontal line connecting ipsi (x=0) to contra (x=1)
            ax.plot([0, 1], [row['Ipsi'], row['Contra']],
                   color=COLOR, alpha=0.5, linewidth=1.2, zorder=1)

        # Plot individual points
        # All ipsilateral points at x=0
        ax.scatter([0] * len(df), df['Ipsi'], color=COLOR,
                  s=100, alpha=0.8,
                  marker='o', edgecolors='black', linewidths=0.8, zorder=2)
        # All contralateral points at x=1
        ax.scatter([1] * len(df), df['Contra'], color=COLOR,
                  s=100, alpha=0.8,
                  marker='s', edgecolors='black', linewidths=0.8, zorder=2)

        # Plot mean lines for ipsilateral and contralateral
        mean_ipsi = df['Ipsi'].mean()
        mean_contra = df['Contra'].mean()

        # Red dotted line at ipsilateral mean (centered at x=0, short horizontal line)
        ax.plot([-0.15, 0.15], [mean_ipsi, mean_ipsi],
               color=MEAN_COLOR, linestyle=':', linewidth=2, zorder=3)
        # Red dotted line at contralateral mean (centered at x=1, short horizontal line)
        ax.plot([0.85, 1.15], [mean_contra, mean_contra],
               color=MEAN_COLOR, linestyle=':', linewidth=2, zorder=3)

        # Calculate y-axis limits with padding for mean boxes at top
        y_min = min(df['Ipsi'].min(), df['Contra'].min())
        y_max = max(df['Ipsi'].max(), df['Contra'].max())
        y_range = y_max - y_min
        y_padding = y_range * 0.20  # Add 20% padding at top for boxes
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_padding)

        # Position both mean boxes at same height (near top of plot, with margin from border)
        box_y_position = y_max + y_range * 0.08

        # Red outlined box with mean value for ipsilateral (at x=0)
        ax.text(0, box_y_position, f'{mean_ipsi:.3f}',
               ha='center', va='bottom', fontsize=14, fontweight='bold',
               color=MEAN_COLOR,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=MEAN_COLOR, linewidth=1.5),
               zorder=4)

        # Red outlined box with mean value for contralateral (at x=1)
        ax.text(1, box_y_position, f'{mean_contra:.3f}',
               ha='center', va='bottom', fontsize=14, fontweight='bold',
               color=MEAN_COLOR,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=MEAN_COLOR, linewidth=1.5),
               zorder=4)

        # Add legend in bottom right corner with reduced width
        legend_box = Patch(facecolor='white', edgecolor=MEAN_COLOR, linewidth=1.5, label='Mean')
        leg = ax.legend(handles=[legend_box], loc='lower right', fontsize=12, framealpha=0.9,
                       handlelength=1.0, handletextpad=0.5)

        # Customize subplot - match box-plot styling
        ax.set_xlabel('Hemisphere Pairs', fontsize=18, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=18, fontweight='bold')
        ax.set_title(subplot_title, fontsize=18, fontweight='bold', pad=10, loc='left')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Set x-axis with only two positions
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Ipsi', 'Contra'], fontsize=18)

        # Tick label sizes for y-axis scale
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Overall title - customize based on group
    if group_name == 'ICAS':
        title = f'ICAS Test Set Evaluation: nnUNet with Perf. - Hemisphere Comparison (n={n_pairs})'
    else:  # AVM
        title = f'AVM Test Set Evaluation: nnUNet with Perf. - Hemisphere Comparison (n={n_pairs})'

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)

    # Adjust spacing - increase vertical spacing between subplots/title, decrease horizontal spacing
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.30, wspace=0.18, top=0.91)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

def main():
    """Main execution function."""
    print("="*80)
    print("Paired Scatter Plots - Ipsilateral vs Contralateral Hemisphere Comparison")
    print(f"Using: {APPROACH}")
    print("="*80)

    # Load data
    print("\nLoading data...")
    group_df, results_dict = load_data()

    # Process each group
    for group in ['ICAS', 'AVM']:
        print(f"\n{'='*80}")
        print(f"Processing {group} group...")
        print(f"{'='*80}")

        # Get the results dataframe for this group
        results_df = results_dict[group]

        # Create paired data
        paired_data = create_paired_data(group_df, results_df, group)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = SCRIPT_DIR / f'{group}_scatter-plots_{timestamp}.png'

        # Create plots
        print(f"\nCreating plots for {group}...")
        plot_paired_scatter(paired_data, group, output_file)

    print("\n" + "="*80)
    print("Paired scatter plots generation complete!")
    print("="*80)

if __name__ == '__main__':
    main()
