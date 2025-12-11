#!/usr/bin/env python3
"""
Inter-Approach Segmentation Comparison Box Plots for Cross-Validation Results

Generates box plots comparing different segmentation approaches:
- threshold segmentation
- single-class
- single-class halfdps
- multi-class
- multi-label

For each input configuration, creates plots showing DSC_Volume performance by:
- Segmentation Approach (5 boxes side by side)
- Hemisphere (Left/Right)
- Separate plot files for each input configuration (CBF, CBF+T1w, CBF+FLAIR, CBF+T1w+FLAIR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class InterApproachPlotter:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(__file__).parent

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

        # Color palette for segmentation approaches
        self.approach_colors = {
            'Threshold': '#1f77b4',        # Blue
            'Single-Class': '#ff7f0e',     # Orange
            'Single-Class HalfDPS': '#2ca02c',  # Green
            'Multi-Class': '#d62728',      # Red
            'Multi-Label': '#9467bd'       # Purple
        }

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

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

                # Extract data for each hemisphere
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi]['DSC_Volume'].dropna()
                    data[hemi] = hemi_data.tolist()

                return data

            elif approach == 'Multi-Class':
                # Load Per_Hemisphere_Details sheet
                df = pd.read_excel(file_path, sheet_name='Per_Hemisphere_Details')

                # Extract data for each hemisphere
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi]['DSC_Volume'].dropna()
                    data[hemi] = hemi_data.tolist()

                return data

            elif approach == 'Threshold':
                # Load Per_Case_Results sheet (similar to other approaches)
                df = pd.read_excel(file_path, sheet_name='Per_Case_Results')

                # Extract data for each hemisphere
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi]['DSC_Volume'].dropna()
                    data[hemi] = hemi_data.tolist()

                return data

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {'Left': [], 'Right': []}

    def create_input_config_plot(self, config_name, files_dict, save_path):
        """Create box plot for a specific input configuration"""

        # Collect all data for this input configuration
        plot_data = []
        missing_approaches = []

        approach_order = ['Threshold', 'Single-Class', 'Single-Class HalfDPS', 'Multi-Class', 'Multi-Label']

        for approach_name in approach_order:
            if approach_name in files_dict and files_dict[approach_name]:
                file_path = files_dict[approach_name]
                hemi_data = self.load_hemisphere_data(file_path, approach_name)

                # Add data for each hemisphere
                for hemi in ['Left', 'Right']:
                    for value in hemi_data[hemi]:
                        plot_data.append({
                            'Approach': approach_name,
                            'Hemisphere': hemi,
                            'DSC_Volume': value
                        })
            else:
                missing_approaches.append(approach_name)

        if not plot_data:
            print(f"No data found for {config_name}")
            return

        df = pd.DataFrame(plot_data)

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 8))

        # Create grouped data structure: hemisphere first, then approach
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

            for approach_idx, approach in enumerate(approach_order):
                approach_data = df[(df['Hemisphere'] == hemisphere) & (df['Approach'] == approach)]

                if not approach_data.empty:
                    plot_data_list.append(approach_data['DSC_Volume'].values)
                else:
                    # Add empty data for missing approaches
                    plot_data_list.append([])

                plot_colors.append(self.approach_colors[approach])
                plot_labels.append(f'{hemisphere}_{approach}')
                plot_positions.append(position)
                hemisphere_positions[hemisphere].append(position)
                position += 0.8  # Spacing within hemisphere

            # Calculate hemisphere center for labeling
            hemisphere_centers[hemisphere] = (start_pos + position - 0.8) / 2

            # Add gap between hemispheres
            if hemi_idx < len(hemispheres) - 1:
                position += 0.5  # Gap between hemispheres

        # Create box plots manually with proper positioning
        box_parts = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=True,
            patch_artist=True,
            widths=0.6  # Box width
        )

        # Color the boxes according to segmentation approach
        for patch, color in zip(box_parts['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Customize the plot
        config_display = config_name.replace('+', '+')  # Keep the + signs
        ax.set_title(f'{config_display} Input Configuration: DSC by Segmentation Approach and Hemisphere\n'
                     f'Cross-validation Results',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Hemisphere', fontsize=14, fontweight='bold')
        ax.set_ylabel('DSC (volume-based)', fontsize=14, fontweight='bold')

        # Set custom x-axis labels for hemisphere groups
        ax.set_xticks([hemisphere_centers['Left'], hemisphere_centers['Right']])
        ax.set_xticklabels(['Left', 'Right'])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Add grid for better readability
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)

        # Create legend for segmentation approaches
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, edgecolor='black')
                          for approach, color in self.approach_colors.items()]

        # Add segmentation approach legend
        approach_legend = ax.legend(legend_elements, list(self.approach_colors.keys()),
                                   title='Segmentation Approach', title_fontsize=12, fontsize=11,
                                   loc='lower right', bbox_to_anchor=(1.0, 0.0))

        # Add median [IQR] explanation as text annotation
        ax.text(0.02, 0.98, 'Median [IQR]',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8),
               weight='bold')

        # Set fixed y-axis scaling from 0.75 to 1.0
        ax.set_ylim(0.75, 1.0)

        # Add statistical annotations and missing approach indicators
        self.add_stats_annotations(ax, df, missing_approaches, hemisphere_positions, config_name)

        # Add median and IQR values above each box
        self.add_median_iqr_labels(ax, df, hemisphere_positions)

        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {config_name} plot: {save_path}")

        plt.close()

    def add_stats_annotations(self, ax, df, missing_approaches, hemisphere_positions, config_name):
        """Add statistical information to the plot"""

        # Add sample size annotations and missing data indicators
        y_pos = ax.get_ylim()[1] - 0.01
        y_pos_missing = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5

        approach_order = ['Threshold', 'Single-Class', 'Single-Class HalfDPS', 'Multi-Class', 'Multi-Label']
        hemispheres = ['Left', 'Right']

        for hemi_idx, hemisphere in enumerate(hemispheres):
            for approach_idx, approach in enumerate(approach_order):
                if approach_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][approach_idx]

                    # Special handling for Threshold approach (CBF only)
                    if approach == 'Threshold' and config_name != 'CBF':
                        # Add "N/A" annotation for non-CBF configs
                        ax.text(box_position, y_pos_missing, 'N/A\n(CBF Only)',
                               ha='center', va='center', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8),
                               color='darkblue', weight='bold')
                    elif approach in missing_approaches:
                        # Add "No Data" annotation for missing approaches
                        ax.text(box_position, y_pos_missing, 'No Data\nAvailable',
                               ha='center', va='center', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8),
                               color='darkred', weight='bold')
                    else:
                        approach_data = df[(df['Approach'] == approach) & (df['Hemisphere'] == hemisphere)]
                        if not approach_data.empty:
                            n = len(approach_data)
                            ax.text(box_position, y_pos, f'n={n}',
                                   ha='center', va='bottom', fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    def add_median_iqr_labels(self, ax, df, hemisphere_positions):
        """Add median [IQR] labels above each box plot"""

        # Get y position for labels (above the plot area)
        y_max = ax.get_ylim()[1]
        label_y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02

        # Calculate positions for each box
        approach_order = ['Threshold', 'Single-Class', 'Single-Class HalfDPS', 'Multi-Class', 'Multi-Label']
        hemispheres = ['Left', 'Right']

        for hemi_idx, hemisphere in enumerate(hemispheres):
            for approach_idx, approach in enumerate(approach_order):
                if approach_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][approach_idx]

                    approach_data = df[(df['Approach'] == approach) & (df['Hemisphere'] == hemisphere)]

                    if not approach_data.empty and len(approach_data) > 0:
                        values = approach_data['DSC_Volume'].dropna()

                        if len(values) > 0:
                            median = values.median()
                            q1 = values.quantile(0.25)
                            q3 = values.quantile(0.75)
                            iqr = q3 - q1

                            y_pos = y_max + label_y_offset

                            # Format label: median [IQR] with 4 decimal places
                            label = f'{median:.4f} [{iqr:.4f}]'

                            # Use approach color for labels
                            color = self.approach_colors[approach]

                            ax.text(box_position, y_pos, label,
                                   ha='center', va='bottom', fontsize=8,
                                   color=color, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           alpha=0.8, edgecolor=color, linewidth=0.5))

        # Adjust y-axis limits to accommodate labels
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.15)

    def generate_all_plots(self):
        """Generate box plots for all input configurations"""

        print("Inter-Approach Segmentation Comparison Box Plot Generator")
        print("=" * 60)

        # Find all Excel files
        files_dict = self.find_excel_files()

        if not any(files_dict.values()):
            print("No Excel files found!")
            return

        print(f"\nGenerating plots...")

        # Generate plot for each input configuration
        for config_name, approach_files in files_dict.items():
            if not any(approach_files.values()):
                print(f"No files found for {config_name}")
                continue

            # Create save path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_filename = config_name.replace('+', '_')  # Replace + with _ for filename
            save_path = self.plots_dir / f"inter_approach_comparison_{config_filename}_{timestamp}.png"

            print(f"\nProcessing {config_name}...")
            self.create_input_config_plot(config_name, approach_files, save_path)

        print(f"\nAll plots saved to: {self.plots_dir}")

    def create_summary_comparison(self):
        """Create a summary comparison across all input configs and approaches"""

        files_dict = self.find_excel_files()

        # Collect summary statistics for each config and approach
        summary_data = []

        for config_name, approach_files in files_dict.items():
            for approach_name, file_path in approach_files.items():
                hemi_data = self.load_hemisphere_data(file_path, approach_name)

                for hemi in ['Left', 'Right']:
                    if hemi_data[hemi]:
                        values = np.array(hemi_data[hemi])
                        summary_data.append({
                            'Input_Config': config_name,
                            'Approach': approach_name,
                            'Hemisphere': hemi,
                            'Mean_DSC': np.mean(values),
                            'Std_DSC': np.std(values),
                            'Count': len(values)
                        })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            # Save summary statistics
            summary_path = self.plots_dir / f"inter_approach_summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            summary_df.to_excel(summary_path, index=False)
            print(f"Summary statistics saved: {summary_path}")


def main():
    # Set up paths
    results_dir = Path(__file__).parent.parent / "evaluation_results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Create plotter and generate plots
    plotter = InterApproachPlotter(results_dir)
    plotter.generate_all_plots()

    print("\nInter-approach segmentation comparison plots completed!")


if __name__ == "__main__":
    main()