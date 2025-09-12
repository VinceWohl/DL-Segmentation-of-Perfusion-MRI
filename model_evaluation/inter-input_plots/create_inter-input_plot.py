#!/usr/bin/env python3
"""
Inter-Input Channel Comparison Box Plots for Cross-Validation Results

Generates box plots comparing the influence of different input channel configurations
across all segmentation approaches:
- single-class
- single-class_halfdps  
- multi-class
- multi-label

For each approach, creates plots showing DSC_Volume performance by:
- Hemisphere (Left/Right) 
- Input Configuration (CBF, CBF+T1w, CBF+FLAIR, CBF+T1w+FLAIR)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class InterInputPlotter:
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
        
        # Color palette for input configurations
        self.input_colors = {
            'CBF': '#1f77b4',           # Blue
            'CBF+T1w': '#ff7f0e',       # Orange  
            'CBF+FLAIR': '#2ca02c',     # Green
            'CBF+T1w+FLAIR': '#d62728'  # Red
        }
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def find_excel_files(self):
        """Find and categorize all crossval Excel files by approach and input config"""
        files = {}
        
        # Define base patterns for each approach
        base_patterns = {
            'single-class': 'crossval_singleclass_',
            'single-class_halfdps': 'crossval_singleclass_halfdps_', 
            'multi-class': 'crossval_multiclass_',
            'multi-label': 'crossval_multilabel_'
        }
        
        for approach, base_pattern in base_patterns.items():
            files[approach] = {}
            
            for config_name, config_patterns in self.input_configs.items():
                matches = []
                
                # Look for files matching any of the config patterns
                for config_pattern in config_patterns:
                    search_pattern = f"{base_pattern}{config_pattern}*.xlsx"
                    found_files = list(self.results_dir.glob(search_pattern))
                    
                    # For CBF-only, ensure we don't match multi-modal files
                    if config_name == 'CBF':
                        found_files = [f for f in found_files 
                                     if 'T1w' not in f.name and 'FLAIR' not in f.name]
                    
                    matches.extend(found_files)
                
                if matches:
                    # Use the most recent file if multiple exist
                    files[approach][config_name] = sorted(matches)[-1]
                    print(f"Found {approach} {config_name}: {matches[-1].name}")
        
        return files
        
    def load_hemisphere_data(self, file_path, approach):
        """Load hemisphere-specific data from Excel file"""
        try:
            if approach in ['single-class', 'single-class_halfdps', 'multi-label']:
                # Load Per_Case_Details sheet with Hemisphere column
                df = pd.read_excel(file_path, sheet_name='Per_Case_Details')
                
                # Extract data for each hemisphere
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi]['DSC_Volume'].dropna()
                    data[hemi] = hemi_data.tolist()
                
                return data
                
            elif approach == 'multi-class':
                # Load Per_Hemisphere_Details sheet 
                df = pd.read_excel(file_path, sheet_name='Per_Hemisphere_Details')
                
                # Extract data for each hemisphere
                data = {}
                for hemi in ['Left', 'Right']:
                    hemi_data = df[df['Hemisphere'] == hemi]['DSC_Volume'].dropna()
                    data[hemi] = hemi_data.tolist()
                    
                return data
                
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {'Left': [], 'Right': []}
            
    def create_approach_plot(self, approach, files_dict, save_path):
        """Create box plot for a specific segmentation approach"""
        
        # Collect all data for this approach
        plot_data = []
        missing_configs = []
        
        for config_name in ['CBF', 'CBF+T1w', 'CBF+FLAIR', 'CBF+T1w+FLAIR']:
            if config_name in files_dict:
                file_path = files_dict[config_name]
                hemi_data = self.load_hemisphere_data(file_path, approach)
                
                # Add data for each hemisphere
                for hemi in ['Left', 'Right']:
                    for value in hemi_data[hemi]:
                        plot_data.append({
                            'Input_Config': config_name,
                            'Hemisphere': hemi, 
                            'DSC_Volume': value
                        })
            else:
                missing_configs.append(config_name)
        
        if not plot_data:
            print(f"No data found for {approach}")
            return
            
        df = pd.DataFrame(plot_data)
        
        # Create the plot
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
                config_data = df[(df['Hemisphere'] == hemisphere) & (df['Input_Config'] == config)]
                
                if not config_data.empty:
                    plot_data_list.append(config_data['DSC_Volume'].values)
                else:
                    # Add empty data for missing configurations
                    plot_data_list.append([])
                
                plot_colors.append(self.input_colors[config])
                plot_labels.append(f'{hemisphere}_{config}')
                plot_positions.append(position)
                hemisphere_positions[hemisphere].append(position)
                position += 0.7  # Further reduced spacing within hemisphere
            
            # Calculate hemisphere center for labeling
            hemisphere_centers[hemisphere] = (start_pos + position - 0.7) / 2
            
            # Add gap between hemispheres
            if hemi_idx < len(hemispheres) - 1:
                position += 0.3  # Minimal gap between hemispheres
        
        # Create box plots manually with proper positioning
        box_parts = ax.boxplot(
            plot_data_list,
            positions=plot_positions,
            notch=True,
            patch_artist=True,
            widths=0.5  # Slightly reduced width to accommodate closer spacing
        )
        
        # Color the boxes according to input configuration
        for patch, color in zip(box_parts['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        # Customize the plot
        ax.set_title(f'{approach.replace("_", " ").title()} Segmentation: DSC by Input Configuration and Hemisphere', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Hemisphere', fontsize=14, fontweight='bold')
        ax.set_ylabel('DSC Score', fontsize=14, fontweight='bold')
        
        # Set custom x-axis labels for hemisphere groups
        ax.set_xticks([hemisphere_centers['Left'], hemisphere_centers['Right']])
        ax.set_xticklabels(['Left', 'Right'])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Add grid for better readability with more significant horizontal lines
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Create legend for input configurations
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, edgecolor='black') 
                          for config, color in self.input_colors.items()]
        
        # Add explanation for median [IQR] labels
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        # Add input configuration legend
        config_legend = ax.legend(legend_elements, list(self.input_colors.keys()), 
                                 title='Input Configuration', title_fontsize=12, fontsize=11, 
                                 loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        # Add median [IQR] explanation as text annotation
        ax.text(0.02, 0.98, 'Numbers above boxes: Median [IQR]', 
               transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8),
               weight='bold')
        
        # Set fixed y-axis scaling from 0.75 to 1.0 as requested
        ax.set_ylim(0.75, 1.0)
        
        # Add statistical annotations and missing config indicators
        self.add_stats_annotations(ax, df, missing_configs, hemisphere_positions)
        
        # Add median and IQR values above each box
        self.add_median_iqr_labels(ax, df, hemisphere_positions)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved {approach} plot: {save_path}")
        
        plt.close()
        
    def add_stats_annotations(self, ax, df, missing_configs, hemisphere_positions):
        """Add statistical information to the plot"""
        
        # Add sample size annotations and missing data indicators
        y_pos = ax.get_ylim()[1] - 0.01
        y_pos_missing = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5
        
        configs = ['CBF', 'CBF+T1w', 'CBF+FLAIR', 'CBF+T1w+FLAIR']
        hemispheres = ['Left', 'Right']
        
        for hemi_idx, hemisphere in enumerate(hemispheres):
            for config_idx, config in enumerate(configs):
                position_idx = hemi_idx * len(configs) + config_idx
                if position_idx < len(hemisphere_positions['Left']) + len(hemisphere_positions['Right']):
                    if hemisphere == 'Left':
                        box_position = hemisphere_positions['Left'][config_idx]
                    else:
                        box_position = hemisphere_positions['Right'][config_idx]
                    
                    if config in missing_configs:
                        # Add "No Data" annotation for missing configs
                        ax.text(box_position, y_pos_missing, 'No Data\\nAvailable', 
                               ha='center', va='center', fontsize=10, 
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8),
                               color='darkred', weight='bold')
                    else:
                        config_data = df[(df['Input_Config'] == config) & (df['Hemisphere'] == hemisphere)]
                        if not config_data.empty:
                            n = len(config_data)
                            ax.text(box_position, y_pos, f'n={n}', 
                                   ha='center', va='bottom', fontsize=8, 
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    def add_median_iqr_labels(self, ax, df, hemisphere_positions):
        """Add median [IQR] labels above each box plot"""
        
        # Get y position for labels (above the plot area)
        y_max = ax.get_ylim()[1]
        label_y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
        
        # Calculate positions for each box
        configs = ['CBF', 'CBF+T1w', 'CBF+FLAIR', 'CBF+T1w+FLAIR']
        hemispheres = ['Left', 'Right']
        
        for hemi_idx, hemisphere in enumerate(hemispheres):
            for config_idx, config in enumerate(configs):
                if config_idx < len(hemisphere_positions[hemisphere]):
                    box_position = hemisphere_positions[hemisphere][config_idx]
                    
                    config_data = df[(df['Input_Config'] == config) & (df['Hemisphere'] == hemisphere)]
                    
                    if not config_data.empty and len(config_data) > 0:
                        values = config_data['DSC_Volume'].dropna()
                        
                        if len(values) > 0:
                            median = values.median()
                            q1 = values.quantile(0.25)
                            q3 = values.quantile(0.75)
                            iqr = q3 - q1
                            
                            y_pos = y_max + label_y_offset
                            
                            # Format label: median [IQR] with 4 decimal places
                            label = f'{median:.4f} [{iqr:.4f}]'
                            
                            # Use input configuration color for labels
                            color = self.input_colors[config]
                            
                            ax.text(box_position, y_pos, label,
                                   ha='center', va='bottom', fontsize=8,
                                   color=color, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                           alpha=0.8, edgecolor=color, linewidth=0.5))
        
        # Adjust y-axis limits to accommodate labels
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], current_ylim[1] + (current_ylim[1] - current_ylim[0]) * 0.15)
        
    def generate_all_plots(self):
        """Generate box plots for all segmentation approaches"""
        
        print("Inter-Input Channel Comparison Box Plot Generator")
        print("=" * 60)
        
        # Find all Excel files
        files_dict = self.find_excel_files()
        
        if not any(files_dict.values()):
            print("No Excel files found!")
            return
            
        print(f"\\nGenerating plots...")
        
        # Generate plot for each approach
        for approach, config_files in files_dict.items():
            if not config_files:
                print(f"No files found for {approach}")
                continue
                
            # Create save path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            save_path = self.plots_dir / f"inter_input_comparison_{approach}_{timestamp}.png"
            
            print(f"\\nProcessing {approach}...")
            self.create_approach_plot(approach, config_files, save_path)
        
        print(f"\\nAll plots saved to: {self.plots_dir}")
        
    def create_summary_comparison(self):
        """Create a summary comparison across all approaches"""
        
        files_dict = self.find_excel_files()
        
        # Collect summary statistics for each approach and config
        summary_data = []
        
        for approach, config_files in files_dict.items():
            for config_name, file_path in config_files.items():
                hemi_data = self.load_hemisphere_data(file_path, approach)
                
                for hemi in ['Left', 'Right']:
                    if hemi_data[hemi]:
                        values = np.array(hemi_data[hemi])
                        summary_data.append({
                            'Approach': approach.replace('_', ' ').title(),
                            'Input_Config': config_name,
                            'Hemisphere': hemi,
                            'Mean_DSC': np.mean(values),
                            'Std_DSC': np.std(values),
                            'Count': len(values)
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary statistics
            summary_path = self.plots_dir / f"inter_input_summary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            summary_df.to_excel(summary_path, index=False)
            print(f"Summary statistics saved: {summary_path}")


def main():
    # Set up paths
    results_dir = Path(__file__).parent.parent / "evaluation_results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Create plotter and generate plots
    plotter = InterInputPlotter(results_dir)
    plotter.generate_all_plots()
    plotter.create_summary_comparison()
    
    print("\\nInter-input channel comparison plots completed!")


if __name__ == "__main__":
    main()