#!/usr/bin/env python3
"""
Quality Control Script for Threshold Segmentation

This script creates quality control visualizations comparing ground truth segmentation
masks with newly generated threshold-based segmentation masks.

For each CBF file, it generates a PNG with:
- Top row: CBF map (grayscale) + Ground truth mask (red, 30% opacity) - 14 slices
- Bottom row: CBF map (grayscale) + Threshold mask (red, 30% opacity) - 14 slices

Input folders:
- data/images/ - CBF maps (PerfTerr###-v#-L/R_0000.nii)
- data/labels/ - Ground truth masks (PerfTerr###-v#-L/R.nii)
- data/thresholded_masks/ - Threshold masks (PerfTerr###-v#-L/R.nii)

Output: PNG files in quality_control/ folder

Author: Generated for threshold segmentation quality control
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class ThresholdQualityControl:
    def __init__(self, data_root, output_dir):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define folder paths
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        self.threshold_dir = self.data_root / "thresholded_masks"
        
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []
    
    def load_nifti_file(self, file_path):
        """Load a NIfTI file and return the data array"""
        try:
            nii = nib.load(file_path)
            return nii.get_fdata()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def normalize_for_display(self, data, percentile_range=(2, 98)):
        """Normalize CBF data for display, preserving zero values as black"""
        if data is None:
            return None
            
        # Create normalized array
        normalized = np.zeros_like(data, dtype=np.float32)
        
        # Only normalize non-zero values
        nonzero_mask = data > 0
        if np.any(nonzero_mask):
            data_min, data_max = np.percentile(data[nonzero_mask], percentile_range)
            if data_max > data_min:
                normalized[nonzero_mask] = np.clip(
                    (data[nonzero_mask] - data_min) / (data_max - data_min), 0, 1
                )
            else:
                normalized[nonzero_mask] = 1.0
        
        return normalized
    
    def create_comparison_visualization(self, cbf_data, gt_mask, threshold_mask, title, output_file):
        """Create comparison visualization: GT vs Threshold masks"""
        if cbf_data is None or gt_mask is None or threshold_mask is None:
            print(f"Warning: Missing data for {title}")
            return False
        
        # Ensure all data shapes match
        if not (cbf_data.shape == gt_mask.shape == threshold_mask.shape):
            print(f"Warning: Shape mismatch for {title}")
            print(f"  CBF: {cbf_data.shape}, GT: {gt_mask.shape}, Threshold: {threshold_mask.shape}")
            return False
        
        # Get dimensions (expecting 14 slices)
        nx, ny, nz = cbf_data.shape
        if nz != 14:
            print(f"Warning: Expected 14 slices, got {nz} for {title}")
        
        # Create figure with 2 rows for 14 slices each
        fig, axes = plt.subplots(2, 14, figsize=(28, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Normalize CBF data for display
        cbf_norm = self.normalize_for_display(cbf_data, (2, 98))
        
        # Process each slice
        for z in range(min(nz, 14)):  # Ensure we don't exceed 14 columns
            # Top row: CBF + Ground Truth mask
            ax_top = axes[0, z]
            
            # Get slices and flip for correct orientation
            cbf_slice = np.flipud(cbf_norm[:, :, z])
            gt_slice = np.flipud(gt_mask[:, :, z])
            
            # Display CBF as grayscale background
            ax_top.imshow(cbf_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            
            # Overlay ground truth mask in red where mask > 0
            gt_overlay = np.zeros((*gt_slice.shape, 4))
            gt_overlay[:, :, 0] = 1.0  # Red channel
            gt_overlay[:, :, 3] = (gt_slice > 0) * 0.3  # Alpha for visibility
            ax_top.imshow(gt_overlay.transpose(1, 0, 2), origin='lower')
            
            ax_top.set_title(f'Slice {z+1}', fontsize=10)
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            
            # Bottom row: CBF + Threshold mask
            ax_bottom = axes[1, z]
            
            # Get threshold mask slice
            threshold_slice = np.flipud(threshold_mask[:, :, z])
            
            # Display CBF as grayscale background
            ax_bottom.imshow(cbf_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            
            # Overlay threshold mask in red where mask > 0
            threshold_overlay = np.zeros((*threshold_slice.shape, 4))
            threshold_overlay[:, :, 0] = 1.0  # Red channel
            threshold_overlay[:, :, 3] = (threshold_slice > 0) * 0.3  # Alpha for visibility
            ax_bottom.imshow(threshold_overlay.transpose(1, 0, 2), origin='lower')
            
            ax_bottom.set_title(f'Slice {z+1}', fontsize=10)
            ax_bottom.set_xticks([])
            ax_bottom.set_yticks([])
        
        # Hide unused columns if less than 14 slices
        for z in range(nz, 14):
            axes[0, z].set_visible(False)
            axes[1, z].set_visible(False)
        
        # Add row labels
        axes[0, 0].set_ylabel('CBF + Ground Truth\n(red)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('CBF + Threshold\n(red)', fontsize=12, fontweight='bold')
        
        # Add legends
        legend_elements = [
            patches.Patch(color='gray', label='CBF (grayscale)'),
            patches.Patch(color='red', alpha=0.3, label='Ground Truth Mask'),
            patches.Patch(color='red', alpha=0.3, label='Threshold Mask', linestyle='--')
        ]
        fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    
    def process_datapoint(self, base_name):
        """Process a single datapoint (CBF image + GT mask + Threshold mask)"""
        # Construct file paths
        cbf_file = self.images_dir / f"{base_name}_0000.nii"
        gt_file = self.labels_dir / f"{base_name}.nii"
        threshold_file = self.threshold_dir / f"{base_name}.nii"
        
        # Check if all files exist
        missing_files = []
        if not cbf_file.exists():
            missing_files.append(f"CBF: {cbf_file}")
        if not gt_file.exists():
            missing_files.append(f"GT: {gt_file}")
        if not threshold_file.exists():
            missing_files.append(f"Threshold: {threshold_file}")
        
        if missing_files:
            print(f"Warning: Missing files for {base_name}:")
            for missing in missing_files:
                print(f"  {missing}")
            self.failed_files.append(base_name)
            return False
        
        # Load data
        print(f"Processing {base_name}...")
        cbf_data = self.load_nifti_file(cbf_file)
        gt_data = self.load_nifti_file(gt_file)
        threshold_data = self.load_nifti_file(threshold_file)
        
        if cbf_data is None or gt_data is None or threshold_data is None:
            self.failed_files.append(base_name)
            return False
        
        # Create output filename
        output_filename = f"{base_name}_quality_control.png"
        output_file = self.output_dir / output_filename
        
        # Create visualization
        title = f"Quality Control: {base_name}"
        
        success = self.create_comparison_visualization(
            cbf_data, gt_data, threshold_data, title, output_file
        )
        
        if success:
            print(f"  OK Saved: {output_filename}")
        else:
            print(f"  FAILED: {output_filename}")
            self.failed_files.append(base_name)
        
        return success
    
    def find_all_datapoints(self):
        """Find all unique datapoint base names from the images folder"""
        datapoints = []
        if not self.images_dir.exists():
            return datapoints
        
        # Find all CBF files and extract base names
        for cbf_file in self.images_dir.glob("*.nii"):
            if cbf_file.stem.endswith("_0000"):
                base_name = cbf_file.stem.replace("_0000", "")
                datapoints.append(base_name)
        
        return sorted(datapoints)
    
    def run_quality_control(self):
        """Run the complete quality control analysis"""
        print("Threshold Segmentation Quality Control")
        print("=" * 60)
        print(f"Data root: {self.data_root}")
        print(f"CBF images: {self.images_dir}")
        print(f"Ground truth labels: {self.labels_dir}")
        print(f"Threshold masks: {self.threshold_dir}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        
        # Check if required directories exist
        missing_dirs = []
        if not self.images_dir.exists():
            missing_dirs.append(f"Images: {self.images_dir}")
        if not self.labels_dir.exists():
            missing_dirs.append(f"Labels: {self.labels_dir}")
        if not self.threshold_dir.exists():
            missing_dirs.append(f"Threshold masks: {self.threshold_dir}")
        
        if missing_dirs:
            print("ERROR: Missing required directories:")
            for missing in missing_dirs:
                print(f"  {missing}")
            return False
        
        # Find all datapoints
        datapoints = self.find_all_datapoints()
        if not datapoints:
            print(f"ERROR: No CBF files found in {self.images_dir}")
            return False
        
        print(f"Found {len(datapoints)} datapoints to process")
        print()
        
        # Process each datapoint
        for datapoint in datapoints:
            self.processed_count += 1
            if self.process_datapoint(datapoint):
                self.successful_count += 1
            print()
        
        # Summary
        print("=" * 60)
        print("QUALITY CONTROL SUMMARY:")
        print(f"Total datapoints processed: {self.processed_count}")
        print(f"Successful visualizations: {self.successful_count}")
        print(f"Failed datapoints: {len(self.failed_files)}")
        if self.failed_files:
            print(f"Failed datapoints: {', '.join(self.failed_files)}")
        print(f"Success rate: {self.successful_count/self.processed_count*100:.1f}%")
        print(f"PNG files saved in: {self.output_dir}")
        
        return True


def main():
    """Main function to run quality control"""
    # Default paths
    script_dir = Path(__file__).parent
    default_data_root = script_dir.parent / "data"
    default_output_dir = script_dir
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = default_data_root
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output_dir
    
    print("Threshold Segmentation Quality Control Script")
    print("=" * 60)
    
    # Initialize quality control
    qc = ThresholdQualityControl(data_root, output_dir)
    
    # Run quality control
    success = qc.run_quality_control()
    
    if success:
        print(f"\nQuality control completed!")
        print(f"Comparison visualizations saved in: {output_dir}")
        print(f"\nEach PNG contains:")
        print(f"  - Top row: CBF (grayscale) + Ground Truth mask (red, 30%) - 14 slices")
        print(f"  - Bottom row: CBF (grayscale) + Threshold mask (red, 30%) - 14 slices")
    else:
        print("Quality control failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)