#!/usr/bin/env python3
"""
Quality Control Visualization Script

This script creates side-by-side visualizations of CBF maps with their corresponding
thresholded segmentations overlaid. Each datapoint generates one image showing all
slices arranged in a grid format.

Usage:
    python quality_control_visualization.py
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import math

def load_nifti_data(file_path):
    """Load NIfTI file and return data array."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def normalize_data(data):
    """Normalize data to 0-1 range for visualization."""
    if data.max() == data.min():
        return np.zeros_like(data)
    return (data - data.min()) / (data.max() - data.min())

def create_slice_visualization(cbf_data, threshold_data, title, output_path):
    """
    Create a grid visualization of all 16 axial slices with threshold overlay.
    
    Args:
        cbf_data: Original CBF data (3D array - shape: H x W x slices)
        threshold_data: Thresholded binary data (3D array - shape: H x W x slices)
        title: Title for the visualization
        output_path: Path to save the image
    """
    # Get all slices (should be 16 axial slices) - slices are in the 3rd dimension
    total_slices = cbf_data.shape[2]
    slice_indices = list(range(total_slices))  # Use all slices
    num_slices = len(slice_indices)
    
    print(f"  Data shape: {cbf_data.shape}, Total slices: {total_slices}")
    
    # Calculate grid dimensions for 16 slices (4x4 grid)
    grid_cols = 4
    grid_rows = 4
    
    # Create figure sized for 4x4 grid with proper aspect ratio
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 16))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Flatten axes array for easier indexing (always 4x4 grid)
    axes = axes.flatten()
    
    # Use original CBF data without normalization (keep original intensity values)
    # Zero values will be black, max values will be white
    
    # Process each slice
    for i in range(16):  # Always process exactly 16 positions in 4x4 grid
        ax = axes[i]
        
        if i < num_slices:
            # Display actual slice - extract from 3rd dimension
            slice_idx = slice_indices[i]
            
            # Get slice data (H x W from the slice dimension) - use original values
            cbf_slice = cbf_data[:, :, slice_idx]  # Original intensities, not normalized
            threshold_slice = threshold_data[:, :, slice_idx]
            
            # Fix orientation: rotate counterclockwise 90 degrees and flip vertically
            # This should match typical medical image viewer orientation
            cbf_slice = np.rot90(cbf_slice, k=3)  # Rotate counterclockwise 90° (k=3 = 270° clockwise = 90° counterclockwise)
            cbf_slice = np.flipud(cbf_slice)      # Flip vertically
            
            threshold_slice = np.rot90(threshold_slice, k=3)
            threshold_slice = np.flipud(threshold_slice)
            
            # Display CBF slice with original intensities (0=black, max=white) - no alpha blending
            ax.imshow(cbf_slice, cmap='gray', alpha=1.0, aspect='equal', vmin=0, vmax=cbf_data.max())
            
            # Overlay threshold mask in red
            threshold_overlay = np.ma.masked_where(threshold_slice == 0, threshold_slice)
            ax.imshow(threshold_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1, aspect='equal')
            
            # Set title and remove ticks
            ax.set_title(f'Slice {slice_idx+1}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide unused subplot
            ax.set_visible(False)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='gray', label='CBF Map'),
        patches.Patch(color='red', alpha=0.6, label='Threshold Overlay')
    ]
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved visualization: {output_path.name}")

def main():
    """Main function to process all CBF maps and create visualizations."""
    # Define paths
    script_dir = Path(__file__).parent
    cbf_maps_dir = script_dir / 'CBF_maps'
    threshold_dir = script_dir / 'thresholding_segmentations'
    output_dir = script_dir / 'quality_control'
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check if directories exist
    if not cbf_maps_dir.exists():
        print(f"Error: CBF_maps directory not found at {cbf_maps_dir}")
        return
    
    if not threshold_dir.exists():
        print(f"Error: thresholding_segmentations directory not found at {threshold_dir}")
        return
    
    # Find all CBF map files
    cbf_files = []
    for ext in ['*.nii', '*.nii.gz']:
        cbf_files.extend(cbf_maps_dir.glob(ext))
    
    if not cbf_files:
        print(f"No NIfTI files found in {cbf_maps_dir}")
        return
    
    print(f"Found {len(cbf_files)} CBF map files")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Process each CBF file
    processed_count = 0
    for cbf_file in sorted(cbf_files):
        print(f"\nProcessing: {cbf_file.name}")
        
        # Find corresponding threshold file
        threshold_file = threshold_dir / cbf_file.name
        
        if not threshold_file.exists():
            print(f"  Warning: Threshold file not found: {threshold_file.name}")
            continue
        
        # Load data
        cbf_data = load_nifti_data(cbf_file)
        threshold_data = load_nifti_data(threshold_file)
        
        if cbf_data is None or threshold_data is None:
            print(f"  Error: Could not load data for {cbf_file.name}")
            continue
        
        # Check data compatibility
        if cbf_data.shape != threshold_data.shape:
            print(f"  Error: Shape mismatch - CBF: {cbf_data.shape}, Threshold: {threshold_data.shape}")
            continue
        
        # Create visualization
        output_filename = f"QC_{cbf_file.stem}.png"
        output_path = output_dir / output_filename
        
        # Create title with file info
        title = f"Quality Control: {cbf_file.name}\nCBF Map (Gray) + Threshold Overlay (Red)"
        
        try:
            create_slice_visualization(cbf_data, threshold_data, title, output_path)
            processed_count += 1
            
            # Print some statistics
            total_voxels = threshold_data.size
            binary_voxels = np.sum(threshold_data == 1)
            percentage = (binary_voxels / total_voxels) * 100
            
            print(f"  Shape: {cbf_data.shape}")
            print(f"  Binary voxels: {binary_voxels}/{total_voxels} ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"  Error creating visualization: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Quality control completed!")
    print(f"Successfully processed: {processed_count}/{len(cbf_files)} files")
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()