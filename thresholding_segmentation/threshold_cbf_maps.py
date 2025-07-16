#!/usr/bin/env python3
"""
CBF Thresholding Script

This script loads NIfTI files from the CBF_maps folder and applies binary thresholding
based on 70% of the median grey value. Thresholded maps are saved to the current directory
with modified filenames.

Usage:
    python threshold_cbf_maps.py
"""

import numpy as np
import nibabel as nib
from pathlib import Path

def load_nifti(file_path):
    """Load a NIfTI file and return the image object and data array."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return img, data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def apply_threshold(data, threshold_percentile=1.0):
    """
    Apply binary thresholding based on percentage of mean value.
    
    Args:
        data: Input image data array
        threshold_percentile: Percentage of mean to use as threshold (default 1.0 = 100%)
    
    Returns:
        Binary thresholded array (0s and 1s)
    """
    # Calculate mean of non-zero values
    non_zero_values = data[data > 0]
    if len(non_zero_values) == 0:
        print("Warning: No non-zero values found in image")
        return np.zeros_like(data)
    
    mean_value = np.mean(non_zero_values)
    threshold = mean_value * threshold_percentile
    
    print(f"  Mean value: {mean_value:.2f}")
    print(f"  Threshold (100%): {threshold:.2f}")
    
    # Apply binary thresholding
    binary_data = np.where(data >= threshold, 1, 0)
    
    return binary_data.astype(np.uint8)

def modify_filename(filename):
    """
    Keep the original filename unchanged.
    
    Args:
        filename: Original filename
    
    Returns:
        Unchanged filename
    """
    # Keep original filename without modifications
    return filename

def main():
    """Main function to process all CBF maps."""
    # Define paths
    script_dir = Path(__file__).parent
    cbf_maps_dir = script_dir / 'CBF_maps'
    output_dir = script_dir / 'thresholding_segmentations'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Check if CBF_maps directory exists
    if not cbf_maps_dir.exists():
        print(f"Error: CBF_maps directory not found at {cbf_maps_dir}")
        return
    
    # Find all NIfTI files in CBF_maps directory
    nifti_files = []
    for ext in ['*.nii', '*.nii.gz']:
        nifti_files.extend(cbf_maps_dir.glob(ext))
    
    if not nifti_files:
        print(f"No NIfTI files found in {cbf_maps_dir}")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files to process")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Process each file
    for file_path in sorted(nifti_files):
        print(f"\nProcessing: {file_path.name}")
        
        # Load the NIfTI file
        img, data = load_nifti(file_path)
        if img is None or data is None:
            continue
        
        # Apply thresholding
        binary_data = apply_threshold(data, threshold_percentile=1.0)
        
        # Create output filename
        output_filename = modify_filename(file_path.name)
        output_path = output_dir / output_filename
        
        # Create new NIfTI image with thresholded data
        binary_img = nib.Nifti1Image(binary_data, img.affine, img.header)
        
        # Save the thresholded image
        try:
            nib.save(binary_img, output_path)
            print(f"  Saved: {output_filename}")
            
            # Print some statistics
            original_range = f"[{data.min():.2f}, {data.max():.2f}]"
            binary_voxels = np.sum(binary_data == 1)
            total_voxels = binary_data.size
            percentage = (binary_voxels / total_voxels) * 100
            
            print(f"  Original range: {original_range}")
            print(f"  Binary voxels: {binary_voxels}/{total_voxels} ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"  Error saving {output_filename}: {e}")
    
    print("\n" + "=" * 60)
    print("Thresholding completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()