#!/usr/bin/env python3
"""
Script to fix NaN values in 250920 dataset images.
Replaces all NaN values with 0.0 in the CBF image files.
"""

import nibabel as nib
import numpy as np
import os
from pathlib import Path

def fix_nan_values_in_image(image_path):
    """
    Load image, replace NaN values with 0.0, and save back to same location.

    Args:
        image_path (str): Path to the NIfTI image file

    Returns:
        tuple: (original_nan_count, total_voxels, filename)
    """
    # Load the image
    img = nib.load(image_path)
    data = img.get_fdata()

    # Count NaN values before fixing
    nan_mask = np.isnan(data)
    original_nan_count = nan_mask.sum()
    total_voxels = data.size

    if original_nan_count > 0:
        # Replace NaN values with 0.0
        data[nan_mask] = 0.0

        # Create new image with fixed data
        fixed_img = nib.Nifti1Image(data, img.affine, img.header)

        # Save back to the same location (overwrites original)
        nib.save(fixed_img, image_path)

        print(f"✓ Fixed {original_nan_count:6d} NaN values in {os.path.basename(image_path)}")

    return original_nan_count, total_voxels, os.path.basename(image_path)

def main():
    # Path to 250920 dataset images
    images_dir = "/home/ubuntu/DLSegPerf/data/other/nnUNet_raw_250920-PerfTerr-41/Dataset001_PerfusionTerritories/imagesTr/"

    print("=" * 80)
    print("FIXING NaN VALUES IN 250920 DATASET IMAGES")
    print("=" * 80)
    print(f"Target directory: {images_dir}")
    print()

    # Check if directory exists
    if not os.path.exists(images_dir):
        print(f"ERROR: Directory not found: {images_dir}")
        return

    # Get all NIfTI files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii')])

    if not image_files:
        print("ERROR: No .nii files found in the directory")
        return

    print(f"Found {len(image_files)} image files to process")
    print()

    # Process each image file
    total_nan_fixed = 0
    total_voxels_processed = 0
    files_with_nans = 0

    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_file)

        try:
            nan_count, voxel_count, filename = fix_nan_values_in_image(image_path)

            if nan_count > 0:
                files_with_nans += 1

            total_nan_fixed += nan_count
            total_voxels_processed += voxel_count

            # Progress indicator
            if i % 10 == 0 or i == len(image_files):
                print(f"Progress: {i}/{len(image_files)} files processed")

        except Exception as e:
            print(f"✗ ERROR processing {image_file}: {e}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed:     {len(image_files)}")
    print(f"Files with NaN values:     {files_with_nans}")
    print(f"Total NaN values fixed:    {total_nan_fixed:,}")
    print(f"Total voxels processed:    {total_voxels_processed:,}")
    print(f"NaN percentage fixed:      {100 * total_nan_fixed / total_voxels_processed:.3f}%")
    print()

    if total_nan_fixed > 0:
        print("✓ All NaN values have been replaced with 0.0")
        print("✓ Dataset is now ready for training")
    else:
        print("ℹ No NaN values found - dataset was already clean")

    print()
    print("VERIFICATION:")
    print("You can verify the fix by running:")
    print("python3 -c \"import nibabel as nib; import numpy as np; import os\"")
    print("python3 -c \"files = [f for f in os.listdir('{}') if f.endswith('.nii')]; print('NaN check:', sum(np.isnan(nib.load(os.path.join('{}', f)).get_fdata()).sum() for f in files[:5]))\"".format(images_dir, images_dir))

if __name__ == "__main__":
    main()