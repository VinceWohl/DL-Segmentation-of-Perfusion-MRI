#!/usr/bin/env python3
"""
Threshold-Based Segmentation Script

This script performs threshold-based segmentation on CBF maps using the mean intensity
as the binary threshold. All voxels below the mean are set to 0, and all voxels
above the mean are set to 1.

Input: CBF images in data/imagesTr/ and data/imagesTs/ with pattern PerfTerr###-v#-L/R_0000.nii
Output: Binary masks in data/thresholded_labelsTr/ and data/thresholded_labelsTs/ with pattern PerfTerr###-v#-L/R.nii

Author: Generated for threshold segmentation analysis
"""

import sys
import numpy as np
from pathlib import Path

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class ThresholdSegmentation:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_count = 0
        self.successful_count = 0
        self.failed_files = []
    
    def load_cbf_image(self, file_path):
        """Load CBF image and return data array and NIfTI object"""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            return data, nii
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def compute_mean_threshold(self, cbf_data):
        """Compute mean intensity threshold for CBF data (0.9 * mean)"""
        # Only consider non-zero voxels for mean calculation
        nonzero_mask = cbf_data > 0
        if np.any(nonzero_mask):
            mean_value = np.mean(cbf_data[nonzero_mask])
            mean_threshold = 0.9 * mean_value
            return mean_threshold
        else:
            print("Warning: No non-zero voxels found in CBF data")
            return 0.0
    
    def create_binary_mask(self, cbf_data, threshold):
        """Create binary segmentation mask based on threshold"""
        # Create binary mask: 1 for voxels above threshold, 0 for below
        binary_mask = (cbf_data >= threshold).astype(np.uint8)
        return binary_mask
    
    def process_cbf_file(self, input_file):
        """Process a single CBF file and create threshold segmentation"""
        print(f"Processing {input_file.name}...")
        
        # Load CBF data
        cbf_data, nii_obj = self.load_cbf_image(input_file)
        if cbf_data is None or nii_obj is None:
            self.failed_files.append(str(input_file))
            return False
        
        # Compute mean threshold
        mean_threshold = self.compute_mean_threshold(cbf_data)
        print(f"  Mean threshold: {mean_threshold:.3f}")
        
        # Create binary mask
        binary_mask = self.create_binary_mask(cbf_data, mean_threshold)
        
        # Count segmented voxels
        segmented_voxels = np.sum(binary_mask)
        total_voxels = binary_mask.size
        percentage = (segmented_voxels / total_voxels) * 100
        print(f"  Segmented voxels: {segmented_voxels}/{total_voxels} ({percentage:.1f}%)")
        
        # Create output filename (remove _0000 suffix)
        output_name = input_file.stem.replace('_0000', '') + '.nii'
        output_file = self.output_dir / output_name
        
        # Save binary mask with same header as original
        try:
            binary_nii = nib.Nifti1Image(binary_mask, nii_obj.affine, nii_obj.header)
            nib.save(binary_nii, output_file)
            print(f"  Saved: {output_name}")
            return True
        except Exception as e:
            print(f"  Error saving {output_name}: {e}")
            self.failed_files.append(str(input_file))
            return False
    
    def run_segmentation(self):
        """Run threshold segmentation on all CBF files"""
        print("Threshold-Based Segmentation")
        print("=" * 50)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Threshold method: Mean intensity of non-zero voxels")
        print()
        
        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        
        if not self.input_dir.exists():
            print(f"ERROR: Input directory does not exist: {self.input_dir}")
            return False
        
        # Find all CBF image files
        cbf_files = list(self.input_dir.glob("*.nii"))
        if not cbf_files:
            print(f"ERROR: No .nii files found in {self.input_dir}")
            return False
        
        print(f"Found {len(cbf_files)} CBF files to process")
        print()
        
        # Process each file
        for cbf_file in sorted(cbf_files):
            self.processed_count += 1
            if self.process_cbf_file(cbf_file):
                self.successful_count += 1
            print()
        
        # Summary
        print("=" * 50)
        print("THRESHOLD SEGMENTATION SUMMARY:")
        print(f"Total files processed: {self.processed_count}")
        print(f"Successful segmentations: {self.successful_count}")
        print(f"Failed files: {len(self.failed_files)}")
        if self.failed_files:
            print(f"Failed files list: {self.failed_files}")
        print(f"Success rate: {self.successful_count/self.processed_count*100:.1f}%")
        print(f"Output directory: {self.output_dir}")
        
        return True


def main():
    """Main function to run threshold segmentation"""
    script_dir = Path(__file__).parent

    # Define input and output directories
    datasets = [
        {
            'input_dir': script_dir / "data" / "imagesTr",
            'output_dir': script_dir / "data" / "thresholded_labelsTr",
            'name': 'Training'
        },
        {
            'input_dir': script_dir / "data" / "imagesTs",
            'output_dir': script_dir / "data" / "thresholded_labelsTs",
            'name': 'Test'
        }
    ]

    print("Threshold-Based Segmentation Script")
    print("=" * 50)

    total_success = True

    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']} Dataset")
        print("-" * 40)

        # Initialize segmentation processor
        segmenter = ThresholdSegmentation(dataset['input_dir'], dataset['output_dir'])

        # Run segmentation
        success = segmenter.run_segmentation()

        if success:
            print(f"\n{dataset['name']} segmentation completed!")
            print(f"Binary masks saved in: {dataset['output_dir']}")
        else:
            print(f"{dataset['name']} segmentation failed!")
            total_success = False

    print("\n" + "=" * 50)
    if total_success:
        print("All threshold segmentations completed successfully!")
        return 0
    else:
        print("Some threshold segmentations failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)