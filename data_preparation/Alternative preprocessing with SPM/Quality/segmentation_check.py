#!/usr/bin/env python3
"""
Final Quality Check Script for Preprocessing

This script creates comprehensive PNG visualizations for final quality assessment
of the preprocessed neuroimaging data.

For each case (Group, Subject, Visit, Hemisphere), it creates a PNG with:
- Upper row: T1w_coreg (grayscale) with CBF overlay (rainbow colormap)
- Lower row: CBF (grayscale) with perfusion territory mask overlay (red)

Both rows show all 14 slices after preprocessing.

Author: Generated for final data quality assurance
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


class FinalQualityChecker:
    def __init__(self, data_root_path, output_path):
        self.data_root = Path(data_root_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Define subject ranges
        self.hc_subjects = [f"sub-p{i:03d}" for i in range(1, 16)]  # p001-p015
        self.patient_subjects = [f"sub-p{i:03d}" for i in range(16, 24)]  # p016-p023
        self.hc_visits = ["First_visit", "Second_visit", "Third_visit"]
        self.patient_visits = ["First_visit", "Second_visit"]
        
        # Hemisphere definitions
        self.hemispheres = {
            'LICA': {
                't1w_path': 'task-AIR/ASL/ssLICA/T1w_coreg/anon_r{subject}_T1w.nii',
                'cbf_path': 'task-AIR/ASL/ssLICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii',
                'mask_path': 'task-AIR/ASL/ssLICA/PerfTerrMask/mask_LICA_manual_Corrected.nii',
                'name': 'Left ICA'
            },
            'RICA': {
                't1w_path': 'task-AIR/ASL/ssRICA/T1w_coreg/anon_r{subject}_T1w.nii',
                'cbf_path': 'task-AIR/ASL/ssRICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii',
                'mask_path': 'task-AIR/ASL/ssRICA/PerfTerrMask/mask_RICA_manual_Corrected.nii',
                'name': 'Right ICA'
            }
        }
    
    def load_nifti_file(self, file_path):
        """Load a NIfTI file and return the data array"""
        try:
            nii = nib.load(file_path)
            return nii.get_fdata()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def normalize_for_display(self, data, percentile_range=(1, 99)):
        """Normalize data for display, preserving zero values as black"""
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
    
    def create_dual_row_visualization(self, t1w_data, cbf_data, mask_data, title, output_file):
        """Create dual-row visualization: T1w+CBF (top), CBF+Mask (bottom)"""
        if t1w_data is None or cbf_data is None or mask_data is None:
            print(f"Warning: Missing data for {title}")
            return False
        
        # Ensure all data shapes match
        if not (t1w_data.shape == cbf_data.shape == mask_data.shape):
            print(f"Warning: Shape mismatch for {title}")
            print(f"  T1w: {t1w_data.shape}, CBF: {cbf_data.shape}, Mask: {mask_data.shape}")
            return False
        
        # Get dimensions (expecting 14 slices)
        nx, ny, nz = t1w_data.shape
        if nz != 14:
            print(f"Warning: Expected 14 slices, got {nz} for {title}")
        
        # Create figure with 2 rows for 14 slices each
        fig, axes = plt.subplots(2, 14, figsize=(28, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Normalize data for display
        t1w_norm = self.normalize_for_display(t1w_data, (2, 98))
        cbf_norm = self.normalize_for_display(cbf_data, (2, 98))
        
        # Process each slice
        for z in range(min(nz, 14)):  # Ensure we don't exceed 14 columns
            # Upper row: T1w (grayscale) + CBF (rainbow overlay)
            ax_top = axes[0, z]
            
            # Get slices and flip for correct orientation
            t1w_slice = np.flipud(t1w_norm[:, :, z])
            cbf_slice = np.flipud(cbf_norm[:, :, z])
            
            # Display T1w as grayscale background
            ax_top.imshow(t1w_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            
            # Overlay CBF with rainbow colormap where CBF > 0
            cbf_masked = np.ma.masked_where(cbf_slice <= 0, cbf_slice)
            ax_top.imshow(cbf_masked.T, cmap='jet', origin='lower', alpha=0.3, vmin=0, vmax=1)
            
            ax_top.set_title(f'Slice {z+1}', fontsize=10)
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            
            # Lower row: CBF (grayscale) + Mask (red overlay)
            ax_bottom = axes[1, z]
            
            # Get mask slice
            mask_slice = np.flipud(mask_data[:, :, z])
            
            # Display CBF as grayscale background
            ax_bottom.imshow(cbf_slice.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            
            # Overlay mask in red where mask > 0
            mask_overlay = np.zeros((*mask_slice.shape, 4))
            mask_overlay[:, :, 0] = 1.0  # Red channel
            mask_overlay[:, :, 3] = (mask_slice > 0) * 0.3  # Alpha for visibility
            ax_bottom.imshow(mask_overlay.transpose(1, 0, 2), origin='lower')
            
            ax_bottom.set_title(f'Slice {z+1}', fontsize=10)
            ax_bottom.set_xticks([])
            ax_bottom.set_yticks([])
        
        # Hide unused columns if less than 14 slices
        for z in range(nz, 14):
            axes[0, z].set_visible(False)
            axes[1, z].set_visible(False)
        
        # Add row labels
        axes[0, 0].set_ylabel('T1w + CBF\n(rainbow)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('CBF + Mask\n(red)', fontsize=12, fontweight='bold')
        
        # Add legends
        legend_elements = [
            patches.Patch(color='gray', label='T1w/CBF (grayscale)'),
            patches.Patch(color='red', label='CBF (rainbow)'),
            patches.Patch(color='red', alpha=0.5, label='Perfusion Mask (red)')
        ]
        fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    
    def process_case(self, group, subject, visit, hemisphere):
        """Process a single case (Group, Subject, Visit, Hemisphere)"""
        # Construct paths
        subject_path = self.data_root / group / visit / "output" / subject
        
        if not subject_path.exists():
            print(f"Warning: Subject path does not exist: {subject_path}")
            return False
        
        # Get file paths
        t1w_path_template = self.hemispheres[hemisphere]['t1w_path']
        t1w_file = subject_path / t1w_path_template.format(subject=subject)
        cbf_file = subject_path / self.hemispheres[hemisphere]['cbf_path']
        mask_file = subject_path / self.hemispheres[hemisphere]['mask_path']
        
        # Check if files exist
        missing_files = []
        if not t1w_file.exists():
            missing_files.append(f"T1w: {t1w_file}")
        if not cbf_file.exists():
            missing_files.append(f"CBF: {cbf_file}")
        if not mask_file.exists():
            missing_files.append(f"Mask: {mask_file}")
        
        if missing_files:
            print(f"Warning: Missing files for {group}_{subject}_{visit}_{hemisphere}:")
            for missing in missing_files:
                print(f"  {missing}")
            return False
        
        # Load data
        print(f"Processing {group}_{subject}_{visit}_{hemisphere}...")
        t1w_data = self.load_nifti_file(t1w_file)
        cbf_data = self.load_nifti_file(cbf_file)
        mask_data = self.load_nifti_file(mask_file)
        
        if t1w_data is None or cbf_data is None or mask_data is None:
            return False
        
        # Create output filename
        output_filename = f"{group}_{subject}_{visit}_{hemisphere}_quality_check.png"
        output_file = self.output_path / output_filename
        
        # Create visualization
        title = f"{group} - {subject} - {visit} - {self.hemispheres[hemisphere]['name']}"
        
        success = self.create_dual_row_visualization(t1w_data, cbf_data, mask_data, title, output_file)
        
        if success:
            print(f"  OK Saved: {output_filename}")
        else:
            print(f"  FAILED: {output_filename}")
        
        return success
    
    def run_quality_check(self):
        """Run the complete final quality check"""
        print(f"Final Quality Check - Preprocessing Assessment")
        print("=" * 60)
        print(f"Data root: {self.data_root}")
        print(f"Output directory: {self.output_path}")
        print()
        
        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        
        if not self.data_root.exists():
            print(f"ERROR: Data root directory does not exist: {self.data_root}")
            return False
        
        total_processed = 0
        total_successful = 0
        
        # Process HC subjects
        print("Processing DATA_HC subjects...")
        for subject in self.hc_subjects:
            for visit in self.hc_visits:
                for hemisphere in ['LICA', 'RICA']:
                    total_processed += 1
                    if self.process_case("DATA_HC", subject, visit, hemisphere):
                        total_successful += 1
        
        # Process Patient subjects
        print("\nProcessing DATA_patients subjects...")
        for subject in self.patient_subjects:
            for visit in self.patient_visits:
                for hemisphere in ['LICA', 'RICA']:
                    total_processed += 1
                    if self.process_case("DATA_patients", subject, visit, hemisphere):
                        total_successful += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("FINAL QUALITY CHECK SUMMARY:")
        print(f"Total cases processed: {total_processed}")
        print(f"Successful visualizations: {total_successful}")
        print(f"Failed cases: {total_processed - total_successful}")
        print(f"Success rate: {total_successful/total_processed*100:.1f}%")
        print(f"\nAll PNG files saved in: {self.output_path}")
        
        return True


def main():
    """Main function to run the final quality check"""
    # Default paths
    default_data_path = r"C:\Users\Vincent Wohlfarth\Data\anon_Data_250808"
    default_output_path = Path(__file__).parent / "final_quality_check"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_data_path
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = default_output_path
    
    print("Final Quality Check Script - Preprocessing Assessment")
    print("=" * 60)
    
    # Initialize checker
    checker = FinalQualityChecker(data_path, output_path)
    
    # Run quality check
    success = checker.run_quality_check()
    
    if success:
        print(f"\nFinal quality check completed successfully!")
        print(f"PNG files saved in: {output_path}")
        print(f"\nEach PNG contains:")
        print(f"  - Upper row: T1w (grayscale) + CBF (rainbow overlay) - 14 slices")
        print(f"  - Lower row: CBF (grayscale) + Perfusion mask (red overlay) - 14 slices")
    else:
        print("Final quality check failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)