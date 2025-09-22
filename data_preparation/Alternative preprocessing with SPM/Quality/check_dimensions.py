#!/usr/bin/env python3
"""
Task-AIR Dimensions Check Script

This script focuses specifically on task-AIR folder NIfTI files and extracts:
- Image dimensions (number of voxels) in x, y, z directions
- Intensity statistics (min and max values)
- Zero-only slice IDs for quality assurance

The script scans through the folder structure:
- DATA_HC and DATA_patients
- Each visit (First_visit, Second_visit, Third_visit)
- Each subject (sub-p001 to sub-p015 for HC, sub-p016 to sub-p023 for patients)
- Only task-AIR folder files:
  * T1w_coreg files
  * FLAIR_coreg files
  * CBF_nativeSpace files
  * PerfTerrMask files

Output: task_air_dimensions.xlsx with dimensions and zero-slice information

Author: Generated for data quality assurance
"""

import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TaskAirAnalyzer:
    def __init__(self, data_root_path, output_excel_path):
        self.data_root = Path(data_root_path)
        self.output_excel = Path(output_excel_path)
        
        # Define subject ranges
        self.hc_subjects = [f"sub-p{i:03d}" for i in range(1, 16)]  # p001-p015
        self.patient_subjects = [f"sub-p{i:03d}" for i in range(16, 24)]  # p016-p023
        self.hc_visits = ["First_visit", "Second_visit", "Third_visit"]
        self.patient_visits = ["First_visit", "Second_visit"]
        
        # Results storage
        self.results = []
    
    def get_nii_info(self, nii_file_path):
        """Get image dimensions and intensity statistics from a NIfTI file"""
        try:
            nii = nib.load(nii_file_path)
            data = nii.get_fdata()
            
            # Handle different dimensionalities
            if len(data.shape) == 3:
                # 3D data (x, y, z)
                dim_x, dim_y, dim_z = data.shape[0], data.shape[1], data.shape[2]
            elif len(data.shape) == 4:
                # 4D data (x, y, z, t) - return x, y, z dimensions
                dim_x, dim_y, dim_z = data.shape[0], data.shape[1], data.shape[2]
            else:
                print(f"Warning: Unexpected dimensions for {nii_file_path}: {data.shape}")
                return None, None, None, None, None
            
            # Calculate intensity statistics
            min_intensity = float(np.min(data))
            max_intensity = float(np.max(data))
            
            return dim_x, dim_y, dim_z, min_intensity, max_intensity
                
        except Exception as e:
            print(f"Error loading {nii_file_path}: {e}")
            return None, None, None, None, None
    
    def find_zero_slices(self, nii_file_path):
        """Find slices that contain only zeros"""
        try:
            nii = nib.load(nii_file_path)
            data = nii.get_fdata()
            
            # Handle different dimensionalities
            if len(data.shape) == 3:
                # 3D data (x, y, z)
                total_slices = data.shape[2]
            elif len(data.shape) == 4:
                # 4D data (x, y, z, t) - use z dimension
                total_slices = data.shape[2]
            else:
                print(f"Warning: Unexpected dimensions for {nii_file_path}: {data.shape}")
                return None
            
            zero_only_slices = []
            
            # Check each slice for non-zero values
            for z in range(total_slices):
                if len(data.shape) == 3:
                    slice_data = data[:, :, z]
                else:
                    slice_data = data[:, :, z, 0]  # Take first time point for 4D
                
                if not np.any(slice_data > 0):
                    zero_only_slices.append(z + 1)  # 1-based indexing
            
            return zero_only_slices
            
        except Exception as e:
            print(f"Error analyzing slices in {nii_file_path}: {e}")
            return None
    
    def find_task_air_files(self, subject_path):
        """Find all .nii files specifically in task-AIR folders and subfolders"""
        nii_files = []
        if not subject_path.exists():
            return nii_files
        
        # Look for task-AIR folder
        task_air_path = subject_path / "task-AIR"
        if not task_air_path.exists():
            return nii_files
        
        # Recursively find all .nii files in task-AIR folder and subfolders
        for root, dirs, files in os.walk(task_air_path):
            for file in files:
                if file.endswith('.nii'):
                    full_path = Path(root) / file
                    # Get relative path from subject directory for cleaner output
                    rel_path = full_path.relative_to(subject_path)
                    nii_files.append((full_path, rel_path))
        
        return nii_files
    
    def process_subject(self, group, subject, visit):
        """Process a single subject and find all task-AIR NIfTI files"""
        subject_path = self.data_root / group / visit / "output" / subject
        
        if not subject_path.exists():
            print(f"Warning: Subject path does not exist: {subject_path}")
            return 0
        
        print(f"Processing {group}_{subject}_{visit}...")
        
        # Find task-AIR .nii files only
        nii_files = self.find_task_air_files(subject_path)
        
        processed_count = 0
        for full_path, rel_path in nii_files:
            # Get image dimensions and intensity statistics
            dim_x, dim_y, dim_z, min_intensity, max_intensity = self.get_nii_info(full_path)
            
            if dim_x is not None:
                # Find zero-only slices
                zero_only_slices = self.find_zero_slices(full_path)
                
                # Store result
                self.results.append({
                    'Group': group,
                    'Subject': subject,
                    'Visit': visit,
                    'File_Path': str(rel_path),
                    'Full_Path': str(full_path),
                    'Dimensions_X': dim_x,
                    'Dimensions_Y': dim_y,
                    'Dimensions_Z': dim_z,
                    'Min_Intensity': min_intensity,
                    'Max_Intensity': max_intensity,
                    'Zero_Only_Slice_IDs': ','.join(map(str, zero_only_slices)) if zero_only_slices else ''
                })
                
                processed_count += 1
                
                if zero_only_slices is not None:
                    print(f"  {rel_path}: dimensions {dim_x}x{dim_y}x{dim_z}, intensity range [{min_intensity:.2f}, {max_intensity:.2f}]")
                    if zero_only_slices:
                        print(f"    Zero-only slices: {zero_only_slices}")
                    else:
                        print(f"    All slices contain data")
                else:
                    print(f"  {rel_path}: dimensions {dim_x}x{dim_y}x{dim_z}, intensity range [{min_intensity:.2f}, {max_intensity:.2f}] (slice analysis failed)")
            else:
                print(f"  {rel_path}: ERROR - Could not determine dimensions")
        
        return processed_count
    
    def run_analysis(self):
        """Run the complete task-AIR file analysis"""
        print(f"Task-AIR Dimensions and Segmentation Analysis")
        print("=" * 60)
        print(f"Data root: {self.data_root}")
        print(f"Output Excel: {self.output_excel}")
        print()
        
        if not NIBABEL_AVAILABLE:
            print("ERROR: nibabel package is required. Please install with: pip install nibabel")
            return False
        
        if not PANDAS_AVAILABLE:
            print("ERROR: pandas package is required. Please install with: pip install pandas openpyxl")
            return False
        
        if not self.data_root.exists():
            print(f"ERROR: Data root directory does not exist: {self.data_root}")
            return False
        
        total_files = 0
        
        # Process HC subjects
        print("Processing DATA_HC subjects...")
        for subject in self.hc_subjects:
            for visit in self.hc_visits:
                total_files += self.process_subject("DATA_HC", subject, visit)
        
        # Process Patient subjects
        print("\nProcessing DATA_patients subjects...")
        for subject in self.patient_subjects:
            for visit in self.patient_visits:
                total_files += self.process_subject("DATA_patients", subject, visit)
        
        # Write results to Excel
        self.write_excel()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"Total task-AIR NIfTI files processed: {total_files}")
        print(f"Results saved to: {self.output_excel}")
        
        return True
    
    def write_excel(self):
        """Write results to Excel file"""
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            # Define column order
            column_order = ['Group', 'Subject', 'Visit', 'File_Path', 'Full_Path',
                           'Dimensions_X', 'Dimensions_Y', 'Dimensions_Z', 
                           'Min_Intensity', 'Max_Intensity', 'Zero_Only_Slice_IDs']
            
            # Reorder columns
            df = df[column_order]
            
            # Write to Excel file
            with pd.ExcelWriter(self.output_excel, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Task_AIR_Analysis', index=False)
                
                # Get the workbook and worksheet objects
                worksheet = writer.sheets['Task_AIR_Analysis']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
            
            print(f"Excel file written successfully: {self.output_excel}")
            
        except Exception as e:
            print(f"Error writing Excel file: {e}")


def main():
    """Main function to run the task-AIR analysis"""
    # Default paths
    default_data_path = r"C:\Users\Vincent Wohlfarth\Data\anon_Data_250808"
    
    # Create timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_path = Path(__file__).parent / f"task_air_dimensions_{timestamp}.xlsx"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_data_path
    
    if len(sys.argv) > 2:
        output_excel = sys.argv[2]
    else:
        output_excel = default_output_path
    
    print("Task-AIR Dimensions and Segmentation Analysis Script")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TaskAirAnalyzer(data_path, output_excel)
    
    # Run analysis
    success = analyzer.run_analysis()
    
    if success:
        print(f"\nTask-AIR analysis completed!")
        print(f"Excel file saved at: {output_excel}")
    else:
        print("Task-AIR analysis failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)