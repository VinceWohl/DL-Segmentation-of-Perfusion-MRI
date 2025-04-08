# author: VW
# compares NIfTI headers
#____________________________________________________________________________________________________________________________________________

import nibabel as nib
from pathlib import Path

# Original reference file
reference_path = Path.home() / "projects" / "mri-defacing" / "input" / "ssASL-Project-anonymisation" / "Sample_Data" / "Processing_mqBOLD" / "output" / \
                            "sub-p007" / "T1w" / "sub-p007_T1w.nii"

# Files to compare
compare_paths = [
    reference_path.with_name("anon_sub-p007_T1w.nii"),
    reference_path.with_name("sub-p007_T1w_defaced.nii"),
    reference_path.with_name("sub-p007_T1w_pydefaced.nii"),
]

# Load reference header
ref_img = nib.load(reference_path)
ref_header = ref_img.header

# Compare headers
for path in compare_paths:
    if not path.exists():
        print(f"❌ File not found: {path}")
        continue

    img = nib.load(path)
    header = img.header

    if ref_header == header:
        print(f"✅ Header matches: {path.name}")
    else:
        print(f"❌ Header differs: {path.name}")

        #Optional: Uncomment to see the differences
        for key in ref_header:
            if not (ref_header[key] == header[key]).all():
                print(f" - {key}: ref={ref_header[key]} | other={header[key]}")