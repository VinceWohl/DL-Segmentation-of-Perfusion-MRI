# author: VW
#
# This file requires the freesurfer toolbox to be installed containing the mri_defacing tool.
#____________________________________________________________________________________________________________________________________________

import sys
import subprocess
from pathlib import Path


# root directory for input MRI files
root_dir = Path("/mnt/c/Users") / "Vincent Wohlfarth" / "Data" / "ssASL-Project-anonymisation"
if not root_dir.exists():
    print(f"❌ Verzeichnis nicht gefunden: {root_dir}")
    sys.exit(1)

# path to GCA files
fshome = Path("/usr/local/freesurfer/8.0.0")
gca1 = fshome / "average" / "talairach_mixed_with_skull.gca"
gca2 = fshome / "average" / "face.gca"

# which files should get defaced
keywords = ["T1w", "FLAIR"]

# iterate over all .nii files
for file in root_dir.rglob("*.nii"):

    # skip files that don't match keywords or are already defaced
    if not any(key in file.name for key in keywords):
        continue
    if "_defaced" in file.stem:
        continue
    if "_pydefaced" in file.stem:
        continue
    if "anon_" in file.stem:
        continue

    # define output file path
    output_file = file.with_name(file.stem + "_defaced.nii")

    # skip if defaced file already exists
    if output_file.exists():
        print(f"✔️  Already defaced: {output_file.name}")
        continue

    # run the mri_deface from freesurfer
    print(f"🔄 Defacing: {file}")
    
    try:
        subprocess.run([
            "mri_deface",
            str(file),
            str(gca1),
            str(gca2),
            str(output_file)
        ], check=True)
        print(f"✅ Success: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {file}")
        sys.exit(1)