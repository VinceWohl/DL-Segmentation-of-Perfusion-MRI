# author: VW
#
#____________________________________________________________________________________________________________________________________________

import sys
import subprocess
from pathlib import Path


# root directory for input MRI files
root_dir = Path.home() / "data" / "ssASL-Project-anonymisation"
# root_dir = Path("/mnt/c/Users") / "Vincent Wohlfarth" / "Data" / "ssASL-Project-anonymisation"
if not root_dir.exists():
    print(f"❌ Verzeichnis nicht gefunden: {root_dir}")
    sys.exit(1)

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
    output_file = file.with_name(file.stem + "_pydefaced.nii")

    # skip if pydefaced file already exists
    if output_file.exists():
        print(f"✔️  Already pydefaced: {output_file.name}")
        continue

    # run pydeface
    print(f"🔄 Pydefacing: {file}")

    try:
        subprocess.run([
        sys.executable,
        "-m", "pydeface",
        str(file),
        "--outfile", str(output_file)
        ], check=True)
        print(f"✅ Success: {output_file.name}")
    except subprocess.CalledProcessError:
        print(f"❌ Error processing {file.name}")
        sys.exit(1)