# nnunetv2/inference/export_prediction_multilabel.py
from __future__ import annotations
import os
import numpy as np
import nibabel as nib

def _infer_case_output_path(props: dict | None, output_dir: str) -> str:
    if props is not None:
        case_id = props.get('case_identifier', None)
        if case_id is None:
            ldf = props.get('list_of_data_files', None)
            if ldf:
                case_id = os.path.basename(ldf[0]).split('_0000')[0]
        if case_id is None:
            case_id = "pred"
    else:
        case_id = "pred"
    return os.path.join(output_dir, f"{case_id}.nii.gz")

def export_multilabel_pred(pred_binary_czyx: np.ndarray,
                           props: dict | None,
                           output_file: str | None = None,
                           output_file_suffix: str = "") -> str:
    """
    Save (C, Z, Y, X) or (C, H, W) binary as 4D NIfTI (channels-first).
    If 2D, a singleton Z-dim is added.
    """
    arr = pred_binary_czyx
    if arr.ndim == 3:  # (C, H, W) -> (C, 1, H, W)
        arr = arr[:, None, ...]
    assert arr.ndim == 4, "Expected (C, Z, Y, X) or (C, H, W)."
    arr = arr.astype(np.uint8)

    affine = None
    if props is not None and 'nifti_affine' in props:
        affine = props['nifti_affine']
    if affine is None:
        affine = np.eye(4, dtype=float)

    img = nib.Nifti1Image(arr, affine)
    if output_file is None or os.path.isdir(output_file):
        out_dir = output_file if output_file is not None else "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = _infer_case_output_path(props, out_dir)
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        base, ext = os.path.splitext(output_file)
        if base.endswith(".nii"):
            base, _ = os.path.splitext(base)
        out_path = base + output_file_suffix + ".nii.gz"

    nib.save(img, out_path)
    return out_path