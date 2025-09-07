from __future__ import annotations
import os
import numpy as np
import nibabel as nib

def _case_id_from_props(props: dict | None) -> str:
    if props is None:
        return "pred"
    cid = props.get('case_identifier', None)
    if cid is None:
        ldf = props.get('list_of_data_files', None)
        if ldf:
            cid = os.path.basename(ldf[0]).split('_0000')[0]
    return cid or "pred"

def _affine_from_props(props: dict | None):
    if props is not None and 'nifti_affine' in props:
        return props['nifti_affine']
    return np.eye(4, dtype=float)

def export_multilabel_pred(pred_bin_czyx: np.ndarray,
                           props: dict | None,
                           out_dir: str) -> dict:
    """
    pred_bin_czyx: (2, Z, Y, X) or (2, H, W) with values {0,1}
    Writes:
      - <case>_2ch.nii (4D channels-last)
      - <case>_L.nii   (3D left)
      - <case>_R.nii   (3D right)
    """
    os.makedirs(out_dir, exist_ok=True)

    arr = pred_bin_czyx
    if arr.ndim == 3:  # (2, H, W) -> (2, 1, H, W)
        arr = arr[:, None, ...]
    assert arr.ndim == 4 and arr.shape[0] == 2, "Expect (2, Z, Y, X) or (2, H, W)"

    case_id = _case_id_from_props(props)
    A = _affine_from_props(props)

    # channels-last for easier viewing: (Z, Y, X, 2)
    arr_last = np.moveaxis(arr, 0, -1).astype(np.uint8)
    p_2ch = os.path.join(out_dir, f"{case_id}_2ch.nii")
    nib.save(nib.Nifti1Image(arr_last, A), p_2ch)

    # per-channel 3D
    p_L = os.path.join(out_dir, f"{case_id}_L.nii")
    p_R = os.path.join(out_dir, f"{case_id}_R.nii")
    nib.save(nib.Nifti1Image(arr[0].astype(np.uint8), A), p_L)
    nib.save(nib.Nifti1Image(arr[1].astype(np.uint8), A), p_R)

    return {"case_id": case_id, "2ch": p_2ch, "left": p_L, "right": p_R}