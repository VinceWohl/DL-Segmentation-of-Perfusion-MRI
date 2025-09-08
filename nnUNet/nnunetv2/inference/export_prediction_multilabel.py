from __future__ import annotations
import os
from typing import Dict, Optional
import numpy as np
import nibabel as nib

def _make_like(props: dict) -> nib.Nifti1Image:
    # Build an identity image using props' affine/shape as template
    # props follow nnU-Net conventions (see export_prediction.py)
    # We default to identity affine if not present.
    affine = props.get('sitk_stuff', {}).get('affine', None)
    if affine is None:
        affine = np.eye(4, dtype=float)
    # Return a dummy to harvest header when needed
    return nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.float32), affine)

def _save(path: str, arr: np.ndarray, props: Optional[dict]) -> None:
    # arr is 3D or 4D numpy; props can carry affine
    if props is not None:
        affine = props.get('sitk_stuff', {}).get('affine', None)
        if affine is None:
            affine = np.eye(4, dtype=float)
    else:
        affine = np.eye(4, dtype=float)
    img = nib.Nifti1Image(arr.astype(np.float32), affine)
    nib.save(img, path)

def export_multilabel_pred(
    pred_2ch: np.ndarray,
    props: Optional[dict],
    out_dir: str,
    case_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Parameters
    ----------
    pred_2ch : np.ndarray
        Shape (2, Z, Y, X) binary prediction after thresholding (0/1).
    props : dict or None
        nnU-Net case properties from the dataloader (for affine/origin).
    out_dir : str
        Output directory.
    case_id : str or None
        Used for filenames. If None, derive from props if possible.

    Returns
    -------
    dict with file paths { "2ch", "left", "right", "case_id" }.
    """
    os.makedirs(out_dir, exist_ok=True)
    assert pred_2ch.ndim == 4 and pred_2ch.shape[0] == 2, "expected (2, Z, Y, X)"

    if case_id is None:
        case_id = props.get('case_identifier', 'case') if isinstance(props, dict) else 'case'

    base = os.path.join(out_dir, case_id)

    # Save 4D two-channel NIfTI (C as 4th dim to play nice with most viewers)
    twoch_path = f"{base}_pred2ch.nii"
    _save(np.transpose(pred_2ch, (1, 2, 3, 0)), props, twoch_path)

    left_path = f"{base}_pred_left.nii"
    right_path = f"{base}_pred_right.nii"
    _save(pred_2ch[0], props, left_path)
    _save(pred_2ch[1], props, right_path)

    return {
        "2ch": twoch_path,
        "left": left_path,
        "right": right_path,
        "case_id": case_id
    }
