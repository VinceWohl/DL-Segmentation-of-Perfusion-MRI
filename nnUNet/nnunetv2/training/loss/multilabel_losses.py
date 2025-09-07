# nnunetv2/training/loss/multilabel_losses.py
from __future__ import annotations

from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_per_channel(x: torch.Tensor) -> torch.Tensor:
    """
    (N, C, ...) -> (C, N * prod(spatial))
    Keeps channels separate so we compute Dice per channel for multilabel.
    """
    c = x.shape[1]
    return x.reshape(x.shape[0], c, -1).permute(1, 0, 2).reshape(c, -1)


class SoftDiceLossMultiLabel(nn.Module):
    """
    Soft Dice over sigmoid probabilities per channel (multilabel).
    Targets are expected as {0,1} per channel (float/bool/int all okay).
    """
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = float(smooth)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor | list | tuple) -> torch.Tensor:
        # Be robust if DS snuck in
        if isinstance(targets, (list, tuple)):
            targets = targets[0]

        # Shape sanity
        if logits.ndim < 3:
            raise RuntimeError(f"Expected logits with shape (N,C,...) but got {tuple(logits.shape)}")
        if targets.ndim != logits.ndim:
            raise RuntimeError(f"Logits ndims {logits.ndim} != targets ndims {targets.ndim}. "
                               f"Make sure targets are converted to multilabel with a channel dim.")
        if logits.shape[0] != targets.shape[0] or logits.shape[1] != targets.shape[1]:
            raise RuntimeError(f"Channel/batch mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}. "
                               f"Ensure targets have the same (N,C,...) as logits.")

        probs = torch.sigmoid(logits)
        probs_f = _flatten_per_channel(probs)
        targ_f  = _flatten_per_channel(targets.float())

        intersect = (probs_f * targ_f).sum(-1)
        denom = probs_f.sum(-1) + targ_f.sum(-1)
        dice = (2.0 * intersect + self.smooth) / (denom + self.smooth)

        loss = 1.0 - dice
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BCEDiceLossMultiLabel(nn.Module):
    """
    BCEWithLogits + SoftDice for multilabel segmentation.
    Accepts optional pos_weight (tensor or list of floats) for BCE.
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor | Sequence[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.bce_w = float(bce_weight)
        self.dice_w = float(dice_weight)
        self.reduction = reduction
        self.dice = SoftDiceLossMultiLabel(reduction=reduction)

        # Register pos_weight as a buffer so it moves with .to(device)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)  # type: ignore[arg-type]
        else:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pw)  # type: ignore[arg-type]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor | list | tuple) -> torch.Tensor:
        # Be robust if DS snuck in
        if isinstance(targets, (list, tuple)):
            targets = targets[0]

        # Shape sanity (same checks as in Dice)
        if logits.ndim < 3:
            raise RuntimeError(f"Expected logits with shape (N,C,...) but got {tuple(logits.shape)}")
        if targets.ndim != logits.ndim:
            raise RuntimeError(f"Logits ndims {logits.ndim} != targets ndims {targets.ndim}. "
                               f"Make sure targets are converted to multilabel with a channel dim.")
        if logits.shape[0] != targets.shape[0] or logits.shape[1] != targets.shape[1]:
            raise RuntimeError(f"Channel/batch mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}. "
                               f"Ensure targets have the same (N,C,...) as logits.")

        # BCE over logits directly
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )

        # Soft Dice over sigmoid(logits)
        dice = self.dice(logits, targets)

        return self.bce_w * bce + self.dice_w * dice
