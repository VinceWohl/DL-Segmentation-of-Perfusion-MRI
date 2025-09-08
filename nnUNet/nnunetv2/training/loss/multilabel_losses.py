from __future__ import annotations
import torch
import torch.nn as nn

def _flatten_per_channel(x: torch.Tensor) -> torch.Tensor:
    # (N, C, ...) -> (C, N * prod(spatial))
    c = x.shape[1]
    return x.reshape(x.shape[0], c, -1).permute(1, 0, 2).reshape(c, -1)

def _to_2ch(target: torch.Tensor) -> torch.Tensor:
    """
    Accept either:
      - (N,2,...) binary → return as float
      - (N,1,...) or (N,...) int mask with {0,1,2,3} → split to 2 channels
      - if a list/tuple is passed (DS on), take first element
    """
    if isinstance(target, (list, tuple)):
        target = target[0]
    if target.ndim >= 2 and target.shape[1] == 2:
        return target.float()
    if target.ndim >= 2 and target.shape[1] == 1:
        tgt = target[:, 0]
    else:
        tgt = target
    ch_left  = (tgt == 1) | (tgt == 3)
    ch_right = (tgt == 2) | (tgt == 3)
    return torch.stack([ch_left, ch_right], dim=1).float()

class SoftDiceLossMultiLabel(nn.Module):
    """
    Soft Dice over sigmoid probabilities per channel (multilabel).
    Targets are expected as {0,1} per channel (we convert if needed).
    """
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targ = _to_2ch(targets)

        probs_f = _flatten_per_channel(probs)
        targ_f  = _flatten_per_channel(targ)

        intersect = (probs_f * targ_f).sum(-1)
        denom = probs_f.sum(-1) + targ_f.sum(-1)
        dice = (2. * intersect + self.smooth) / (denom + self.smooth)
        loss = 1. - dice
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class BCEDiceLossMultiLabel(nn.Module):
    """
    BCEWithLogits + SoftDice for multilabel segmentation.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = SoftDiceLossMultiLabel()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targ = _to_2ch(targets)
        return self.bce_w * self.bce(logits, targ) + self.dice_w * self.dice(logits, targ)
