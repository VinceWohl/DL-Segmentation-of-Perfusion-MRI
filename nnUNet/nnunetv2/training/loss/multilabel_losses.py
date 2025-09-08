from __future__ import annotations
import torch
import torch.nn as nn

def _flatten_per_channel(x: torch.Tensor) -> torch.Tensor:
    # (N, C, ...) -> (C, N * prod(spatial))
    c = x.shape[1]
    return x.reshape(x.shape[0], c, -1).permute(1, 0, 2).reshape(c, -1)

def _as_tensor(t):
    # Be robust if DS accidentally returns a list
    if isinstance(t, (list, tuple)):
        t = t[0]
    return t

class SoftDiceLossMultiLabel(nn.Module):
    """
    Soft Dice over sigmoid probabilities per channel (multilabel).
    Targets are expected as {0,1} per channel.
    """
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _as_tensor(targets).float()
        probs = torch.sigmoid(logits)
        probs_f = _flatten_per_channel(probs)
        targ_f  = _flatten_per_channel(targets)

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
        targets = _as_tensor(targets).float()
        return self.bce_w * self.bce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)
