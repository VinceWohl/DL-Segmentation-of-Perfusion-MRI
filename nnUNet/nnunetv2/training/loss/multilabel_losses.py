from __future__ import annotations
import torch
import torch.nn as nn


def _flatten_per_channel(x: torch.Tensor) -> torch.Tensor:
    """
    (N, C, ...) -> (C, N * prod(spatial)) for channelwise Dice stats.
    """
    c = x.shape[1]
    return x.reshape(x.shape[0], c, -1).permute(1, 0, 2).reshape(c, -1)


def _to_2ch(target: torch.Tensor) -> torch.Tensor:
    """
    Convert targets to 2-channel (left, right) binary maps:
      - If already (N,2,...) -> float
      - If (N,1,...) or (N,...) with integer classes:
            left  := label in {1,3}
            right := label in {2,3}
    """
    if isinstance(target, (list, tuple)):      # robustness if something upstream passes [target]
        target = target[0]

    if target.ndim >= 2 and target.shape[1] == 2:
        return target.float()

    if target.ndim >= 2 and target.shape[1] == 1:
        t = target[:, 0]
    else:
        t = target

    left = (t == 1) | (t == 3)
    right = (t == 2) | (t == 3)
    return torch.stack([left, right], dim=1).float()


class SoftDiceLossMultiLabel(nn.Module):
    """Soft Dice over sigmoid probabilities, per-channel averaged."""
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
    Robust to accidentally receiving [target] (unwraps) and asserts single logits tensor.
    Also auto-converts integer labelmaps -> 2 channels.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = SoftDiceLossMultiLabel()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        if isinstance(targets, (list, tuple)):
            targets = targets[0]
        if isinstance(logits, (list, tuple)):
            raise RuntimeError(
                "BCEDiceLossMultiLabel expects a single logits tensor. "
                "Disable deep supervision in the trainer (enable_deep_supervision=False)."
            )
        t2 = _to_2ch(targets.to(logits.dtype))
        if logits.shape[1] != 2:
            raise RuntimeError(
                f"Network outputs {logits.shape[1]} channels, expected 2. "
                f"Ensure determine_num_output_channels() returns 2."
            )
        return self.bce_w * self.bce(logits, t2) + self.dice_w * self.dice(logits, t2)
