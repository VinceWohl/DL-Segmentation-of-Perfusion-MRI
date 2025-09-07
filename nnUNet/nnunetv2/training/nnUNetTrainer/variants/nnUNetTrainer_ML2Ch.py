from __future__ import annotations
from typing import Dict, Any, Tuple
import os, json
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.multilabel_losses import BCEDiceLossMultiLabel
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.inference.export_prediction_multilabel import export_multilabel_pred


class nnUNetTrainer_ML2Ch(nnUNetTrainer):
    """
    Two-channel multilabel trainer (left/right hemispheres):
      - Network outputs 2 logits channels (sigmoid).
      - Targets may be (N,2,...) binary OR a single int mask with {0,1,2,3}.
      - Overlap allowed (label 3 -> both channels active).
    """

    # ---------- IMPORTANT: keep the same signature as base ----------
    def __init__(self,
                 plans,
                 configuration: str,
                 fold: int | str,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # 30 epochs for your quick sanity check run
        self.num_epochs = 30
        self.validate_every = 1
        self.save_every = max(1, self.num_epochs // 5)  # ~5 checkpoints
        # optional gradient clipping (uncomment if needed)
        # self.gradient_clipping = 12.0

    # ---------- Force 2 output channels and disable deep supervision ----------
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Tuple[str, ...],
                                   num_input_channels: int,
                                   num_output_channels: int,   # ignored on purpose
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # Force 2 outputs (L/R) and no deep supervision (single tensor output)
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            2,  # <---- our multilabel two channels
            allow_init=True,
            deep_supervision=False  # <---- disable DS so loss sees a single tensor
        )

    # ---------- Use our multilabel loss ----------
    def _build_loss(self):
        # Optionally use a class-wise pos_weight if one side is rarer:
        # pos_weight = torch.tensor([1.0, 1.3], device=self.device)
        pos_weight = None
        loss = BCEDiceLossMultiLabel(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)
        return loss

    # ---------- Utilities ----------
    @staticmethod
    def _to_multilabel(target: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
          - (N,2,...) binary → return as float
          - (N,1,...) or (N,...) int mask with {0,1,2,3} → split to 2 channels
        """
        if target.ndim >= 2 and target.shape[1] == 2:
            return target.float()

        if target.ndim >= 2 and target.shape[1] == 1:
            tgt = target[:, 0]
        else:
            tgt = target

        ch_left  = (tgt == 1) | (tgt == 3)
        ch_right = (tgt == 2) | (tgt == 3)
        return torch.stack([ch_left, ch_right], dim=1).float()

    # ---------- One training iteration ----------
    def run_training_iteration(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        x = data['data'].to(self.device, non_blocking=True)
        y = data['target'].to(self.device, non_blocking=True)
        y = self._to_multilabel(y)  # (N, 2, ...)

        ctx = autocast if self.fp16 else dummy_context
        with ctx():
            logits = self.network(x)           # (N, 2, ...)
            if isinstance(logits, (list, tuple)):
                # we forced deep supervision off, but be safe:
                logits = logits[0]
            loss = self.loss(logits, y)

        if self.fp16:
            self.grad_scaler.scale(loss).backward()
            if self.gradient_clipping is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
            self.optimizer.step()

        # return a plain float for logging
        return {"loss": float(loss.detach().cpu().item())}

    # ---------- Validation & export (.nii) ----------
    @torch.no_grad()
    def validate(self,
                 do_mirroring: bool = True,
                 use_gaussian: bool = True,
                 tiled: bool = True) -> Dict[str, Any]:
        self.network.eval()
        outdir = os.path.join(self.output_folder, 'validation_ml')
        os.makedirs(outdir, exist_ok=True)

        pred_thresh = 0.45
        stats = {"n": 0, "dice_L": 0.0, "dice_R": 0.0}
        per_case = []

        # nnU-Net uses self.dataloader_val
        for batch in self.dataloader_val:
            x = batch['data'].to(self.device, non_blocking=True)
            y = batch['target'].to(self.device, non_blocking=True)
            y_ml = self._to_multilabel(y)  # (N, 2, ...)

            with (autocast() if self.fp16 else dummy_context()):
                logits = self.network(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = torch.sigmoid(logits)

            pred = (probs >= pred_thresh).float()

            # dice per channel
            y_f = y_ml.reshape(y_ml.shape[0], y_ml.shape[1], -1)
            p_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
            inter = (y_f * p_f).sum(-1)
            denom = torch.clamp(y_f.sum(-1) + p_f.sum(-1), min=1.0)
            dice = (2.0 * inter) / denom  # (N, 2)

            # export each case as .nii (2ch stack + separate L/R)
            props_list = batch.get('properties', [None] * pred.shape[0])
            for i in range(pred.shape[0]):
                props = props_list[i]
                paths = export_multilabel_pred(
                    pred[i].cpu().numpy(), props, out_dir=outdir
                )  # ensure this writes .nii (not .nii.gz)
                dL = float(dice[i, 0].cpu().item())
                dR = float(dice[i, 1].cpu().item())
                per_case.append({
                    "case": paths.get("case_id", f"case_{len(per_case)}"),
                    "dice_left": dL,
                    "dice_right": dR,
                    "pred_2ch": paths.get("2ch"),
                    "pred_left": paths.get("left"),
                    "pred_right": paths.get("right"),
                })

            stats["dice_L"] += float(dice[:, 0].mean().cpu().item())
            stats["dice_R"] += float(dice[:, 1].mean().cpu().item())
            stats["n"] += 1

        if stats["n"] > 0:
            stats["dice_L"] /= stats["n"]
            stats["dice_R"] /= stats["n"]
        stats["dice_mean"] = 0.5 * (stats["dice_L"] + stats["dice_R"])

        self.print_to_log_file(
            f"[ML] Dice -> left={stats['dice_L']:.4f}  right={stats['dice_R']:.4f}  mean={stats['dice_mean']:.4f}"
        )
        with open(os.path.join(outdir, "summary_ml.json"), "w") as f:
            json.dump({"summary": stats, "metric_per_case": per_case}, f, indent=2)

        return stats

    # ---------- drive 'best' tracking by our ML Dice ----------
    def _on_epoch_end_do_validation(self):
        results = self.validate()
        current = float(results["dice_mean"])
        # nnU-Net provides maybe_update_best_ema; fall back if name differs
        if hasattr(self, "maybe_update_best_ema"):
            self.maybe_update_best_ema(current)
        elif hasattr(self, "_maybe_update_best_ema"):
            self._maybe_update_best_ema(current)
        return current
