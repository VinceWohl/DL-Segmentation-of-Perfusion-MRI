from __future__ import annotations
from typing import Dict, Any
import os, json, torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.multilabel_losses import BCEDiceLossMultiLabel
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.inference.export_prediction_multilabel import export_multilabel_pred


class nnUNetTrainer_ML2Ch(nnUNetTrainer):
    """
    Two-channel MULTILABEL trainer (left/right):
      - net outputs 2 logits with sigmoid
      - targets can be (N,2,...) binary *or* a single int mask with {0,1,2,3}
      - overlap allowed
    """

    # >>> IMPORTANT: mirror base signature; NO *args/**kwargs here! <<<
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # defaults we want
        self.enable_deep_supervision = False
        self.enable_online_evaluation = False
        self.num_epochs = 30
        self.validate_every = 1
        self.save_every = max(1, self.num_epochs // 5)

    def initialize(self):
        # make sure flags are set BEFORE network/transforms are built
        self.enable_deep_supervision = False
        self.enable_online_evaluation = False
        super().initialize()
        self.print_to_log_file("[ML2Ch] initialized (DS OFF, online eval disabled).")

    # force 2 outputs and DS OFF for the network, irrespective of label_manager
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # call the base staticmethod but override outputs & DS
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            2,                  # <- two multilabel channels
            False               # <- no deep supervision
        )

    def _build_loss(self):
        # set pos_weight per-channel here if needed for imbalance
        pos_weight = None  # e.g. torch.tensor([1.0, 1.3], device=self.device)
        return BCEDiceLossMultiLabel(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

    # ----- helpers -----
    @staticmethod
    def _to_multilabel(target: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
          - (N,2,...) binary → return as float
          - (N,1,...) or (N,...) int mask with {0,1,2,3} → split to 2 channels
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

    # ----- training step -----
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = batch['data'].to(self.device, non_blocking=True)
        y = batch['target']
        if isinstance(y, (list, tuple)):
            y = y[0]
        y = y.to(self.device, non_blocking=True)
        y = self._to_multilabel(y)

        ctx = autocast(self.device.type, enabled=(self.grad_scaler is not None)) if self.device.type in ('cuda', 'mps') else dummy_context
        with ctx:
            logits = self.network(x)                    # (N, 2, ...)
            loss = self.loss(logits, y)                # BCE + Dice (multilabel)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {"loss": float(loss.detach().cpu().item())}

    # ----- validation (our own, with multilabel export) -----
    @torch.no_grad()
    def validate(self, do_mirroring: bool = True, use_gaussian: bool = True, tiled: bool = True) -> Dict[str, Any]:
        self.network.eval()
        outdir = os.path.join(self.output_folder, 'validation_ml')
        os.makedirs(outdir, exist_ok=True)

        pred_thresh = 0.45
        meter = {"n": 0, "dice_L": 0.0, "dice_R": 0.0}
        per_case = []

        for batch in self.val_data_loader:
            x = batch['data'].to(self.device, non_blocking=True)
            y = batch['target']
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.to(self.device, non_blocking=True)
            y_ml = self._to_multilabel(y)  # (N, 2, ...)

            ctx = autocast(self.device.type, enabled=(self.grad_scaler is not None)) if self.device.type in ('cuda', 'mps') else dummy_context
            with ctx:
                logits = self.network(x)
                probs = torch.sigmoid(logits)

            pred = (probs >= pred_thresh).float()

            # per-channel Dice on binarized predictions
            y_f = y_ml.reshape(y_ml.shape[0], y_ml.shape[1], -1)
            p_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
            inter = (y_f * p_f).sum(-1)
            denom = y_f.sum(-1) + p_f.sum(-1)
            dice = (2.0 * inter) / torch.clamp(denom, min=1.0)  # (N,2)

            # export
            props_list = batch.get('properties', [None] * pred.shape[0])
            for i in range(pred.shape[0]):
                props = props_list[i]
                paths = export_multilabel_pred(pred[i].cpu().numpy(), props, out_dir=outdir)
                dL = float(dice[i, 0].cpu().item())
                dR = float(dice[i, 1].cpu().item())
                per_case.append({
                    "case": paths.get("case_id", f"case_{len(per_case)}"),
                    "dice_left": dL,
                    "dice_right": dR,
                    "pred_2ch": paths["2ch"],
                    "pred_left": paths["left"],
                    "pred_right": paths["right"],
                })

            meter["dice_L"] += float(dice[:, 0].mean().cpu().item())
            meter["dice_R"] += float(dice[:, 1].mean().cpu().item())
            meter["n"] += 1

        if meter["n"] > 0:
            meter["dice_L"] /= meter["n"]
            meter["dice_R"] /= meter["n"]
            meter["dice_mean"] = 0.5 * (meter["dice_L"] + meter["dice_R"])
        else:
            meter["dice_mean"] = 0.0

        self.print_to_log_file(
            f"ML Dice -> left={meter['dice_L']:.4f}, right={meter['dice_R']:.4f}, mean={meter['dice_mean']:.4f}"
        )
        with open(os.path.join(outdir, "summary_ml.json"), "w") as f:
            json.dump({"summary": meter, "metric_per_case": per_case}, f, indent=2)

        return meter

    # make the built-in “actual validation” call our multilabel validator
    def perform_actual_validation(self, export_validation_probabilities: bool = False) -> None:
        # Avoid the standard predictor pipeline (expects num_segmentation_heads)
        _ = self.validate()
        # nothing to return; just to satisfy the base call

    # have the 'best' tracking use our ML dice
    def _on_epoch_end_do_validation(self):
        meter = self.validate()
        current = float(meter["dice_mean"])
        if hasattr(self, "_maybe_update_best_ema"):
            self._maybe_update_best_ema(current)
        elif hasattr(self, "maybe_update_best_ema"):
            self.maybe_update_best_ema(current)
        return current
