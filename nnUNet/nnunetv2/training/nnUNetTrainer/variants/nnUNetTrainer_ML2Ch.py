from __future__ import annotations
from typing import Dict, Any, Optional
import os, json, shutil, torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.multilabel_losses import BCEDiceLossMultiLabel
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.inference.export_prediction_multilabel import export_multilabel_pred

# robust autocast context (torch & device agnostic)
try:
    from torch.cuda.amp import autocast as _autocast_cuda
except Exception:
    _autocast_cuda = None
try:
    from torch import autocast as _autocast_generic  # torch>=2.0
except Exception:
    _autocast_generic = None


def _mixed_precision_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return dummy_context()
    if device.type == 'cuda' and _autocast_cuda is not None:
        return _autocast_cuda(enabled=True)
    if _autocast_generic is not None:
        return _autocast_generic(device.type, enabled=True)
    return dummy_context()


class nnUNetTrainer_ML2Ch(nnUNetTrainer):
    """
    Multi-LABEL trainer with two independent output channels (left/right):
      - Network outputs 2 logits (sigmoid -> two binary masks).
      - Targets may be (N,2,...) binary or single {0,1,2,3} mask.
      - Exports ONE 4-D NIfTI (channels-last: ..., 2) per case AFTER training.
    """

    # mirror the base signature EXACTLY
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.enable_online_evaluation = False
        self.num_epochs = 30
        self.validate_every = 1
        self.save_every = max(1, self.num_epochs // 5)

    def initialize(self):
        self.enable_deep_supervision = False
        self.enable_online_evaluation = False
        super().initialize()
        self.print_to_log_file("[ML2Ch] initialized (DS OFF, online eval disabled).")

    # Force 2 outputs and no deep supervision for the network
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            2,      # two multilabel channels
            False   # no deep supervision heads
        )

    # BCEWithLogits + SoftDice (multilabel)
    def _build_loss(self):
        pos_weight = None  # e.g. torch.tensor([1.0, 1.3], device=self.device)
        return BCEDiceLossMultiLabel(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

    # ---------- helpers ----------
    @staticmethod
    def _to_multilabel(target: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
          - (N,2,...) binary -> return float
          - (N,1,...) or (N,...) with {0,1,2,3} -> split into 2 channels
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

    @staticmethod
    def _dice_from_binary(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        """
        pred, targ: (N, 2, ...) binary {0,1}
        returns (N, 2) dice in [0,1]
        """
        y_f = targ.reshape(targ.shape[0], targ.shape[1], -1)
        p_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
        inter = (y_f * p_f).sum(-1)
        denom = y_f.sum(-1) + p_f.sum(-1)
        return (2. * inter) / torch.clamp(denom, min=1.0)

    # ---------- training ----------
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = batch['data'].to(self.device, non_blocking=True)
        y = batch['target']
        if isinstance(y, (list, tuple)):
            y = y[0]
        y = y.to(self.device, non_blocking=True)
        y = self._to_multilabel(y)

        with _mixed_precision_ctx(self.device, self.grad_scaler is not None):
            logits = self.network(x)            # (N, 2, ...)
            loss = self.loss(logits, y)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {"loss": float(loss.detach().cpu().item())}

    # ---------- epoch validation (metrics ONLY, no file export) ----------
    @torch.no_grad()
    def _validate_metrics_only(self) -> Dict[str, Any]:
        # Ensure we have a val loader
        if not hasattr(self, "dataloader_val") or self.dataloader_val is None:
            old = self.num_val_workers
            self.num_val_workers = 0
            _, self.dataloader_val = self.get_dataloaders()
            self.num_val_workers = old

        self.network.eval()
        pred_thresh = 0.5
        meter = {"n": 0, "dice_L": 0.0, "dice_R": 0.0}

        for batch in self.dataloader_val:
            x = batch['data'].to(self.device, non_blocking=True)
            y = batch['target']
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.to(self.device, non_blocking=True)
            y_ml = self._to_multilabel(y)

            with _mixed_precision_ctx(self.device, self.grad_scaler is not None):
                logits = self.network(x)
                probs = torch.sigmoid(logits)

            pred = (probs >= pred_thresh).float()
            dice = self._dice_from_binary(pred, y_ml)  # (N,2)

            meter["dice_L"] += float(dice[:, 0].mean().cpu().item())
            meter["dice_R"] += float(dice[:, 1].mean().cpu().item())
            meter["n"] += 1

        if meter["n"] > 0:
            meter["dice_L"] /= meter["n"]
            meter["dice_R"] /= meter["n"]
        meter["dice_mean"] = 0.5 * (meter["dice_L"] + meter["dice_R"])
        return meter

    # Hook called by base trainer each epoch.
    # Return POSITIVE dice so the progress plot shows 0..1.
    # For the internal "best EMA", we pass the NEGATIVE (lower is better).
    def _on_epoch_end_do_validation(self):
        meter = self._validate_metrics_only()
        dice_pos = float(meter["dice_mean"])  # 0..1
        # keep nnUNet selection logic (minimize):
        if hasattr(self, "_maybe_update_best_ema"):
            self._maybe_update_best_ema(-dice_pos)
        elif hasattr(self, "maybe_update_best_ema"):
            self.maybe_update_best_ema(-dice_pos)
        # Log a friendly line as well:
        self.print_to_log_file(
            f"[ML] epoch dice -> left={meter['dice_L']:.4f}, right={meter['dice_R']:.4f}, mean={dice_pos:.4f}"
        )
        # <- return POSITIVE for the plot/log line "Pseudo dice [...]"
        return dice_pos

    # ---------- final validation/export AFTER training ----------
    @torch.no_grad()
    def perform_actual_validation(self, export_validation_probabilities: bool = False) -> None:
        """
        Runs *once* after training:
         - computes dice again (not strictly necessary),
         - exports 4-D NIfTI (channels-last, uint8) & summary for all val cases.
        """
        self.network.eval()

        # fresh single-worker loader to be robust
        old = self.num_val_workers
        self.num_val_workers = 0
        _, val_loader = self.get_dataloaders()
        self.num_val_workers = old

        outdir = os.path.join(self.output_folder, 'validation_ml')
        # clean the folder to avoid leftovers from previous runs
        if os.path.isdir(outdir):
            for fn in os.listdir(outdir):
                p = os.path.join(outdir, fn)
                try:
                    os.remove(p)
                except IsADirectoryError:
                    shutil.rmtree(p)
        else:
            os.makedirs(outdir, exist_ok=True)

        pred_thresh = 0.5
        meter = {"n": 0, "dice_L": 0.0, "dice_R": 0.0}
        per_case = []

        for batch in val_loader:
            x = batch['data'].to(self.device, non_blocking=True)
            y = batch['target']
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.to(self.device, non_blocking=True)
            y_ml = self._to_multilabel(y)

            with _mixed_precision_ctx(self.device, self.grad_scaler is not None):
                logits = self.network(x)
                probs = torch.sigmoid(logits)

            pred = (probs >= pred_thresh).float()
            dice = self._dice_from_binary(pred, y_ml)  # (N,2)

            props_list = batch.get('properties', [None] * pred.shape[0])
            for i in range(pred.shape[0]):
                paths = export_multilabel_pred(
                    pred[i].detach().cpu().numpy(), props_list[i], out_dir=outdir
                )
                dL = float(dice[i, 0].cpu().item())
                dR = float(dice[i, 1].cpu().item())
                per_case.append({
                    "case": paths["case_id"],
                    "dice_left": dL,
                    "dice_right": dR,
                    "pred_4d": paths["2ch"],
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

        with open(os.path.join(outdir, "summary_ml.json"), "w") as f:
            json.dump({"summary": meter, "metric_per_case": per_case}, f, indent=2)

        self.print_to_log_file(
            f"[ML] final export -> left={meter['dice_L']:.4f}, right={meter['dice_R']:.4f}, mean={meter['dice_mean']:.4f} "
            f"(files in {outdir})"
        )