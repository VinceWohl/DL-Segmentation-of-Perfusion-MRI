# nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainer_ML2Ch.py
from __future__ import annotations
from typing import Dict, Any
import os, json, torch
from torch.cuda.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.multilabel_losses import BCEDiceLossMultiLabel
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.inference.export_prediction_multilabel import export_multilabel_pred

class nnUNetTrainer_ML2Ch(nnUNetTrainer):
    """
    Two-channel multilabel trainer (left/right hemispheres):
      - Network outputs 2 logits channels (sigmoid).
      - Targets can be (N,2,...) binary OR a single int mask with {0,1,2,3}.
      - Overlap allowed (both channels can be 1).
    """

    # --- IMPORTANT: disable stock online evaluation & noise ---
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.enable_online_evaluation = False
        self.print_to_log_file("[ML2Ch] custom trainer initialized (online eval disabled).")

    def finish_online_evaluation(self):
        # Some nnU-Net versions still call this; make it a no-op
        return

    def print_to_log_file(self, *args, also_print_to_console: bool = True, add_timestamp: bool = True):
        # Swallow the base class "Pseudo dice [...]" line to avoid confusion
        msg = " ".join(str(a) for a in args if a is not None)
        if "Pseudo dice" in msg:
            return
        return super().print_to_log_file(*args,
                                         also_print_to_console=also_print_to_console,
                                         add_timestamp=add_timestamp)

    # --- outputs ---
    def determine_num_output_channels(self, plans_manager, dataset_json) -> int:
        return 2

    def configure_loss(self):
        # tip: bump pos_weight for a rarer/right class, e.g. torch.tensor([1.0, 1.2], device=self.device)
        pos_weight = None
        self.loss = BCEDiceLossMultiLabel(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

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

    def run_training_iteration(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        x = data['data']     # (N, Cin, ...)
        y = self._to_multilabel(data['target'])  # (N, 2, ...)

        # (optional) quick probe every ~200 iters to confirm class presence in batch
        if getattr(self, "iteration", 0) % 200 == 0:
            ysum = y.sum(dim=list(range(2, y.ndim))).float().mean(0)
            self.print_to_log_file(f"[probe] avg GT voxels in batch -> L={ysum[0].item():.1f}, R={ysum[1].item():.1f}")

        ctx = autocast if self.fp16 else dummy_context
        with ctx():
            logits = self.network(x)    # (N, 2, ...)
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

        return {"loss": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def validate(self, do_mirroring: bool = True, use_gaussian: bool = True, tiled: bool = True) -> Dict[str, Any]:
        """
        Multilabel validation:
          - sigmoid -> threshold
          - export 2ch + per-channel masks into validation_ml/ (all .nii)
          - compute Dice per channel in [0,1]
          - write summary_ml.json
        """
        self.network.eval()
        outdir = os.path.join(self.output_folder, 'validation_ml')
        os.makedirs(outdir, exist_ok=True)

        pred_thresh = 0.45  # try 0.35–0.5 if one channel is weaker
        meter = {"n": 0, "dice_L": 0.0, "dice_R": 0.0}
        per_case = []

        for batch in self.val_data_loader:
            x = batch['data'].to(self.device, non_blocking=True)
            y = batch['target'].to(self.device, non_blocking=True)
            y_ml = self._to_multilabel(y)  # (N, 2, ...)

            with (autocast() if self.fp16 else dummy_context()):
                logits = self.network(x)
                probs = torch.sigmoid(logits)

            pred = (probs >= pred_thresh).float()

            # metrics
            y_f = y_ml.reshape(y_ml.shape[0], y_ml.shape[1], -1)
            p_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
            inter = (y_f * p_f).sum(-1)
            denom = y_f.sum(-1) + p_f.sum(-1)
            dice = (2. * inter) / torch.clamp(denom, min=1.0)  # (N, 2)

            # export per case
            for i in range(pred.shape[0]):
                props = batch['properties'][i] if 'properties' in batch else None
                paths = export_multilabel_pred(pred[i].cpu().numpy(), props, out_dir=outdir)
                dL = float(dice[i, 0].cpu().item())
                dR = float(dice[i, 1].cpu().item())
                per_case.append({"case": paths["case_id"], "dice_left": dL, "dice_right": dR,
                                 "pred_2ch": paths["2ch"], "pred_left": paths["left"], "pred_right": paths["right"]})

            meter["dice_L"] += float(dice[:, 0].mean().cpu().item())
            meter["dice_R"] += float(dice[:, 1].mean().cpu().item())
            meter["n"] += 1

        if meter["n"] > 0:
            meter["dice_L"] /= meter["n"]
            meter["dice_R"] /= meter["n"]
            meter["dice_mean"] = 0.5 * (meter["dice_L"] + meter["dice_R"])
        else:
            meter["dice_mean"] = 0.0

        # log + write multilabel summary WITHOUT touching nnU-Net's single-label summary.json
        self.print_to_log_file(f"ML Dice -> left={meter['dice_L']:.4f}, right={meter['dice_R']:.4f}, mean={meter['dice_mean']:.4f}")
        with open(os.path.join(outdir, "summary_ml.json"), "w") as f:
            json.dump({"summary": meter, "metric_per_case": per_case}, f, indent=2)

        return meter

    # (optional) also make "best model" selection use our ML Dice:
    def _on_epoch_end_do_validation(self):
        meter = self.validate()
        current = float(meter["dice_mean"])
        if hasattr(self, "_maybe_update_best_ema"):
            self._maybe_update_best_ema(current)
        elif hasattr(self, "maybe_update_best_ema"):
            self.maybe_update_best_ema(current)
        return current