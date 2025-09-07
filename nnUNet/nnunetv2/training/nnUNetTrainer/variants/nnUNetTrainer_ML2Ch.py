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
    Two-channel multilabel trainer for left/right hemispheres.

    - Network outputs 2 logits channels (sigmoid).
    - Targets may be (N,2,...) binary OR a single int mask with {0,1,2,3}:
        0=bg, 1=left, 2=right, 3=both (overlap allowed).
    - Custom validation writes .nii files (2ch, left, right) to validation_ml/.
    """

    # ---------------- core knobs ----------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # run short sanity trainings by default
        self.num_epochs = 30
        self.validate_every = 1
        self.save_every = max(1, self.num_epochs // 5)  # ~5 checkpoints

    # ---------------- setup ----------------
    def initialize(self, *args, **kwargs):
        """
        Use base initialization but disable online evaluation
        to avoid the 'Pseudo dice' path.
        """
        super().initialize(*args, **kwargs)
        self.enable_online_evaluation = False
        self.print_to_log_file("[ML2Ch] initialized: online eval disabled, custom validation active.")

    def print_to_log_file(self, *args, also_print_to_console: bool = True, add_timestamp: bool = True):
        """
        Filter out nnUNet's 'Pseudo dice ...' spam if something still prints it.
        """
        msg = " ".join(str(a) for a in args if a is not None)
        if "Pseudo dice" in msg:
            return
        return super().print_to_log_file(*args,
                                         also_print_to_console=also_print_to_console,
                                         add_timestamp=add_timestamp)

    def finish_online_evaluation(self):
        # no-op: we don't use base online eval
        return

    # ---------------- model / loss ----------------
    def determine_num_output_channels(self, plans_manager, dataset_json) -> int:
        # 2 channels: left, right (sigmoid)
        return 2

    def configure_loss(self):
        """
        Multilabel BCE+Dice. You can set a per-channel pos_weight if one side is rarer, e.g.:
          pos_weight = torch.tensor([1.0, 1.3], device=self.device)
        """
        pos_weight = None
        self.loss = BCEDiceLossMultiLabel(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

    # ---------------- target handling ----------------
    @staticmethod
    def _to_multilabel(target: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
          - (N,2,...) already-binary → return as float
          - (N,1,...) or (N,...) integer mask with {0,1,2,3} → split into 2 channels
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

    # ---------------- train step ----------------
    def run_training_iteration(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        x = data['data']
        y = self._to_multilabel(data['target'])

        # quick sanity probe every ~200 iterations to ensure both channels are present
        if getattr(self, "iteration", 0) % 200 == 0:
            ysum = y.sum(dim=list(range(2, y.ndim))).float().mean(0)
            self.print_to_log_file(
                f"[probe] avg GT voxels per batch -> L={ysum[0].item():.1f}, R={ysum[1].item():.1f}"
            )

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

    # ---------------- validation ----------------
    @torch.no_grad()
    def validate(self,
                 do_mirroring: bool = True,
                 use_gaussian: bool = True,
                 tiled: bool = True) -> Dict[str, Any]:
        """
        Custom validation:
        - Forward val set
        - Threshold sigmoid probs to binary
        - Compute Dice per channel (safe when empty)
        - Export .nii predictions via export_multilabel_pred
        - Log a compact ML Dice summary
        """
        self.network.eval()
        outdir = os.path.join(self.output_folder, 'validation_ml')
        os.makedirs(outdir, exist_ok=True)

        pred_thresh = 0.45
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

            # compute per-channel Dice safely
            y_f = y_ml.reshape(y_ml.shape[0], y_ml.shape[1], -1)
            p_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
            inter = (y_f * p_f).sum(-1)
            denom = y_f.sum(-1) + p_f.sum(-1)
            dice = (2. * inter) / torch.clamp(denom, min=1.0)  # (N, 2)

            # export per case (needs case properties from dataloader)
            props_list = batch.get('properties', None)
            if props_list is None:
                self.print_to_log_file("[warning] no properties in batch; skipping export.")
            else:
                for i in range(pred.shape[0]):
                    props = props_list[i]
                    paths = export_multilabel_pred(pred[i].cpu().numpy(), props, out_dir=outdir)
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

        self.network.train()  # be explicit
        return meter

    # ---------------- hook for base trainer ----------------
    def _on_epoch_end_do_validation(self):
        """
        Drive nnUNet's 'best model' bookkeeping with our ML Dice.
        """
        meter = self.validate()
        current = float(meter["dice_mean"])
        # let the base trainer know our score
        self.output_metric = current
        self.was_best_on_epoch = False

        # different nnUNet commits expose one of these:
        if hasattr(self, "maybe_update_best_ema"):
            self.was_best_on_epoch = self.maybe_update_best_ema(current)
        elif hasattr(self, "_maybe_update_best_ema"):
            self.was_best_on_epoch = self._maybe_update_best_ema(current)

        return current