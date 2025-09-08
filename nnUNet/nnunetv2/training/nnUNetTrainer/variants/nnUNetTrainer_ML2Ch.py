from __future__ import annotations
from typing import Dict, Any
import os, json, torch
from torch.cuda.amp import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.multilabel_losses import BCEDiceLossMultiLabel
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.inference.export_prediction_multilabel import export_multilabel_pred
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

class nnUNetTrainer_ML2Ch(nnUNetTrainer):
    """
    Two-channel multilabel trainer (left/right). Network outputs 2 sigmoid logits.
    Targets can be (N,2,...) binary or an int mask with {0,1,2,3} (3 = overlap).
    """

    # keep the *args/**kwargs signature – this works with your base class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make sure targets are a tensor, not a list: no deep supervision
        self.enable_deep_supervision = False
        # shorter run (adjust as you like)
        self.num_epochs = 30
        self.validate_every = 1
        self.save_every = max(1, self.num_epochs // 5)

    # quieter logs: hide 'pseudo dice' and our old probe lines
    def print_to_log_file(self, *args, also_print_to_console: bool = True, add_timestamp: bool = True):
        msg = " ".join(str(a) for a in args if a is not None).lower()
        if "pseudo dice" in msg or "[probe] avg gt voxels" in msg:
            return
        return super().print_to_log_file(*args,
                                         also_print_to_console=also_print_to_console,
                                         add_timestamp=add_timestamp)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.enable_online_evaluation = False  # turn off stock online eval
        self.print_to_log_file("[ML2Ch] initialized (DS OFF, online eval disabled).")

    # ---- fix network outputs (2 heads, no DS) ----
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # ignore supplied num_output_channels/DS and force our choices
        arch = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
        )
        # nnUNet v2 networks expose these attributes
        if hasattr(arch, 'num_classes'):
            arch.num_classes = 2
        if hasattr(arch, 'seg_output_use_sigmoid'):
            arch.seg_output_use_sigmoid = True
        if hasattr(arch, 'deep_supervision'):
            arch.deep_supervision = False
        return arch

    # ---- loss ----
    def _build_loss(self):
        pos_weight = None  # e.g., torch.tensor([1.0, 1.2], device=self.device)
        return BCEDiceLossMultiLabel(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

    # ---- helpers ----
    @staticmethod
    def _to_multilabel(target: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - (N,2,...) binary → return as float
          - (N,1,...) or (N,...) int mask with {0,1,2,3} → split to 2 channels
        """
        if isinstance(target, (list, tuple)):   # robustness if DS ever sneaks in
            target = target[0]
        if target.ndim >= 2 and target.shape[1] == 2:
            return target.float()
        tgt = target[:, 0] if (target.ndim >= 2 and target.shape[1] == 1) else target
        ch_left  = (tgt == 1) | (tgt == 3)
        ch_right = (tgt == 2) | (tgt == 3)
        return torch.stack([ch_left, ch_right], dim=1).float()

    # ---- training step ----
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = batch['data'].to(self.device, non_blocking=True)
        y = self._to_multilabel(batch['target'].to(self.device, non_blocking=True))

        use_amp = bool(getattr(self, "fp16", False))
        ctx = autocast if use_amp else dummy_context
        with ctx():
            logits = self.network(x)          # (N, 2, ...)
            loss = self.loss(logits, y)

        if use_amp:
            self.grad_scaler.scale(loss).backward()
            if getattr(self, "gradient_clipping", None) is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            if getattr(self, "gradient_clipping", None) is not None:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)
            self.optimizer.step()

        return {"loss": float(loss.detach().cpu().item())}

    # ---- our own multilabel validation & export ----
    @torch.no_grad()
    def validate(self, do_mirroring: bool = True, use_gaussian: bool = True, tiled: bool = True) -> Dict[str, Any]:
        self.network.eval()
        outdir = os.path.join(self.output_folder, 'validation_ml')
        os.makedirs(outdir, exist_ok=True)

        pred_thresh = 0.5
        meter = {"n": 0, "dice_L": 0.0, "dice_R": 0.0}
        per_case = []

        for batch in self.val_data_loader:
            x = batch['data'].to(self.device, non_blocking=True)
            y = self._to_multilabel(batch['target'].to(self.device, non_blocking=True))

            with (autocast() if bool(getattr(self, "fp16", False)) else dummy_context()):
                logits = self.network(x)
                probs = torch.sigmoid(logits)

            pred = (probs >= pred_thresh).float()

            # dice per channel
            y_f = y.reshape(y.shape[0], y.shape[1], -1)
            p_f = pred.reshape(pred.shape[0], pred.shape[1], -1)
            inter = (y_f * p_f).sum(-1)
            denom = y_f.sum(-1) + p_f.sum(-1)
            dice = (2.0 * inter) / torch.clamp(denom, min=1.0)  # (N,2)

            # export per case
            props_list = batch.get('properties', [None] * pred.shape[0])
            for i in range(pred.shape[0]):
                props = props_list[i]
                paths = export_multilabel_pred(pred[i].cpu().numpy(), props, out_dir=outdir)
                dL, dR = float(dice[i, 0].cpu()), float(dice[i, 1].cpu())
                per_case.append({
                    "case": paths.get("case_id", f"case_{len(per_case)}"),
                    "dice_left": dL, "dice_right": dR,
                    "pred_2ch": paths["2ch"], "pred_left": paths["left"], "pred_right": paths["right"],
                })

            meter["dice_L"] += float(dice[:, 0].mean().cpu())
            meter["dice_R"] += float(dice[:, 1].mean().cpu())
            meter["n"] += 1

        if meter["n"] > 0:
            meter["dice_L"] /= meter["n"]
            meter["dice_R"] /= meter["n"]
        meter["dice_mean"] = 0.5 * (meter["dice_L"] + meter["dice_R"]) if meter["n"] > 0 else 0.0

        self.print_to_log_file(
            f"ML Dice -> left={meter['dice_L']:.4f}, right={meter['dice_R']:.4f}, mean={meter['dice_mean']:.4f}"
        )
        with open(os.path.join(outdir, "summary_ml.json"), "w") as f:
            json.dump({"summary": meter, "metric_per_case": per_case}, f, indent=2)

        return meter

    # ---- use our validator at epoch end & at the very end ----
    def _on_epoch_end_do_validation(self):
        meter = self.validate()
        current = float(meter["dice_mean"])
        # update best-EMA with our metric (not pseudo dice)
        if hasattr(self, "_maybe_update_best_ema"):
            self._maybe_update_best_ema(current)
        elif hasattr(self, "maybe_update_best_ema"):
            self.maybe_update_best_ema(current)
        return current

    def perform_actual_validation(self, export_validation_probabilities: bool = False):
        # The stock validator assumes softmax & K-class; our multilabel needs custom export.
        self.print_to_log_file("[ML2Ch] running custom multilabel validation/export.")
        self.validate()
