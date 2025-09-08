from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, json, torch
from torch import autocast
from torch import distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.multilabel_losses import BCEDiceLossMultiLabel
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.inference.export_prediction_multilabel import export_multilabel_pred

class nnUNetTrainer_ML2Ch(nnUNetTrainer):
    """
    Two-channel MULTILABEL trainer (left/right):
      - net outputs 2 logits with sigmoid
      - targets may be:
          * (N,2,...) binary channels (LICA/RICA), or
          * (N,3,...) with last channel = ignore (strip it), or
          * single labelmap with {0 bg, 1 left, 2 right, 3 overlap}.
      - overlap allowed
    """

    # keep init signature identical to base
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # we want no DS and no stock online eval (we provide our own)
        self.enable_deep_supervision = False
        self.enable_online_evaluation = False
        # epochs/checkpoint cadence as you like
        self.num_epochs = 30
        self.validate_every = 1
        self.save_every = max(1, self.num_epochs // 5)

    def initialize(self):
        # make sure flags are set BEFORE network is built
        self.enable_deep_supervision = False
        self.enable_online_evaluation = False
        super().initialize()
        self.print_to_log_file("[ML2Ch] initialized (DS OFF, online eval disabled).")

    # force 2 outputs, DS OFF irrespective of label_manager
    @staticmethod
    def build_network_architecture(architecture_class_name: str, arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import, num_input_channels: int,
                                   num_output_channels: int, enable_deep_supervision: bool = True):
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            2,          # two multilabel channels
            False       # deep supervision off
        )

    def _build_loss(self):
        # Optional: per-channel class imbalance weighting via pos_weight
        pos_weight = None  # e.g., torch.tensor([1.0, 1.3], device=self.device)
        return BCEDiceLossMultiLabel(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)

    # ---------- helpers ----------
    @staticmethod
    def _to_multilabel(target: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - (N,2,...) binary -> return float
          - (N,3,...) where last is ignore -> use first two, return float
          - (N,1,...) or (N,...) int mask with {0,1,2,3} -> split to 2 channels
        """
        if isinstance(target, (list, tuple)):
            target = target[0]

        # Channels present
        if target.ndim >= 2 and target.shape[1] >= 2:
            # Strip ignore channel if present
            if target.shape[1] > 2:
                target = target[:, :2]
            return target.float()

        # Single-channel or dense labelmap
        if target.ndim >= 2 and target.shape[1] == 1:
            tgt = target[:, 0]
        else:
            tgt = target

        ch_left  = (tgt == 1) | (tgt == 3)
        ch_right = (tgt == 2) | (tgt == 3)
        return torch.stack([ch_left, ch_right], dim=1).float()

    @staticmethod
    def _dice_per_channel(pred: torch.Tensor, targ: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        pred, targ: (N,2,...) binary (0/1) tensors
        returns (N,2) dice (positive)
        """
        y = targ.reshape(targ.shape[0], targ.shape[1], -1)
        p = pred.reshape(pred.shape[0], pred.shape[1], -1)
        inter = (y * p).sum(-1)
        denom = y.sum(-1) + p.sum(-1)
        return (2. * inter + eps) / (denom + eps)

    # ---------- training ----------
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        x = batch['data'].to(self.device, non_blocking=True)
        y = batch['target']
        if isinstance(y, (list, tuple)): y = y[0]
        y = y.to(self.device, non_blocking=True)
        y = self._to_multilabel(y)

        ctx = autocast(self.device.type, enabled=(self.grad_scaler is not None)) \
            if self.device.type in ('cuda', 'mps') else dummy_context()
        with ctx:
            logits = self.network(x)  # (N, 2, ...)
            loss = self.loss(logits, y)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": float(loss.detach().cpu().item())}

    # ---------- validation for logging (no file export here) ----------
    @torch.no_grad()
    def _validate_and_log_once(self) -> Dict[str, float]:
        self.network.eval()
        pred_thresh = 0.5

        n_batches = 0
        diceL, diceR = 0.0, 0.0

        for batch in self.dataloader_val:
            x = batch['data'].to(self.device, non_blocking=True)
            y = batch['target']
            if isinstance(y, (list, tuple)): y = y[0]
            y = y.to(self.device, non_blocking=True)
            y_ml = self._to_multilabel(y)  # (N,2,...)

            ctx = autocast(self.device.type, enabled=(self.grad_scaler is not None)) \
                if self.device.type in ('cuda', 'mps') else dummy_context()
            with ctx:
                logits = self.network(x)
                probs = torch.sigmoid(logits)
                pred = (probs >= pred_thresh).float()

            d = self._dice_per_channel(pred, y_ml)  # (N,2)
            diceL += float(d[:, 0].mean().cpu())
            diceR += float(d[:, 1].mean().cpu())
            n_batches += 1

        if n_batches > 0:
            diceL /= n_batches
            diceR /= n_batches
        mean_dice = 0.5 * (diceL + diceR)

        # update nnU-Net logger so the progress plot shows POSITIVE pseudo dice
        self.logger.log('mean_fg_dice', mean_dice, self.current_epoch)

        return {"dice_left": diceL, "dice_right": diceR, "dice_mean": mean_dice}

    # Hook into base validation lifecycle **without** exporting files every epoch
    def on_validation_epoch_start(self):
        if self.dataloader_val is None:
            self.get_dataloaders()
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        # We still need a val loss to populate plots; compute loss only
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list): target = target[0]
        target = target.to(self.device, non_blocking=True)
        target = self._to_multilabel(target)
        with autocast(self.device.type, enabled=(self.grad_scaler is not None)) \
                if self.device.type in ('cuda', 'mps') else dummy_context():
            logits = self.network(data)
            l = self.loss(logits, target)
        return {'loss': l.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # log mean val loss like base
        import numpy as np
        losses = [v['loss'] for v in val_outputs]
        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_all = [None for _ in range(world_size)]
            dist.all_gather_object(losses_all, losses)
            loss_here = float(np.mean([np.mean(x) for x in losses_all]))
        else:
            loss_here = float(np.mean(losses))
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # now compute and log POSITIVE multilabel dice
        meter = self._validate_and_log_once()
        self.print_to_log_file(
            f"ML Dice -> left={meter['dice_left']:.4f}, right={meter['dice_right']:.4f}, mean={meter['dice_mean']:.4f}"
        )

    # Export predictions only at the very end (no per-epoch spam)
    def on_train_end(self):
        super().on_train_end()  # saves checkpoints and cleans up
        try:
            self.perform_actual_validation(export_validation_probabilities=False)
        except Exception as e:
            self.print_to_log_file(f"[ML2Ch] final export skipped (perform_actual_validation incompatible): {e}")

    # Override actual validation to export 2-ch predictions once, into validation_ml/
    @torch.no_grad()
    def perform_actual_validation(self, export_validation_probabilities: bool = False) -> None:
        self.network.eval()
        outdir = os.path.join(self.output_folder, 'validation_ml')
        os.makedirs(outdir, exist_ok=True)
        pred_thresh = 0.5

        # Traverse val set and write 2-ch + per-channel NIfTIs
        for batch in self.dataloader_val:
            x = batch['data'].to(self.device, non_blocking=True)
            props_list = batch.get('properties', [None] * x.shape[0])

            logits = self.network(x)
            probs = torch.sigmoid(logits)
            pred = (probs >= pred_thresh).float().cpu().numpy()  # (N,2, Z, Y, X)

            for i in range(pred.shape[0]):
                props = props_list[i]
                case_id = None
                if isinstance(props, dict):
                    case_id = props.get('case_identifier', None)
                export_multilabel_pred(pred[i], props, out_dir=outdir, case_id=case_id)
