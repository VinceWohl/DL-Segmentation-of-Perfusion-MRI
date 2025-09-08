import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Union, Tuple

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p


class MultiLabelHead(nn.Module):
    """Multi-label head that produces independent binary predictions for each hemisphere."""
    def __init__(self, original_conv):
        super().__init__()
        
        # Store original properties
        in_features = original_conv.in_channels
        kernel_size = original_conv.kernel_size
        padding = original_conv.padding
        has_bias = original_conv.bias is not None
        
        # Create multi-label head - single conv layer outputting 2 channels
        # This maintains spatial relationships while allowing independent predictions
        self.multi_label_head = nn.Conv2d(in_features, 2, kernel_size=kernel_size, 
                                        padding=padding, bias=has_bias)
        
        # Initialize with original weights if available
        if original_conv.out_channels == 2:
            with torch.no_grad():
                self.multi_label_head.weight.copy_(original_conv.weight)
                if has_bias:
                    self.multi_label_head.bias.copy_(original_conv.bias)
    
    def forward(self, x):
        # Single forward pass through multi-label head
        # Output: (batch, 2, H, W) where each channel is an independent binary prediction
        return self.multi_label_head(x)


class SharedDecoderNetwork(nn.Module):
    """
    Network with shared encoder and decoder, but multi-label head for independent predictions.
    This preserves spatial relationships while allowing independent hemisphere predictions.
    """
    def __init__(self, base_network):
        super().__init__()
        
        # Store the base network and expose its decoder attribute for nnUNet compatibility
        self.base_network = base_network
        
        # Expose decoder attribute for nnUNet's deep supervision management
        if hasattr(base_network, 'decoder'):
            self.decoder = base_network.decoder
        
        # Find and replace the final convolution layer with multi-label head
        self._replace_final_conv()
    
    def _find_and_replace_final_conv(self, module, parent=None, name=None):
        """Recursively find and replace the final conv layer."""
        for child_name, child in module.named_children():
            if hasattr(child, 'out_channels') and child.out_channels == 2:
                # Found the final conv layer - replace it with multi-label head
                new_module = MultiLabelHead(child)
                setattr(module, child_name, new_module)
                return True
            
            # Recurse into child modules
            if self._find_and_replace_final_conv(child, module, child_name):
                return True
        
        return False
    
    def _replace_final_conv(self):
        """Find and replace the final convolution layer."""
        if not self._find_and_replace_final_conv(self.base_network):
            raise RuntimeError("Could not find final convolution layer with 2 output channels")
    
    def forward(self, x):
        # Forward through the base network (now with multi-label head)
        return self.base_network(x)


class SpatialComplementarySharedDecoderLoss(nn.Module):
    """
    SharedDecoder loss with both spatial and complementary enhancements.
    Combines BCE + Dice + Spatial consistency + Complementary loss.
    """
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, 
                 spatial_weight=0.1, complementary_weight=0.1, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.spatial_weight = spatial_weight
        self.complementary_weight = complementary_weight
        
        # BCE loss for each channel
        self.bce = nn.BCEWithLogitsLoss(**bce_kwargs)
        
        # Dice loss with sigmoid activation
        self.dice = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)
        
    def _spatial_consistency_loss(self, pred_probs: torch.Tensor, target: torch.Tensor):
        """
        Compute target-guided spatial consistency losses for perfusion territories (vectorized).
        
        Args:
            pred_probs: Sigmoid probabilities (B, 2, H, W)
            target: Ground truth masks (B, 2, H, W)
        """
        pred_left = pred_probs[:, 0]   # (B, H, W)
        pred_right = pred_probs[:, 1]  # (B, H, W)
        target_left = target[:, 0]     # (B, H, W)
        target_right = target[:, 1]    # (B, H, W)
        
        # 1. Overlap penalty: discourage simultaneous high confidence in non-overlap regions
        non_overlap_mask = (target_left + target_right) <= 1.0
        overlap_penalty = torch.mean(pred_left * pred_right * non_overlap_mask.float())
        
        # 2. Coverage consistency: encourage proper brain coverage
        target_coverage = torch.clamp(target_left + target_right, 0, 1)
        pred_coverage = torch.clamp(pred_left + pred_right, 0, 1)
        coverage_loss = F.mse_loss(pred_coverage, target_coverage)
        
        # 3. Mutual exclusivity in appropriate regions (vectorized)
        left_only_regions = (target_left > 0) & (target_right == 0)
        right_only_regions = (target_right > 0) & (target_left == 0)
        
        # Use masked means for efficiency
        exclusivity_loss = 0
        if left_only_regions.sum() > 0:
            exclusivity_loss += torch.mean(pred_right[left_only_regions] ** 2)
        if right_only_regions.sum() > 0:
            exclusivity_loss += torch.mean(pred_left[right_only_regions] ** 2)
        
        return overlap_penalty + coverage_loss + exclusivity_loss
        
    def _complementary_loss(self, pred_probs: torch.Tensor, target: torch.Tensor):
        """
        Complementary information loss for perfusion territories.
        Encourages mutual exclusivity and proper brain coverage.
        """
        pred_left = pred_probs[:, 0]   # (B, H, W)
        pred_right = pred_probs[:, 1]  # (B, H, W)
        target_left = target[:, 0]     # (B, H, W)
        target_right = target[:, 1]    # (B, H, W)
        
        # 1. Encourage mutual exclusivity where appropriate
        # Penalize simultaneous predictions in non-overlap regions
        exclusivity_loss = torch.mean(pred_left * pred_right * (1 - target_left) * (1 - target_right))
        
        # 2. Encourage coverage of the brain region
        # Ensure predicted coverage matches target coverage
        coverage_target = torch.clamp(target_left + target_right, 0, 1)
        coverage_pred = torch.clamp(pred_left + pred_right, 0, 1)
        coverage_loss = F.mse_loss(coverage_pred, coverage_target)
        
        return exclusivity_loss + coverage_loss
        
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        target = target.float()
        
        # Base losses: BCE + Dice
        dice_loss = self.dice(net_output, target) if self.weight_dice != 0 else 0
        
        if self.weight_ce != 0:
            bce_loss_ch0 = self.bce(net_output[:, 0], target[:, 0])
            bce_loss_ch1 = self.bce(net_output[:, 1], target[:, 1])
            bce_loss = 0.5 * bce_loss_ch0 + 0.5 * bce_loss_ch1
        else:
            bce_loss = 0
        
        # Enhanced consistency losses
        pred_probs = torch.sigmoid(net_output)
        
        # Spatial consistency loss (now target-guided)
        spatial_loss = self._spatial_consistency_loss(pred_probs, target) if self.spatial_weight > 0 else 0
        
        # Complementary loss
        complementary_loss = 0
        if self.complementary_weight > 0:
            complementary_loss = self._complementary_loss(pred_probs, target)
        
        total_loss = (self.weight_ce * bce_loss + 
                     self.weight_dice * dice_loss + 
                     self.spatial_weight * spatial_loss +
                     self.complementary_weight * complementary_loss)
        
        return total_loss


class nnUNetTrainer_multilabel(nnUNetTrainer):
    """
    Trainer with shared decoder and both spatial and complementary loss enhancements.
    Uses shared spatial features with spatial smoothness and complementary constraints.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Override label manager
        self.label_manager._num_segmentation_heads = 2
        
        # Set max epochs for test run
        self.num_epochs = 50
        
        # Set loss weights
        self.spatial_weight = 0.1
        self.complementary_weight = 0.1
        
        # Get number of input channels
        num_input_channels = len(self.dataset_json.get('channel_names', {'0': 'CBF LICA', '1': 'CBF RICA'}))
        channel_names = list(self.dataset_json.get('channel_names', {'0': 'CBF LICA', '1': 'CBF RICA'}).values())
        
        print("Shared Decoder with Spatial + Complementary Loss Enhancement")
        print(f"Input channels ({num_input_channels}): {', '.join(channel_names)}")
        print(f"Spatial loss weight: {self.spatial_weight}")
        print(f"Complementary loss weight: {self.complementary_weight}")
        print(f"Max epochs set to: {self.num_epochs}")
        print("Training for 50 epochs for quick test run with optimized validation")
        
    @property  
    def num_segmentation_heads(self):
        return 2
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # Build the base network with 2 output channels
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, 2, enable_deep_supervision
        )
        
        # Wrap it with shared decoder + multi-label head architecture
        network = SharedDecoderNetwork(base_network)
        return network
        
    def _build_loss(self):
        """Build enhanced loss function with BCE + Dice + Spatial + Complementary."""
        loss = SpatialComplementarySharedDecoderLoss(
            bce_kwargs={},
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': False,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            weight_ce=1,
            weight_dice=1,
            spatial_weight=self.spatial_weight,
            complementary_weight=self.complementary_weight,
            dice_class=MemoryEfficientSoftDiceLoss
        )
        
        if self._do_i_compile():
            loss.dice = torch.compile(loss.dice)
        
        # Handle deep supervision
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss

    def validation_step(self, batch: dict) -> dict:
        """Validation step with per-channel metrics."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Data loaded and ready for forward pass

        # Forward pass
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.no_grad():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # Apply sigmoid and compute predictions
        predicted_probabilities = torch.sigmoid(output)
        predicted_segmentation_onehot = (predicted_probabilities > 0.5).long()
        target = target.float()
        
        # Compute per-channel metrics
        tp_hard = torch.zeros((target.shape[0], 2), dtype=torch.float32, device=target.device)
        fp_hard = torch.zeros((target.shape[0], 2), dtype=torch.float32, device=target.device)
        fn_hard = torch.zeros((target.shape[0], 2), dtype=torch.float32, device=target.device)

        for c in range(2):
            spatial_axes_sliced = tuple(range(1, target[:, c].ndim))
            tp_hard[:, c] = torch.sum((predicted_segmentation_onehot[:, c] == 1) & (target[:, c] == 1), dim=spatial_axes_sliced)
            fp_hard[:, c] = torch.sum((predicted_segmentation_onehot[:, c] == 1) & (target[:, c] == 0), dim=spatial_axes_sliced)
            fn_hard[:, c] = torch.sum((predicted_segmentation_onehot[:, c] == 0) & (target[:, c] == 1), dim=spatial_axes_sliced)

        tp_hard = tp_hard.detach().cpu().numpy()
        fp_hard = fp_hard.detach().cpu().numpy()
        fn_hard = fn_hard.detach().cpu().numpy()

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Compute and log per-channel Dice scores."""
        outputs_collated = collate_outputs(val_outputs)
        
        tp_array = outputs_collated['tp_hard']
        fp_array = outputs_collated['fp_hard']  
        fn_array = outputs_collated['fn_hard']
        
        tp_per_channel = np.sum(tp_array, axis=(0, 1))
        fp_per_channel = np.sum(fp_array, axis=(0, 1))
        fn_per_channel = np.sum(fn_array, axis=(0, 1))
        
        dice_scores = []
        for i in range(2):
            tp_i = float(tp_per_channel[i])
            fp_i = float(fp_per_channel[i])
            fn_i = float(fn_per_channel[i])
                
            dice = (2 * tp_i) / (2 * tp_i + fp_i + fn_i + 1e-8)
            dice_scores.append(dice)
        
        mean_dice = np.mean(dice_scores)
        
        # Validation dice computed
        
        self.logger.log('dice_per_class_or_region', dice_scores, self.current_epoch)
        self.logger.log('mean_fg_dice', float(mean_dice), self.current_epoch)
        
        val_loss = np.mean([output['loss'] for output in val_outputs])
        self.logger.log('val_losses', val_loss, self.current_epoch)
        
        if 'val_scores' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['val_scores'] = []
        self.logger.my_fantastic_logging['val_scores'].append(-mean_dice)
        
        return mean_dice
    
    def initialize_network(self):
        """Override to ensure our custom label manager is used consistently."""
        # Set the label manager BEFORE network initialization
        self.label_manager._num_segmentation_heads = 2
        
        # Now initialize the network
        super().initialize_network()
        
        
    def perform_actual_validation(self, save_probabilities: bool = False):
        """Use original nnUNet validation pipeline for consistency with training."""
        print("Using original nnUNet validation pipeline for optimal performance...")
        
        # Disable deep supervision and set network to eval mode
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        
        # Use original nnUNetPredictor with all optimizations
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,                    # 50% overlap for sliding window
            use_gaussian=True,                     # Gaussian blending of overlaps
            use_mirroring=True,                    # Test-time augmentation
            perform_everything_on_device=True,     # GPU acceleration
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        
        # Initialize predictor with our trained network
        predictor.manual_initialization(
            self.network, 
            self.plans_manager, 
            self.configuration_manager, 
            None,
            self.dataset_json, 
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes
        )
        
        print("nnUNetPredictor initialized - using sliding window + TTA + Gaussian blending")
        
        # Use the original validation pipeline from parent class
        # This ensures consistency with standard nnUNet validation
        return super().perform_actual_validation(save_probabilities)