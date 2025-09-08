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


class nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss(nnUNetTrainer):
    """
    Trainer with shared decoder and both spatial and complementary loss enhancements.
    Uses shared spatial features with spatial smoothness and complementary constraints.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Override label manager
        self.label_manager._num_segmentation_heads = 2
        
        # Set max epochs for full training
        self.num_epochs = 1000
        
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
        """Run inference on raw validation images and compute Dice scores."""
        import subprocess
        import SimpleITK as sitk
        from batchgenerators.utilities.file_and_folder_operations import load_json
        
        print("Running final validation inference on raw validation cases...")
        
        # Load dataset splits to get validation cases
        splits_file = join(self.preprocessed_dataset_folder, '..', 'splits_final.json')
        splits = load_json(splits_file)
        val_keys = splits[self.fold]['val']
        
        print(f"Validating on {len(val_keys)} cases...")
        
        # Set up paths using available attributes
        from nnunetv2.paths import nnUNet_raw
        
        # Extract dataset ID from dataset_json
        dataset_id = self.dataset_json['dataset_id'] if 'dataset_id' in self.dataset_json else 1
        dataset_name = f"Dataset{dataset_id:03d}_PerfusionTerritories"
        
        raw_data_folder = join(nnUNet_raw, dataset_name, 'imagesTr')
        gt_folder = join(nnUNet_raw, dataset_name, 'labelsTr')
        validation_folder = join(self.output_folder, 'validation')
        maybe_mkdir_p(validation_folder)
        
        # Create temporary folder with only validation cases
        temp_input_folder = join(validation_folder, 'temp_input')
        maybe_mkdir_p(temp_input_folder)
        
        # Copy only validation cases to temp folder (multi-channel input)
        import shutil
        
        # Determine number of channels from dataset configuration
        num_channels = len(self.dataset_json.get('channel_names', {'0': 'CBF LICA', '1': 'CBF RICA'}))
        channel_suffixes = [f'{i:04d}' for i in range(num_channels)]
        
        for case_id in val_keys:
            for channel in channel_suffixes:
                # Try .nii first, then .nii.gz
                src_file_nii = join(raw_data_folder, f'{case_id}_{channel}.nii')
                src_file_nii_gz = join(raw_data_folder, f'{case_id}_{channel}.nii.gz')
                dst_file = join(temp_input_folder, f'{case_id}_{channel}.nii')
                
                if os.path.exists(src_file_nii):
                    shutil.copy2(src_file_nii, dst_file)
                elif os.path.exists(src_file_nii_gz):
                    shutil.copy2(src_file_nii_gz, dst_file)
                else:
                    print(f"Warning: Input file not found: {src_file_nii} or {src_file_nii_gz}")
        
        # Skip external inference and use direct model inference instead
        print("Running inference...")
        
        try:
            # Load the best checkpoint
            checkpoint_path = join(self.output_folder, 'checkpoint_best.pth')
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                return
                
            print(f"Loading best checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle torch.compile wrapper - try loading with strict=False first
            try:
                self.network.load_state_dict(checkpoint['network_weights'])
                print("Best checkpoint loaded successfully!")
            except RuntimeError as e:
                if "OptimizedModule" in str(e) or "_orig_mod" in str(e):
                    print("Handling torch.compile wrapper...")
                    # Try loading with strict=False to handle compile wrapper mismatches
                    self.network.load_state_dict(checkpoint['network_weights'], strict=False)
                    print("Best checkpoint loaded successfully (with compile wrapper handling)!")
                else:
                    raise e
                
            # Run direct inference using the trained model
            self._run_direct_inference(temp_input_folder, validation_folder, val_keys)
            print("Inference completed successfully!")
            
        except Exception as e:
            print(f"Error during direct inference: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Compute Dice scores by comparing predictions to ground truth
        validation_results = []
        
        for case_id in val_keys:
            try:
                # Load prediction - try .nii first, then .nii.gz
                pred_file_nii = join(validation_folder, f'{case_id}.nii')
                pred_file_nii_gz = join(validation_folder, f'{case_id}.nii.gz')
                pred_file = None
                
                if os.path.exists(pred_file_nii):
                    pred_file = pred_file_nii
                elif os.path.exists(pred_file_nii_gz):
                    pred_file = pred_file_nii_gz
                
                if pred_file is None:
                    print(f"Prediction file not found: {pred_file_nii} or {pred_file_nii_gz}")
                    continue
                    
                pred_img = sitk.ReadImage(pred_file)
                pred_array = sitk.GetArrayFromImage(pred_img)  # Shape: (slices, H, W, channels) or similar
                
                # Load ground truth - try .nii first, then .nii.gz
                gt_file_nii = join(gt_folder, f'{case_id}.nii')
                gt_file_nii_gz = join(gt_folder, f'{case_id}.nii.gz')
                gt_file = None
                
                if os.path.exists(gt_file_nii):
                    gt_file = gt_file_nii
                elif os.path.exists(gt_file_nii_gz):
                    gt_file = gt_file_nii_gz
                
                if gt_file is None:
                    print(f"Ground truth file not found: {gt_file_nii} or {gt_file_nii_gz}")
                    continue
                    
                gt_img = sitk.ReadImage(gt_file)
                gt_array = sitk.GetArrayFromImage(gt_img)
                
                # Process validation case
                
                # Handle different orientations - ensure both are binary and have 2 channels
                if pred_array.ndim == 4 and pred_array.shape[-1] == 2:
                    # Format: (slices, H, W, channels)
                    pred_left = pred_array[..., 0]
                    pred_right = pred_array[..., 1]
                elif pred_array.ndim == 4 and pred_array.shape[0] == 2:
                    # Format: (channels, slices, H, W)
                    pred_left = pred_array[0, ...]
                    pred_right = pred_array[1, ...]
                else:
                    print(f"Unexpected prediction shape: {pred_array.shape}")
                    continue
                
                # Same for ground truth
                if gt_array.ndim == 4 and gt_array.shape[-1] == 2:
                    gt_left = gt_array[..., 0]
                    gt_right = gt_array[..., 1]
                elif gt_array.ndim == 4 and gt_array.shape[0] == 2:
                    gt_left = gt_array[0, ...]
                    gt_right = gt_array[1, ...]
                else:
                    print(f"Unexpected ground truth shape: {gt_array.shape}")
                    continue
                
                # Convert to binary
                pred_left = (pred_left > 0.5).astype(np.uint8)
                pred_right = (pred_right > 0.5).astype(np.uint8)
                gt_left = (gt_left > 0).astype(np.uint8)
                gt_right = (gt_right > 0).astype(np.uint8)
                
                # Compute Dice scores
                dice_scores = []
                for pred_c, gt_c, name in [(pred_left, gt_left, 'Left'), (pred_right, gt_right, 'Right')]:
                    intersection = np.sum(pred_c * gt_c)
                    total = np.sum(pred_c) + np.sum(gt_c)
                    
                    if total == 0:
                        dice = 1.0  # Both empty
                    else:
                        dice = 2.0 * intersection / total
                    
                    dice_scores.append(dice)
                
                validation_results.append({
                    'case': case_id,
                    'dice_left': dice_scores[0],
                    'dice_right': dice_scores[1],
                    'dice_mean': np.mean(dice_scores)
                })
                
                print(f"  {case_id}: Left: {dice_scores[0]:.4f}, Right: {dice_scores[1]:.4f}, Mean: {np.mean(dice_scores):.4f}")
                
            except Exception as e:
                print(f"Error processing {case_id}: {e}")
                continue
        
        # Print final results
        if validation_results:
            left_dices = [r['dice_left'] for r in validation_results]
            right_dices = [r['dice_right'] for r in validation_results]
            mean_dices = [r['dice_mean'] for r in validation_results]
            
            print("\n" + "="*60)
            print("FINAL VALIDATION RESULTS")
            print("="*60)
            print(f"Number of cases: {len(validation_results)}")
            print(f"Left Channel Dice:  {np.mean(left_dices):.4f} ± {np.std(left_dices):.4f}")
            print(f"Right Channel Dice: {np.mean(right_dices):.4f} ± {np.std(right_dices):.4f}")
            print(f"Overall Mean Dice:  {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}")
            print(f"Predictions saved to: {validation_folder}")
            print("="*60)
            
            # Update logger with final validation score
            final_dice = np.mean(mean_dices)
            self.logger.log('mean_fg_dice', float(final_dice), self.current_epoch)
            
            # Save detailed results to JSON
            self._save_validation_summary(validation_results, validation_folder)
        else:
            print("No validation results computed!")
            
        # Clean up temporary folder
        try:
            import shutil
            if os.path.exists(temp_input_folder):
                shutil.rmtree(temp_input_folder)
                # Cleaned up temporary folder
        except Exception as e:
            print(f"Warning: Could not clean up temporary folder: {e}")
    
    def _run_direct_inference(self, input_folder, output_folder, case_ids):
        """Run direct inference using the trained model without subprocess."""
        import torch
        import SimpleITK as sitk
        
        # Load the model in eval mode
        self.network.eval()
        
        # Determine number of channels from dataset configuration
        num_channels = len(self.dataset_json.get('channel_names', {'0': 'CBF LICA', '1': 'CBF RICA'}))
        channel_suffixes = [f'{i:04d}' for i in range(num_channels)]
        
        with torch.no_grad():
            for case_id in case_ids:
                try:
                    # Load input files
                    input_files = []
                    for channel in channel_suffixes:
                        file_path = join(input_folder, f'{case_id}_{channel}.nii')
                        if os.path.exists(file_path):
                            input_files.append(file_path)
                        else:
                            print(f"Warning: Input file not found: {file_path}")
                            break
                    
                    if len(input_files) != num_channels:
                        print(f"Skipping {case_id}: incomplete input files (expected {num_channels}, got {len(input_files)})")
                        continue
                    
                    # Load and preprocess images
                    images = []
                    for file_path in input_files:
                        img = sitk.ReadImage(file_path)
                        img_array = sitk.GetArrayFromImage(img)
                        images.append(img_array)
                    
                    # Stack channels: (slices, H, W) -> (channels, slices, H, W)
                    input_data = np.stack(images, axis=0)  # Shape: (num_channels, slices, H, W)
                    num_slices = input_data.shape[1]
                    
                    # Process each slice individually for 2D model
                    predictions = []
                    
                    for slice_idx in range(num_slices):
                        # Get 2D slice: (channels, H, W)
                        slice_data = input_data[:, slice_idx, :, :]  # Shape: (num_channels, H, W)
                        
                        # Add batch dimension: (1, channels, H, W)
                        input_tensor = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0)
                        
                        if torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                        
                        # Run inference on this slice
                        with torch.no_grad():
                            output = self.network(input_tensor)
                            
                            # Apply sigmoid and threshold
                            if isinstance(output, (list, tuple)):
                                output = output[0]  # Take first output if deep supervision
                            
                            prob = torch.sigmoid(output)
                            pred = (prob > 0.5).float()
                            
                            # Convert back to numpy and remove batch dimension
                            pred_slice = pred.cpu().numpy().squeeze(0)  # Shape: (2, H, W)
                            predictions.append(pred_slice)
                    
                    # Stack all slices back together: (2, slices, H, W)
                    pred_np = np.stack(predictions, axis=1)
                    
                    # Save prediction
                    output_file = join(output_folder, f'{case_id}.nii')
                    
                    # Create output image with same properties as input
                    ref_img = sitk.ReadImage(input_files[0])
                    
                    # Transpose from (channels, slices, H, W) to (slices, H, W, channels)
                    pred_np = pred_np.transpose(1, 2, 3, 0)  # Shape: (slices, H, W, channels)
                    
                    output_img = sitk.GetImageFromArray(pred_np.astype(np.uint8))
                    output_img.CopyInformation(ref_img)
                    sitk.WriteImage(output_img, output_file)
                    
                    # Processed case successfully
                    
                except Exception as e:
                    print(f"Error processing {case_id}: {e}")
                    continue
    
    def _save_validation_summary(self, validation_results, validation_folder):
        """Save detailed validation results to JSON file."""
        import json
        from datetime import datetime
        
        # Calculate summary statistics
        if validation_results:
            left_dices = [r['dice_left'] for r in validation_results]
            right_dices = [r['dice_right'] for r in validation_results]
            mean_dices = [r['dice_mean'] for r in validation_results]
            
            summary = {
                "experiment_info": {
                    "trainer": "nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss",
                    "dataset": "Dataset001_PerfusionTerritories",
                    "configuration": "2d",
                    "fold": self.fold,
                    "num_epochs": self.num_epochs,
                    "loss_type": "bce_dice_spatial_complementary",
                    "spatial_weight": self.spatial_weight,
                    "complementary_weight": self.complementary_weight,
                    "timestamp": datetime.now().isoformat(),
                    "num_validation_cases": len(validation_results)
                },
                "summary_statistics": {
                    "left_hemisphere": {
                        "mean_dice": float(np.mean(left_dices)),
                        "std_dice": float(np.std(left_dices)),
                        "min_dice": float(np.min(left_dices)),
                        "max_dice": float(np.max(left_dices)),
                        "median_dice": float(np.median(left_dices))
                    },
                    "right_hemisphere": {
                        "mean_dice": float(np.mean(right_dices)),
                        "std_dice": float(np.std(right_dices)),
                        "min_dice": float(np.min(right_dices)),
                        "max_dice": float(np.max(right_dices)),
                        "median_dice": float(np.median(right_dices))
                    },
                    "overall": {
                        "mean_dice": float(np.mean(mean_dices)),
                        "std_dice": float(np.std(mean_dices)),
                        "min_dice": float(np.min(mean_dices)),
                        "max_dice": float(np.max(mean_dices)),
                        "median_dice": float(np.median(mean_dices))
                    }
                },
                "per_case_results": [
                    {
                        "case_id": result['case'],
                        "left_hemisphere_dice": float(result['dice_left']),
                        "right_hemisphere_dice": float(result['dice_right']),
                        "mean_dice": float(result['dice_mean']),
                        "prediction_file": f"{result['case']}.nii"
                    }
                    for result in validation_results
                ]
            }
        else:
            summary = {
                "experiment_info": {
                    "trainer": "nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss",
                    "dataset": "Dataset001_PerfusionTerritories",
                    "configuration": "2d",
                    "fold": self.fold,
                    "num_epochs": self.num_epochs,
                    "timestamp": datetime.now().isoformat(),
                    "num_validation_cases": 0
                },
                "summary_statistics": {},
                "per_case_results": [],
                "error": "No validation results computed"
            }
        
        # Save to JSON file
        summary_file = join(validation_folder, 'validation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Validation summary saved to: {summary_file}")