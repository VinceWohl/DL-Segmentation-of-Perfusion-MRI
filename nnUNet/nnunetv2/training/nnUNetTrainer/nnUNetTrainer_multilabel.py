import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Union, Tuple

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from copy import deepcopy
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA as default_num_processes


class MultiLabelPredictor(nnUNetPredictor):
    """Custom predictor that natively handles 2-channel binary multi-label output."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def predict_sliding_window_return_logits(self, input_image,
                                           tile_step_size: float = 0.5,
                                           mirror_axes: Tuple[int, ...] = None,
                                           use_gaussian: bool = True,
                                           precomputed_gaussian=None):
        """Override the entire sliding window prediction to handle 2-channel output."""
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        
        # Setup mirroring
        if mirror_axes is None:
            mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else ()

        # Compute Gaussian for weighting patches
        if use_gaussian:
            gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                      value_scaling_factor=1000,
                                      device=self.device) if precomputed_gaussian is None else precomputed_gaussian
        else:
            gaussian = 1

        # Initialize prediction array for 2 channels instead of default 3
        # This is the key fix - using 2 channels throughout
        n_channels = 2  # Our model outputs 2 channels
        
        # Get image properties
        spatial_shape = input_image.shape[1:]
        prediction_shape = [n_channels] + list(spatial_shape)
        
        # Create prediction arrays
        predicted_logits = torch.zeros(prediction_shape, dtype=torch.float32, device='cpu' if not self.perform_everything_on_device else self.device)
        n_predictions = torch.zeros(prediction_shape, dtype=torch.float32, device='cpu' if not self.perform_everything_on_device else self.device)
        
        # Compute step size
        step_size = [max(1, int(np.round(i * tile_step_size))) for i in self.configuration_manager.patch_size]
        
        # Generate slicers for sliding window
        slicers = []
        for d in range(len(spatial_shape)):
            positions = list(range(0, spatial_shape[d], step_size[d]))
            if positions[-1] < spatial_shape[d] - self.configuration_manager.patch_size[d]:
                positions.append(spatial_shape[d] - self.configuration_manager.patch_size[d])
            
            slicers.append([slice(pos, pos + self.configuration_manager.patch_size[d]) for pos in positions])
        
        # Import necessary functions
        from itertools import product
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        
        # Process each patch
        for slicers_batch in product(*slicers):
            # Extract patch
            patch_slice = [slice(None)] + list(slicers_batch)  # Add channel dimension
            data_patch = input_image[tuple(patch_slice)]
            
            if not isinstance(data_patch, torch.Tensor):
                data_patch = torch.from_numpy(data_patch)
            
            data_patch = data_patch[None]  # Add batch dimension
            
            # Predict on patch
            with torch.no_grad():
                if self.perform_everything_on_device:
                    data_patch = data_patch.to(self.device)
                    
                # Forward pass through network
                prediction = self.network(data_patch)[0]  # Remove batch dimension
                
                # Handle mirroring if needed
                if len(mirror_axes) > 0:
                    # Add mirrored predictions (simplified version)
                    for axis in mirror_axes:
                        mirrored = torch.flip(data_patch, dims=(axis + 1,))  # +1 for batch dimension
                        mirrored_pred = torch.flip(self.network(mirrored)[0], dims=(axis,))
                        prediction = (prediction + mirrored_pred) / 2
                
                # Move to target device
                if not self.perform_everything_on_device:
                    prediction = prediction.cpu()
                
                # Add to accumulated prediction
                target_slice = [slice(None)] + list(slicers_batch)
                predicted_logits[tuple(target_slice)] += prediction * gaussian
                n_predictions[tuple(target_slice)] += gaussian
        
        # Normalize by number of predictions
        predicted_logits = predicted_logits / (n_predictions + 1e-8)
        
        return predicted_logits

    def convert_predicted_logits_to_segmentation_with_correct_shape(self, predicted_logits, plans_manager,
                                                                   configuration_manager, label_manager,
                                                                   properties_dict, save_probabilities=False):
        """Override to handle 2-channel binary format without conversion to 3-channel."""
        # Apply sigmoid to get probabilities
        predicted_probabilities = torch.sigmoid(predicted_logits.to('cpu'))
        
        # Convert to binary segmentation (threshold at 0.5)
        segmentation = (predicted_probabilities > 0.5).long()
        
        # Return both probability maps and segmentation in 2-channel format
        # This matches the ground truth format: each channel is independent binary mask
        return segmentation.numpy(), predicted_probabilities.numpy() if save_probabilities else None


class MultiLabelHead(nn.Module):
    """Multi-label head that produces independent binary predictions for each hemisphere."""
    def __init__(self, original_conv):
        super().__init__()
        
        # Store original properties
        in_features = original_conv.in_channels
        kernel_size = original_conv.kernel_size
        padding = original_conv.padding
        has_bias = original_conv.bias is not None
        
        # Create multi-label head - single conv layer outputting 3 channels
        # Channel 0: background, Channel 1: left hemisphere, Channel 2: right hemisphere
        # This makes it compatible with standard nnUNet validation pipeline
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
        # Output: (batch, 2, H, W) where:
        # Channel 0: left hemisphere logits  
        # Channel 1: right hemisphere logits
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
        
        # Base losses: BCE + Dice for 2-channel output
        dice_loss = self.dice(net_output, target) if self.weight_dice != 0 else 0
        
        if self.weight_ce != 0:
            # Use sigmoid on output for BCE loss
            bce_loss_ch0 = self.bce(net_output[:, 0], target[:, 0])  # Left hemisphere
            bce_loss_ch1 = self.bce(net_output[:, 1], target[:, 1])  # Right hemisphere
            bce_loss = 0.5 * bce_loss_ch0 + 0.5 * bce_loss_ch1
        else:
            bce_loss = 0
        
        # Enhanced consistency losses
        # Convert logits to probabilities for consistency losses
        pred_probs = torch.sigmoid(net_output)
        
        # Spatial consistency loss
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
        
        # Use standard label manager - no override needed for 3-channel output
        # This will work with standard nnUNet validation pipeline
        
        # Set max epochs for quick test run
        self.num_epochs = 10
        
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
        print("Training for 30 epochs for quick test run with optimized validation")
        
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
        """Initialize network with standard 3-channel multi-label output."""
        # Use standard initialization - no label manager override needed
        super().initialize_network()
        
        
    def perform_actual_validation(self, save_probabilities: bool = False):
        """Multi-label validation using original nnUNet approach with minimal modifications."""
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
        from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
        from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
        from nnunetv2.configuration import default_num_processes
        import multiprocessing
        import torch
        import warnings
        from time import sleep
        
        # This is copied from the original nnUNet trainer with minimal multi-label modifications
        
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        # Use our custom predictor that natively handles 2-channel multi-label output
        predictor = MultiLabelPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                       perform_everything_on_device=True, device=self.device, verbose=False,
                                       verbose_preprocessing=False, allow_tqdm=False)
        
        # Standard initialization - our custom predictor handles the 2->3 channel conversion internally
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                import torch.distributed as dist
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []

            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)

                # we do [:] to convert blosc2 to numpy
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                # Get prediction from our custom predictor (automatically converts 2->3 channels)
                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(self.preprocessed_dataset_folder_base, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        # next stage may have a different dataset class, do not use self.dataset_class
                        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
                        dataset_name = maybe_convert_to_dataset_name(self.plans_manager.dataset_name)
                        from nnunetv2.dataset_conversion.generate_dataset_json import infer_dataset_class
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!")
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        from nnunetv2.inference.export_prediction import resample_and_save
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class),
                            )
                        ))
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < len(dataset_val.identifiers) - 1 and (i + 1) % 20 == 0:
                    import torch.distributed as dist
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            import torch.distributed as dist
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

        self.set_deep_supervision_enabled(True)
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        compute_gaussian.cache_clear()
