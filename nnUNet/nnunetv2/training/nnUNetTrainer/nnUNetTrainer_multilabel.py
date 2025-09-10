import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Union, Tuple
import json
import nibabel as nib

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

import warnings
# Quiet PyTorch's NVML availability warning (benign; only about GPU stats)
warnings.filterwarnings(
    "ignore",
    message="Can't initialize NVML",
    module=r"torch\.cuda(\..*)?$"
)


class MultiLabelPredictor(nnUNetPredictor):
    """Custom predictor that natively handles 2-channel binary multi-label output."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _internal_predict_sliding_window_return_logits(self, data, slicers, do_on_device: bool):
        """Override to handle 2-channel predictions by creating a mock label manager."""
        class MockLabelManager:
            def __init__(self):
                self.num_segmentation_heads = 2

        original_label_manager = getattr(self, 'label_manager', None)
        self.label_manager = MockLabelManager()

        try:
            return super()._internal_predict_sliding_window_return_logits(data, slicers, do_on_device)
        finally:
            self.label_manager = original_label_manager

    def convert_predicted_logits_to_segmentation_with_correct_shape(
        self, predicted_logits, plans_manager, configuration_manager, label_manager,
        properties_dict, save_probabilities: bool = False
    ):
        """Handle 2-channel binary format and ensure correct spatial dimensions."""
        predicted_probabilities = torch.sigmoid(predicted_logits.to('cpu'))
        segmentation = (predicted_probabilities > 0.5).long()

        if len(segmentation.shape) == 4:  # (2, D, H, W)
            segmentation = segmentation.permute(1, 2, 3, 0)  # -> (D,H,W,2)
            if save_probabilities:
                predicted_probabilities = predicted_probabilities.permute(1, 2, 3, 0)

        return segmentation.numpy(), (predicted_probabilities.numpy() if save_probabilities else None)


class MultiLabelHead(nn.Module):
    """Multi-label head that produces independent binary predictions for each hemisphere."""
    def __init__(self, original_conv):
        super().__init__()
        in_features = original_conv.in_channels
        kernel_size = original_conv.kernel_size
        padding = original_conv.padding
        has_bias = original_conv.bias is not None

        self.multi_label_head = nn.Conv2d(in_features, 2, kernel_size=kernel_size,
                                          padding=padding, bias=has_bias)

        if original_conv.out_channels == 2:
            with torch.no_grad():
                self.multi_label_head.weight.copy_(original_conv.weight)
                if has_bias:
                    self.multi_label_head.bias.copy_(original_conv.bias)

    def forward(self, x):
        return self.multi_label_head(x)


class SharedDecoderNetwork(nn.Module):
    """
    Network with shared encoder and decoder, but multi-label head for independent predictions.
    """
    def __init__(self, base_network):
        super().__init__()
        self.base_network = base_network
        if hasattr(base_network, 'decoder'):
            self.decoder = base_network.decoder
        self._replace_final_conv()

    def _find_and_replace_final_conv(self, module, parent=None, name=None):
        for child_name, child in module.named_children():
            if hasattr(child, 'out_channels') and child.out_channels == 2:
                setattr(module, child_name, MultiLabelHead(child))
                return True
            if self._find_and_replace_final_conv(child, module, child_name):
                return True
        return False

    def _replace_final_conv(self):
        if not self._find_and_replace_final_conv(self.base_network):
            raise RuntimeError("Could not find final convolution layer with 2 output channels")

    def forward(self, x):
        return self.base_network(x)


class SpatialComplementarySharedDecoderLoss(nn.Module):
    """
    BCE + Soft Dice + spatial consistency + complementary coverage loss.
    """
    def __init__(
        self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1,
        spatial_weight=0.1, complementary_weight=0.1, dice_class=MemoryEfficientSoftDiceLoss
    ):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.spatial_weight = spatial_weight
        self.complementary_weight = complementary_weight

        self.bce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dice = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def _spatial_consistency_loss(self, pred_probs: torch.Tensor, target: torch.Tensor):
        pred_left = pred_probs[:, 0]
        pred_right = pred_probs[:, 1]
        target_left = target[:, 0]
        target_right = target[:, 1]

        non_overlap_mask = (target_left + target_right) <= 1.0
        overlap_penalty = torch.mean(pred_left * pred_right * non_overlap_mask.float())

        target_coverage = torch.clamp(target_left + target_right, 0, 1)
        pred_coverage = torch.clamp(pred_left + pred_right, 0, 1)
        coverage_loss = F.mse_loss(pred_coverage, target_coverage)

        left_only_regions = (target_left > 0) & (target_right == 0)
        right_only_regions = (target_right > 0) & (target_left == 0)

        exclusivity_loss = 0
        if left_only_regions.sum() > 0:
            exclusivity_loss += torch.mean(pred_right[left_only_regions] ** 2)
        if right_only_regions.sum() > 0:
            exclusivity_loss += torch.mean(pred_left[right_only_regions] ** 2)

        return overlap_penalty + coverage_loss + exclusivity_loss

    def _complementary_loss(self, pred_probs: torch.Tensor, target: torch.Tensor):
        pred_left = pred_probs[:, 0]
        pred_right = pred_probs[:, 1]
        target_left = target[:, 0]
        target_right = target[:, 1]

        exclusivity = torch.mean(pred_left * pred_right * (1 - target_left) * (1 - target_right))
        cov_t = torch.clamp(target_left + target_right, 0, 1)
        cov_p = torch.clamp(pred_left + pred_right, 0, 1)
        cov_loss = F.mse_loss(cov_p, cov_t)
        return exclusivity + cov_loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        target = target.float()

        dice_loss = self.dice(net_output, target) if self.weight_dice != 0 else 0.0
        if self.weight_ce != 0:
            bce0 = self.bce(net_output[:, 0], target[:, 0])
            bce1 = self.bce(net_output[:, 1], target[:, 1])
            bce_loss = 0.5 * (bce0 + bce1)
        else:
            bce_loss = 0.0

        pred_probs = torch.sigmoid(net_output)
        spatial_loss = self._spatial_consistency_loss(pred_probs, target) if self.spatial_weight > 0 else 0.0
        complementary_loss = self._complementary_loss(pred_probs, target) if self.complementary_weight > 0 else 0.0

        total = (
            self.weight_ce * bce_loss
            + self.weight_dice * dice_loss
            + self.spatial_weight * spatial_loss
            + self.complementary_weight * complementary_loss
        )
        return total


class nnUNetTrainer_multilabel(nnUNetTrainer):
    """Trainer with shared decoder and spatial/complementary loss for 2-channel multi-label output."""

    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10
        self.spatial_weight = 0.1
        self.complementary_weight = 0.1

        _ = len(self.dataset_json.get('channel_names', {'0': 'CBF LICA', '1': 'CBF RICA'}))

    @property
    def num_segmentation_heads(self):
        return 2

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str, arch_init_kwargs: dict, arch_init_kwargs_req_import: Union[List[str], Tuple[str]],
        num_input_channels: int, num_output_channels: int, enable_deep_supervision: bool = True
    ) -> nn.Module:
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, 2, enable_deep_supervision
        )
        return SharedDecoderNetwork(base_network)

    def _build_loss(self):
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

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else torch.no_grad():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        probs = torch.sigmoid(output)
        pred_onehot = (probs > 0.5).long()
        target = target.float()

        tp_hard = torch.zeros((target.shape[0], 2), dtype=torch.float32, device=target.device)
        fp_hard = torch.zeros((target.shape[0], 2), dtype=torch.float32, device=target.device)
        fn_hard = torch.zeros((target.shape[0], 2), dtype=torch.float32, device=target.device)

        for c in range(2):
            spatial_axes = tuple(range(1, target[:, c].ndim))
            tp_hard[:, c] = torch.sum((pred_onehot[:, c] == 1) & (target[:, c] == 1), dim=spatial_axes)
            fp_hard[:, c] = torch.sum((pred_onehot[:, c] == 1) & (target[:, c] == 0), dim=spatial_axes)
            fn_hard[:, c] = torch.sum((pred_onehot[:, c] == 0) & (target[:, c] == 1), dim=spatial_axes)

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard.detach().cpu().numpy(),
            'fp_hard': fp_hard.detach().cpu().numpy(),
            'fn_hard': fn_hard.detach().cpu().numpy(),
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Compute and log per-channel Dice scores."""
        outputs_collated = collate_outputs(val_outputs)

        tp = np.sum(outputs_collated['tp_hard'], axis=(0, 1))
        fp = np.sum(outputs_collated['fp_hard'], axis=(0, 1))
        fn = np.sum(outputs_collated['fn_hard'], axis=(0, 1))

        dice_left  = (2.0 * float(tp[0])) / (2.0 * float(tp[0]) + float(fp[0]) + float(fn[0]) + 1e-8)
        dice_right = (2.0 * float(tp[1])) / (2.0 * float(tp[1]) + float(fp[1]) + float(fn[1]) + 1e-8)
        mean_dice  = 0.5 * (dice_left + dice_right)  # average of hemispheres

        self.logger.log('dice_per_class_or_region', [dice_left, dice_right], self.current_epoch)
        self.logger.log('mean_fg_dice', float(mean_dice), self.current_epoch)

        val_loss = float(np.mean([o['loss'] for o in val_outputs]))
        self.logger.log('val_losses', val_loss, self.current_epoch)

        if 'val_scores' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['val_scores'] = []
        self.logger.my_fantastic_logging['val_scores'].append(-mean_dice)
        return mean_dice

    def export_multilabel_prediction(
        self, predicted_logits, properties_dict, configuration_manager, plans_manager,
        dataset_json, output_filename_truncated, save_probabilities: bool = False
    ):
        """Custom export function for 2-channel multi-label predictions."""
        import torch as _torch
        import numpy as _np
        from nnunetv2.imageio.base_reader_writer import BaseReaderWriter  # noqa: F401
        from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name  # noqa: F401
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        from nnunetv2.configuration import default_num_processes as _def_np

        old_threads = _torch.get_num_threads()
        _torch.set_num_threads(_def_np)

        pred_probs = torch.sigmoid(predicted_logits.to('cpu'))
        seg = (pred_probs > 0.5).long()  # (2,D,H,W)

        if len(seg.shape) == 4 and seg.shape[0] == 2:
            seg = seg.permute(1, 2, 3, 0)  # -> (D,H,W,2)
            if save_probabilities:
                pred_probs = pred_probs.permute(1, 2, 3, 0)

        seg = seg.numpy() if isinstance(seg, torch.Tensor) else seg

        spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
        current_spacing = configuration_manager.spacing if \
            len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *configuration_manager.spacing]

        target_shape_3d = properties_dict['shape_after_cropping_and_before_resampling']
        if tuple(seg.shape[:3]) != tuple(target_shape_3d):
            resampled = []
            for ch in range(seg.shape[3]):
                resampled_ch = configuration_manager.resampling_fn_seg(
                    seg[:, :, :, ch],
                    target_shape_3d,
                    current_spacing,
                    [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
                )
                resampled.append(resampled_ch)
            seg = np.stack(resampled, axis=-1)

        if save_probabilities and tuple(pred_probs.shape[:3]) != tuple(target_shape_3d):
            prob_channels = []
            for ch in range(pred_probs.shape[3]):
                prob_ch = configuration_manager.resampling_fn_probabilities(
                    pred_probs[:, :, :, ch],
                    target_shape_3d,
                    current_spacing,
                    [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
                )
                prob_channels.append(prob_ch)
            pred_probs = np.stack(prob_channels, axis=-1)

        target_shape_3d = properties_dict['shape_before_cropping']
        target_shape_4d = list(target_shape_3d) + [seg.shape[-1]]
        seg_rev_crop = np.zeros(target_shape_4d, dtype=np.uint8)

        if len(seg.shape) == 4:
            for ch in range(seg.shape[-1]):
                tgt3d = np.zeros(target_shape_3d, dtype=np.uint8)
                tgt3d = insert_crop_into_image(tgt3d, seg[:, :, :, ch], properties_dict['bbox_used_for_cropping'])
                seg_rev_crop[:, :, :, ch] = tgt3d
        else:
            seg_rev_crop = insert_crop_into_image(seg_rev_crop, seg, properties_dict['bbox_used_for_cropping'])

        if len(seg_rev_crop.shape) == 4:
            seg_rev_crop = seg_rev_crop.transpose(1, 2, 0, 3)          # (H,W,D,C)
            seg_rev_crop = np.flip(seg_rev_crop, axis=(0, 1))          # flip both in-plane axes
            seg_rev_crop = np.flip(np.rot90(seg_rev_crop, k=3, axes=(0, 1)), axis=0)  # rotate CW then vertical flip
        else:
            transpose_indices = list(plans_manager.transpose_backward) + [len(seg_rev_crop.shape) - 1]
            seg_rev_crop = seg_rev_crop.transpose(transpose_indices)

        affine = np.eye(4)
        if 'sitk_stuff' in properties_dict and 'spacing' in properties_dict:
            spacing = properties_dict['spacing']
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]

        nib.save(nib.Nifti1Image(seg_rev_crop.astype(np.uint8), affine), output_filename_truncated + '.nii')

        if save_probabilities:
            prob_target_shape_3d = properties_dict['shape_before_cropping']
            prob_target_shape_4d = list(prob_target_shape_3d) + [pred_probs.shape[-1]]
            prob_rev_crop = np.zeros(prob_target_shape_4d, dtype=np.float32)

            if len(pred_probs.shape) == 4:
                for ch in range(pred_probs.shape[-1]):
                    tgt3d = np.zeros(prob_target_shape_3d, dtype=np.float32)
                    tgt3d = insert_crop_into_image(tgt3d, pred_probs[:, :, :, ch], properties_dict['bbox_used_for_cropping'])
                    prob_rev_crop[:, :, :, ch] = tgt3d
            else:
                prob_rev_crop = insert_crop_into_image(prob_rev_crop, pred_probs, properties_dict['bbox_used_for_cropping'])

            if len(prob_rev_crop.shape) == 4:
                prob_rev_crop = prob_rev_crop.transpose(1, 2, 0, 3)
                prob_rev_crop = np.flip(prob_rev_crop, axis=(0, 1))
            else:
                prob_transpose_indices = list(plans_manager.transpose_backward) + [len(prob_rev_crop.shape) - 1]
                prob_rev_crop = prob_rev_crop.transpose(prob_transpose_indices)

            np.savez_compressed(output_filename_truncated + '.npz', probabilities=prob_rev_crop)

        _torch.set_num_threads(old_threads)

    def initialize_network(self):
        """Initialize network with standard 2-channel multi-label output."""
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

        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file(
                "WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                "encounter crashes in validation then this is because torch.compile forgets "
                "to trigger a recompilation of the model with deep supervision disabled. "
                "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                "validation with --val (exactly the same as before) and then it will work. "
                "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                "forward pass (where compile is triggered) already has deep supervision disabled. "
                "This is exactly what we need in perform_actual_validation"
            )

        predictor = MultiLabelPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network, self.plans_manager, self.configuration_manager, None,
            self.dataset_json, self.__class__.__name__, self.inference_allowed_mirroring_axes
        )

        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            _, val_keys = self.do_split()
            if self.is_ddp:
                import torch.distributed as dist
                _ = len(val_keys) // dist.get_world_size() - 1  # noqa: F841
                val_keys = val_keys[self.local_rank::dist.get_world_size()]

            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder, val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
            )

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = []
            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results, allowed_num_queued=2)

                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]

                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data).cpu()

                results.append(
                    segmentation_export_pool.starmap_async(
                        self.export_multilabel_prediction,
                        ((prediction, properties, self.configuration_manager, self.plans_manager,
                          self.dataset_json, output_filename_truncated, save_probabilities),)
                    )
                )

                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(
                            self.preprocessed_dataset_folder_base, self.plans_manager.dataset_name,
                            next_stage_config_manager.data_identifier
                        )
                        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
                        _ = maybe_convert_to_dataset_name(self.plans_manager.dataset_name)  # noqa: F841
                        from nnunetv2.dataset_conversion.generate_dataset_json import infer_dataset_class
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)

                        try:
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
                        results.append(
                            segmentation_export_pool.starmap_async(
                                resample_and_save,
                                ((prediction, target_shape, output_file_truncated, self.plans_manager,
                                  self.configuration_manager, properties, self.dataset_json,
                                  default_num_processes, dataset_class),)
                            )
                        )
                if self.is_ddp and i < len(dataset_val.identifiers) - 1 and (i + 1) % 20 == 0:
                    import torch.distributed as dist
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            import torch.distributed as dist
            dist.barrier()

        if self.local_rank == 0:
            # Keep nnU-Net's default summary for compatibility
            summary_path = join(validation_output_folder, 'summary.json')
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                validation_output_folder,
                summary_path,
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels,
                self.label_manager.ignore_label, chill=True,
                num_processes=default_num_processes
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)

            # ---- Multi-label: Left / Right / Overall (mean of hemispheres) ----
            def _counts(pred_bool, gt_bool):
                tp = int(np.count_nonzero(pred_bool & gt_bool))
                fp = int(np.count_nonzero(pred_bool & (~gt_bool)))
                fn = int(np.count_nonzero((~pred_bool) & gt_bool))
                return tp, fp, fn

            def _dice(tp, fp, fn):
                return (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

            val_dir = validation_output_folder
            gt_dir = join(self.preprocessed_dataset_folder_base, 'gt_segmentations')
            case_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(val_dir) if f.endswith('.nii'))

            sumL = {'tp': 0, 'fp': 0, 'fn': 0}
            sumR = {'tp': 0, 'fp': 0, 'fn': 0}

            for k in case_ids:
                pred_nii = nib.load(join(val_dir, f"{k}.nii"))
                gt_nii   = nib.load(join(gt_dir, f"{k}.nii"))
                pred = pred_nii.get_fdata()
                gt   = gt_nii.get_fdata()

                if pred.ndim == 3:  # fallback
                    pred = np.stack([pred, pred], axis=-1)
                if gt.ndim == 3:
                    gt = np.stack([gt, gt], axis=-1)

                pred = pred > 0.5
                gt   = gt > 0.5

                tpL, fpL, fnL = _counts(pred[..., 0], gt[..., 0])
                tpR, fpR, fnR = _counts(pred[..., 1], gt[..., 1])

                sumL['tp'] += tpL; sumL['fp'] += fpL; sumL['fn'] += fnL
                sumR['tp'] += tpR; sumR['fp'] += fpR; sumR['fn'] += fnR

            dice_left  = _dice(sumL['tp'], sumL['fp'], sumL['fn'])
            dice_right = _dice(sumR['tp'], sumR['fp'], sumR['fn'])
            dice_overall = 0.5 * (dice_left + dice_right)  # *** requested definition ***

            # Pretty log output
            self.print_to_log_file(
                "\nValidation (multi-label, hemispheres):\n"
                f"  Left   Dice (ch0): {dice_left:.4f}\n"
                f"  Right  Dice (ch1): {dice_right:.4f}\n"
                f"  Overall Dice (mean of ch0 & ch1): {dice_overall:.4f}\n",
                also_print_to_console=True
            )

            # Ensure logger keys exist (nnUNet logger expects lists)
            for key in ("val_dice_left", "val_dice_right", "val_dice_overall"):
                if key not in self.logger.my_fantastic_logging:
                    self.logger.my_fantastic_logging[key] = []

            # Log scalars
            self.logger.log("val_dice_left",  float(dice_left),  self.current_epoch)
            self.logger.log("val_dice_right", float(dice_right), self.current_epoch)
            self.logger.log("val_dice_overall", float(dice_overall), self.current_epoch)

            # Update nnU-Net's summary.json with multilabel section
            try:
                with open(summary_path, "r") as f:
                    summary_obj = json.load(f)
            except Exception:
                summary_obj = {}

            summary_obj["_multilabel"] = {
                "dice_left": float(dice_left),
                "dice_right": float(dice_right),
                "overall_dice_mean_hemispheres": float(dice_overall)
            }
            with open(summary_path, "w") as f:
                json.dump(summary_obj, f, indent=2)

        self.set_deep_supervision_enabled(True)
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        compute_gaussian.cache_clear()