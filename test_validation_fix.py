#!/usr/bin/env python3
"""
Quick test script to run 30 epochs with the fixed validation method
"""
import os
import sys
import torch

# Add nnUNet to path
sys.path.append('/home/ubuntu/DLSegPerf/nnUNet')

# Set environment variables
os.environ['nnUNet_raw'] = '/home/ubuntu/DLSegPerf/data/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/ubuntu/DLSegPerf/data/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/ubuntu/DLSegPerf/data/nnUNet_results'

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss import nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import join, load_json

def main():
    print("Testing fixed validation method with 30 epochs...")
    
    dataset_name_or_id = 'Dataset001_PerfusionTerritories'
    configuration = '2d'
    fold = 0
    
    # Load plans and dataset info
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_dataset_folder = join(os.environ['nnUNet_preprocessed'], dataset_name)
    dataset_json = load_json(join(preprocessed_dataset_folder, 'dataset.json'))
    plans = load_json(join(preprocessed_dataset_folder, 'nnUNetPlans.json'))
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = nnUNetTrainer_SharedDecoder_SpatialLoss_ComplementaryLoss(
        plans=plans,
        configuration=configuration, 
        fold=fold,
        dataset_json=dataset_json,
        device=device
    )
    
    # Override num_epochs to 30 for testing
    trainer.num_epochs = 30
    print(f"Set training to {trainer.num_epochs} epochs for validation testing")
    
    # Initialize and run training
    trainer.initialize()
    trainer.run_training()
    
    print("Training completed! Check the validation results.")

if __name__ == '__main__':
    main()