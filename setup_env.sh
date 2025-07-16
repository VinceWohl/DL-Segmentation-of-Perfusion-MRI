#!/bin/bash
# Setup script for nnUNet environment

# Set environment variables
export nnUNet_raw="/home/ubuntu/DLSegPerf/data/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/DLSegPerf/data/nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/DLSegPerf/data/nnUNet_results"

# Activate virtual environment
source /home/ubuntu/DLSegPerf/venv/bin/activate

echo "Environment setup complete!"
echo "You can now run nnUNet commands directly."