#!/bin/bash
# Setup script for nnUNet environment

# Set environment variables
export nnUNet_raw="/home/ubuntu/DLSegPerf/data/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/DLSegPerf/data/nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/DLSegPerf/data/nnUNet_results"

# Activate virtual environment (if not already activated)
if [[ "$VIRTUAL_ENV" != "/home/ubuntu/DLSegPerf/venv" ]]; then
    source /home/ubuntu/DLSegPerf/venv/bin/activate
fi

echo "Environment setup complete!"
echo "nnUNet environment variables:"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo "You can now run nnUNet commands directly."