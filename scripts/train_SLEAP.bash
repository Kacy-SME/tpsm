#!/bin/bash
#SBATCH --job-name=tpsm_train         # Job name
#SBATCH --output=logs/tpsm_train_%j.out  # Log file
#SBATCH --error=logs/tpsm_train_%j.err   # Error log
#SBATCH --partition=general                # Specify partition
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G                      # Memory size
#SBATCH --time=24:00:00                # Maximum execution time

module load mamba/latest
source activate tpsm-wasps

# Run the training script
python run-SLEAP.py --config "$1" --mode train --source_image "$2" --driving_video "$3" --keypoint_hdf5 "$4"

