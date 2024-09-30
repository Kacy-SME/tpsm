#!/bin/bash

#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

module load mamba/latest
source activate tpsm-wasps

echo "Running the demo with the wasp-sol-interaction checkpoint and the interaction source and driving video."

python demo.py --config config/wasp-sol-interaction.yaml --checkpoint checkpoints/00000099-checkpoint.pth.tar --source_image assets/interaction-source.png --driving_video assets/interaction-driving.mp4

