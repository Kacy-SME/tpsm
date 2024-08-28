#!/bin/bash

#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00
#SBATCH --gres=gpu:1

module load mamba/latest
source activate tpsm-wasps

echo "Running the demo with the vox-256 checkpoint and the wasp source image and driving video."
echo "Ensure the vox-256.pht.tar checkpoint is in the checkpoints directory"
echo " (instructions https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)." 
python demo.py --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image assets/source.png --driving_video assets/driving.mp4
