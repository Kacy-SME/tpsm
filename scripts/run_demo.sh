#!/bin/bash

#SBATCH -p general
#SBATCH -q public
#SBATCH --time=0-00:05:00
#SBATCH --gres=gpu:1g.10gb:1

module load mamba/latest
source activate tpsm

python demo.py --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image assets/source.png --driving_video assets/driving.mp4
