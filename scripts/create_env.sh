#!/bin/bash

#SBATCH --job-name=tpsm_env
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --mem=4G

module load mamba/latest

echo "Creating the tpsm-wasps environment."
mamba create -y -f environment.yml
