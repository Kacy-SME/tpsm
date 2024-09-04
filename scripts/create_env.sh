#!/bin/bash

#SBATCH --job-name=tpsm_env
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=8
#SBATCH --time=0-2
#SBATCH --mem=20G

module load mamba/latest

echo "Creating the tpsm-wasps environment."
conda env create -f reqs.yaml
