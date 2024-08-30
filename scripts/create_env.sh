#!/bin/bash

#SBATCH --job-name=tpsm_env
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --time=0-1
#SBATCH --mem=4G

module load mamba/latest

echo "Creating the tpsm-wasps environment."
conda env create -f reqs.yaml
