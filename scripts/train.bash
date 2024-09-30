#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --cpus-per-task=48
#SBATCH --time=1-0
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G


if [[ $# -ne 1 ]]; then
    echo "Speficy the YAML configuration file." >&2
    exit 2
fi

CONFIG=$1

module load mamba/latest
source activate tpsm-wasps

echo "Train model using:"
echo -e "\tCONFIG="${CONFIG}"\n"

# train tpsm model with using one GPU
python run.py --config ${CONFIG} --mode train
