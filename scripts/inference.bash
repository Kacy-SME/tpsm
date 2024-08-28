#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --cpus-per-task=48
#SBATCH --time=30:00
#SBATCH --gres=gpu:a100:1

if [[ $# -ne 4 ]]; then
    echo "Icorrect number of parameters" >&2
	echo " config.yaml checkpoint source-image driving-video"
    exit 2
fi

CONFIG=$1
CHECKPOINT=$2
SOURCE=$3
DRIVING=$4

module load mamba/latest
source activate tpsm-wasps

echo "Inference using:"
echo -e "\tCONFIG="${CONFIG}""
echo -e "\tCHECKPOINT="${CHECKPOINT}""
echo -e "\tSOURCE="${SOURCE}""
echo -e "\tDRIVING="${DRIVING}"\n"

python demo.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --source_image "${SOURCE}" --driving_video "${DRIVING}"

