#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --cpus-per-task=48
#SBATCH --time=30:00
#SBATCH --gres=gpu:a100:1

# Check if the correct number of arguments are provided
if [[ $# -ne 5 ]]; then
    echo "Incorrect number of parameters" >&2
    echo "Usage: config.yaml checkpoint source-image driving-video keypoint-hdf5"
    exit 2
fi

# Assign the provided arguments to variables
CONFIG=$1
CHECKPOINT=$2
SOURCE=$3
DRIVING=$4
KEYPOINT_HDF5=$5  # New argument for the HDF5 keypoints file

# Load necessary modules and activate the conda environment
module load mamba/latest
source activate tpsm-wasps

# Display the arguments being used for inference
echo "Inference using:"
echo -e "\tCONFIG="${CONFIG}""
echo -e "\tCHECKPOINT="${CHECKPOINT}""
echo -e "\tSOURCE="${SOURCE}""
echo -e "\tDRIVING="${DRIVING}""
echo -e "\tKEYPOINT_HDF5="${KEYPOINT_HDF5}"\n"

# Run the demo.py script with the required arguments, including the keypoints HDF5 file
python demo-SLEAP.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --source_image "${SOURCE}" --driving_video "${DRIVING}" --keypoint_hdf5 "${KEYPOINT_HDF5}"

