#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH -t 1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G  # Requesting 25GB of memory
#SBATCH -c 4

# Function to launch sbatch script
launch_sbatch() {
    job_type=$1
    shift
    sbatch_args="$@"

    # Choose the sbatch script based on job type
    if [ "$job_type" == "train" ]; then
        echo "Enter the following separated by a space:"
        echo "(1) path to the YAML configuration file"
        echo "(2) source image file"
        echo "(3) driving video file"
        echo "(4) HDF5 keypoints file"
        read -p "> " -a args

        sbatch_script="scripts/train_SLEAP.bash"
    elif [ "$job_type" == "inference" ]; then
        echo "Enter the following separated by a space:"
        echo "(1) path to the YAML configuration file"
        echo "(2) model checkpoint file"
        echo "(3) source image file"
        echo "(4) driving video file"
        read -p "> " -a args

        sbatch_script="scripts/inference_SLEAP.bash"
    else
        echo "Invalid job type selected."
        exit 1
    fi

    # Submit the job using the specified SBATCH script
    if [[ -z "${sbatch_args// }" ]]; then
        sbatch "${sbatch_script}" "${args[@]}"
    else
        # Correctly interpret sbatch_args as individual flags
        sbatch ${sbatch_args} "${sbatch_script}" "${args[@]}"
    fi
}

# Prompt user for the type of job
echo "Do you want to train or do inference? (type 'train' or 'inference'):"
read -p "> " job_type

# Prompt user for additional sbatch arguments (e.g., memory and time limits)
echo "Please enter any sbatch additional arguments (e.g., --mem=100G --time=01:00:00)."
echo "Press Enter to use the default values."
read -r -p "> " args

# Launch the appropriate sbatch script with the provided arguments
launch_sbatch "${job_type}" "${args[@]}"

