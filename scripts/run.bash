#!/bin/bash

# Launch an sbatch job to train or do inference with the TPSM model

# Function to launch sbatch script
launch_sbatch() {
    job_type=$1
    shift
    sbatch_args="$@"
    
    # Choose the sbatch script based on job type
    if [ "$job_type" == "train" ]; then
        sbatch_script="scripts/train.bash"
        echo "Enter the path to the YAML configuration file:"
        read -p "> " -a args

    elif [ "$job_type" == "inference" ]; then
        sbatch_script="scripts/inference.bash"
        echo "Enter the following separated by a space:"
        echo "(1) path to the YAML configuration file"
        echo "(2) model checkpoint file"
        echo "(3) source image file"
        echo "(4) driving video file"
        read -p "> " -a args
    else
        echo "Invalid job type selected."
        exit 1
    fi

    if [[ -z "${sbatch_args// }" ]]; then
        sbatch "${sbatch_script}" "${args[@]}"
    else
        sbatch "${sbatch_args[@]}" "${sbatch_script}" "${args[@]}"
    fi
}

# Prompt user for the type of job
echo "Do you want to train or do inference? (type 'train' or 'inference'):"
read -p "> " job_type

# Prompt user for additional arguments
echo "Please enter any sbatch additional arguments (e.g., --mem=100G --time=01:00)."
echo "Press Enter to use the default values."
read -r -p "> " args

# Launch the appropriate sbatch script with the provided arguments
launch_sbatch "${job_type}" "${args}"

