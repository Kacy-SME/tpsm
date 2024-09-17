# Documentation for ASU's Sol Supercomputer

## Table of Contents


## Install on ASU-Sol

### Step 1: Clone the repository
```bash
git clone -b asu-sol git@github.com:Kacy-SME/tpsm.git
cd tpsm
```

### Step 2: Create a mamba environment
This will create a new mamba environment `tpsm-wasps` with the necessary dependencies.
```bash
sbatch scripts/create_env.sh
```

## Run on ASU-Sol

### Run the demo
This assumes the `checkpoints/vox.pth.tar` model checkpoint is available, together with
the `config/vox.yml` configuration file. You can find these files as part of
[the original TPSM model](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model).
```bash
sbatch scripts/run_demo.sh
```

### Download and preprocess the *P dominula* dataset
To download and convert the *P dominula* dataset to 256x256, run the command below.
The dataset will be downloaded to `assets/p_dominula` and the data ready to be used
by the model will be saved in `assets/p_dominula_256`.
```bash
sbatch scripts/setup_pdominula_data.sh
```

### Training the model and inference
Run the `scripts/run.bash` command as follows to train the model and perform inference.
```bash
bash scripts/run.bash
```

The bash program will ask for optional `sbatch` arguments if you wish to override the
defaults and information for the training/inference.

Training:
- config file

Inference:
- config file
- model checkpoint file
- input image
- driving video

#### Results
The resulting video for inference will be saved as a `result.mp4` file in the current directory.

For training, the model checkpoints will be saved into the `log` directory. It is recommended
to move the last checkpoint to the `checkpoints` directory for inference.

### Interactive mode
If you wish to run the program interactively to debug, test, or develop, you can request
an interactive session using the following syntax:
`interactive -p <partition> -q <qos> -t <time> -c <cores> --mem=<memory> --gres=gpu:[gpu_type:]<number>`.
Example:
```bash
interactive -p htc -q public -t 0-2 -c 48 --mem=120G --gres=gpu:a100:1
```

For more information about the available resources, sbatch options, the interactive command,
and more, please visit [ASU's Research Computing documentation website](links.asu.edu/docs).
