#!/bin/bash

#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00
#SBATCH --mem=20G

module load mamba/latest
source activate tpsm-wasps

# create data and p_dominula directories
echo "Create p_dominula directories under assets"
mkdir -p assets/p_dominula assets/p_dominula_1080 assets/p_dominula_256
cd assets/p_dominula

# download p_dominula data
url=$(head -n1 assets/pdominula_url.txt)
wget -O p_dominula.zip ${url}

# unzip p_dominula data
echo "Unzip p_dominula data"
7za x p_dominula.zip 

# crop videos to be square
cd ..
for video in $(ls p_dominula/*.mp4); do echo "Crop $(basename ${video}) to square 1080x1080"; ffmpeg -i ${video} -vf crop=1080:1080:420:0 -c:a copy p_dominula_1080/$(basename ${video}) -hide_banner -loglevel error; done

# resize videos to be 256x256
for video in $(ls p_dominula_1080/*.mp4); do echo "Resize $(basename ${video}) to 256x256"; ffmpeg -i ${video} -s 256x256 -c:a copy p_dominula_256/$(basename ${video}) -hide_banner -loglevel error; done

