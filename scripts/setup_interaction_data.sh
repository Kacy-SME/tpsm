#!/bin/bash

#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00
#SBATCH --mem=20G

module load mamba/latest
source activate tpsm-wasps

# create data and p_dominula directories
echo "Create interaction directories under assets"
mkdir -p assets/interaction assets/interaction_720 assets/interaction_256

# Check if the zip file is already downloaded
if [[ ! -f assets/interaction.zip ]]; then
  cd assets
  # check if interaction_url.txt exists
  if [[ ! -f interaction_url.txt ]]; then
    echo "Error: assets/interaction_url.txt not found!"
    exit 1
  fi

  # download p_dominula data
  url=$(head -n1 interaction_url.txt)
  wget -O interaction.zip ${url}
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to download interaction.zip"
    exit 1
  fi
  cd ..
else
  echo "interaction.zip already exists, skipping download"
fi

# Unzip only if not already unzipped
if [[ ! -d assets/interaction || -z "$(ls -A assets/interaction)" ]]; then
  echo "Unzipping interaction.zip"
  cd assets
  7za x interaction.zip
  cd ..
else
  echo "Files are already unzipped, skipping unzip step"
fi

# Rename all .MP4 files to .mp4
echo "Renaming .MP4 files to .mp4"
for file in assets/interaction/*.MP4; do
    if [ -f "$file" ]; then
        mv "$file" "${file%.MP4}.mp4"
    fi
done

# Rename all .MOV files to .mp4 (if any)
for file in assets/interaction/*.MOV; do
    if [ -f "$file" ]; then
        mv "$file" "${file%.MOV}.mp4"
    fi
done

# Crop videos to 720x720 if not already done
if [[ -z "$(ls -A assets/interaction_720/*.MP4 2>/dev/null)" ]]; then
  echo "Cropping videos to 720x720"
  for video in $(ls assets/interaction/*.{mp4,MP4} 2>/dev/null); do
    resolution=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "${video}")
    width=$(echo $resolution | cut -d'x' -f1)
    height=$(echo $resolution | cut -d'x' -f2)

    # Since the video is 1280x720, we crop to 720x720 based on height
    if (( width >= 720 && height >= 720 )); then
      echo "Crop $(basename ${video}) to square 720x720"
      ffmpeg -i ${video} -vf crop=720:720 -c:a copy assets/interaction_720/$(basename ${video}) -hide_banner -loglevel error
    else
      echo "Skipping $(basename ${video}), resolution too small: ${width}x${height}"
    fi
  done
else
  echo "720x720 cropped videos already exist, skipping cropping step"
fi

# Resize videos to 256x256 if not already done
if [[ -z "$(ls -A assets/interaction_256/*.MP4 2>/dev/null)" ]]; then
  echo "Resizing videos to 256x256"
  for video in $(ls assets/interaction_720/*.{mp4,MP4} 2>/dev/null); do
    echo "Resize $(basename ${video}) to 256x256"
    ffmpeg -i ${video} -s 256x256 -c:a copy assets/interaction_256/$(basename ${video}) -hide_banner -loglevel error
  done
else
  echo "256x256 resized videos already exist, skipping resizing step"
fi

