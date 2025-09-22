#!/bin/bash
# GPU to use
export CUDA_VISIBLE_DEVICES='0'

#Path to repo direcotry
export REPO_DIR=#'/data'

# Path to your local nnUNetv2 installation and dependencies (change as needed)
export PATH=#"/home/username/.local/bin:${PATH}"
export LD_LIBRARY_PATH=#"/home/username/.local/lib:${LD_LIBRARY_PATH}"

# Path to model weights (this repo)
export nnUNet_results=#"$REPO_DIR/PulmonaryEmbolismSegmentation/dataset/nnUNet_results"

# Dataset + Trainer + Plans identifiers
DATASET_ID=256
PLAN_NAME='nnUNetResEncUNetPlans_80G'
TRAINER='nnUNetTrainer_500epochs'

# Input and output paths 
INPUT_NIFTI_PTH= # Path to the directory containing the nifti files
SAVE_ROOT= #

# Run nnU-Net v2 prediction
nnUNetv2_predict \
    -i $INPUT_NIFTI_PTH \
    -d $DATASET_ID \
    -o $SAVE_ROOT \
    -c 3d_fullres \
    -f all \
    --verbose \
    -tr $TRAINER \
    -p $PLAN_NAME \
    --save_probabilities
