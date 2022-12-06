#!/bin/bash -l

# Name of the job
#SBATCH --job-name=score_xrays

# Walltime (job duration)
#SBATCH --time=990:15:00

# #SBATCH --account=cairo
# #SBATCH --partition=v100_1_2
# #SBATCH --gres=gpu:v100
# #SBATCH --mem=40G

# Email address
#SBATCH --mail-user=Travis.T.Byrum.MED@Dartmouth.edu

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL

# STDOUT and STDERR
#SBATCH --output=/dartfs/rc/nosnapshots/H/HillB-scratch/OAI/score_xrays_%j.txt
#SBATCH --error=/dartfs/rc/nosnapshots/H/HillB-scratch/OAI/score_xrays_%j.err

export LOGLEVEL="DEBUG"
export REMOTE_PROJECT_DIR="/dartfs-hpc/rc/home/f/f005gzf/projects/xray_fingerprints"
export REMOTE_IMAGE_SRC="/dartfs/rc/nosnapshots/H/HillB-scratch/OAI/processed_images/xrays/knee/BilatPAFixedFlex/224x224/no_dicom_proc/self_scaled/group_norm"
export REMOTE_IMAGE_WEIGHTS="/dartfs/rc/nosnapshots/H/HillB-scratch/OAI/images/processing_1661529906/model_weights/train_best_model_continuous_y_col_koos_pain_subscore_epochs_25_model_resnet18/2022_08_28_15_35_04_884724/final/model_weights.pth"
export REMOTE_OUTPUT_CSV="/dartfs/rc/nosnapshots/H/HillB-scratch/OAI/model_predictions"

cd $REMOTE_PROJECT_DIR
conda activate opt_model_selector

python "./analysis/results_bias/score_xrays.py" \
    --weights $REMOTE_IMAGE_WEIGHTS \
    --source $REMOTE_IMAGE_SRC \
    --output $REMOTE_OUTPUT_CSV
