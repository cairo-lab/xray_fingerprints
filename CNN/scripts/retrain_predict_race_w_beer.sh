#!/bin/bash
# ==============================================================================
# This script performs cloud training for a PyTorch model.

# Set in variable.sh
# runtimeVersion - The AI Platform Training runtime version to use for the job.
# pythonVersion # The Python version to use for the job. Python 3.5 is available in runtime versions 1.13 through 1.14. Python 3.7 is available in runtime versions 1.15 and later.
#
# Set by gcloud initialization during .bashrc
# serviceAccount - The email address of a service account for AI Platform Training to use when it runs your training application. This can provide your training application access to Google Cloud resources without granting direct access to your project's AI Platform Google-managed service account. This field is optional. Learn more about the requirements for custom service accounts.
#
# Get status of job:
# gcloud ai-platform jobs describe ${JOB_NAME}
#
# List all jobs and their status:
# gcloud ai-platform jobs list
echo "Submitting Vertex AI PyTorch job"

PHASE=experiment
TARGET="RACE"
TYPE="multiclass"
FOLD_COLUMN='ID'
TRAINING_ZSCORE="True"
NUM_EPOCHS=20
LR=0.0001
BATCH_SIZE=256
UNFREEZE=0
#PRETRAIN_WEIGHTS="xray-shortcutting-jobs/exploration/classification/FFQ70/2/training_20240307_184248/FFQ70_classification_2_20240307_184248"
PRETRAIN_WEIGHTS="xray-shortcutting-jobs/experiment/classification/FFQ70/training_20241017_113542/FFQ70_classification_2_splitcol_ID_20241017_113542_f0e4"
DATA_BUCKET="xray-shortcutting-data"
LABEL_FILE="xray_shortcutting_labels.parquet"

IMAGES_DIR="oai_processed/xray/knee/BilatPAFixedFlex/224x224/fix_inversion/min_max_norm/nothing"

source scripts/launch_train_cloud_v100_vertex.sh