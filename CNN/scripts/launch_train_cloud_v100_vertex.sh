#!/bin/bash
# Get status of job:
# gcloud ai-platform jobs describe ${JOB_NAME}
#
# List all jobs and their status:
# gcloud ai-platform jobs list


# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the job will be run.
REGION=us-central1

# The PyTorch image provided by Vertex AI Training.
#IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest"
IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest"

# JOB_NAME: the name of your job running on Vertex AI.
PACKAGE_PATH=./trainer # this can be a GCS location to a zipped and uploaded package

trainer_args="\
--in-cloud,\
--data-dir=${DATA_BUCKET},\
--label-file=${LABEL_FILE},\
--images-dir=${IMAGES_DIR},\
--target-name=${TARGET},\
--model_type=${TYPE},\
--num-epochs=${NUM_EPOCHS},\
--batch-size=${BATCH_SIZE},\
--learning-rate=${LR}"

JOB_DATE=$(date +%Y%m%d_%H%M%S)
JOB_NAME=${TARGET}_${TYPE}
JOB_DIR="${PHASE}/${TYPE}/${TARGET}"

# Tweak the name based on whether kfolds, retrain, etc
if [[ ${TYPE} == "classification" ]]; then
  JOB_NAME=${JOB_NAME}_${THRESHOLD}
  JOB_DIR="${JOB_DIR}/${THRESHOLD}"
  trainer_args="${trainer_args},--threshold=${THRESHOLD}"
fi
if [[ -n ${KFOLDS} && ${KFOLDS} -gt 1 ]]; then
  JOB_NAME=${JOB_NAME}_kfold_${KFOLDS}
  JOB_DIR="${JOB_DIR}/kfolds"
  trainer_args="${trainer_args},--kfolds=${KFOLDS}"
fi
if [[ -n ${FOLD_COLUMN} ]]; then
  JOB_NAME=${JOB_NAME}_splitcol_${FOLD_COLUMN}
  JOB_DIR="${JOB_DIR}/split_${FOLD_COLUMN}"
  trainer_args="${trainer_args},--fold_column=${FOLD_COLUMN}"
fi
if [[ -n ${WEIGHT} ]]; then
  trainer_args="${trainer_args},--weight=${WEIGHT}"
fi
if [[ -n ${TRAINING_ZSCORE} ]]; then
  trainer_args="${trainer_args},--training-zscore=${TRAINING_ZSCORE}"
fi
if [[ -n ${IMAGENET_ZSCORE} ]]; then
  trainer_args="${trainer_args},--imagenet-zscore=${IMAGENET_ZSCORE}"
fi
if [[ -n ${UNFREEZE} ]]; then
  trainer_args="${trainer_args},--layers_to_unfreeze=${UNFREEZE}"
fi
if [[ -n ${PRETRAIN_WEIGHTS} ]]; then
  JOB_NAME=${JOB_NAME}_retrain
  JOB_DIR="${JOB_DIR}/retrain/training_"
  trainer_args="${trainer_args},--pretrain_weights=${PRETRAIN_WEIGHTS}"
else
  JOB_DIR="${PHASE}/${TYPE}/${TARGET}/training_"
fi

JOB_NAME=${JOB_NAME}_${JOB_DATE}
JOB_DIR="${JOB_DIR}${JOB_DATE}"
JOB_BUCKET="xray-shortcutting-jobs/${JOB_DIR}"
trainer_args="${trainer_args},--model-name=${JOB_NAME},--job-dir=${JOB_BUCKET}"

python setup.py sdist --formats=gztar
gcloud storage cp dist/trainer-0.1.tar.gz gs://${JOB_BUCKET}/packages/

# worker pool spec
worker_pool_spec="\
replica-count=1,\
machine-type=n1-standard-8,\
accelerator-type=NVIDIA_TESLA_V100,\
accelerator-count=1,\
executor-image-uri=${IMAGE_URI},\
python-module=trainer.task"

echo "${trainer_args}"

# submit training job to Vertex Training with
# pre-built container using gcloud CLI
gcloud ai custom-jobs create \
    --display-name=${JOB_NAME} \
    --region=${REGION} \
    --python-package-uris="gs://${JOB_BUCKET}/packages/trainer-0.1.tar.gz" \
    --worker-pool-spec="${worker_pool_spec}" \
    --args="${trainer_args}"

echo "Display Name: ${JOB_NAME}"
VERTEXAI_NAME=$(gcloud ai custom-jobs list --region=$REGION --filter="displayName:"$JOB_NAME --format="get(name)")

mkdir -p "${JOB_DIR}"
echo "${JOB_NAME}" > "${JOB_DIR}/job_name"
echo "${JOB_BUCKET}" > "${JOB_DIR}/gcs_location"
echo "${VERTEXAI_NAME}" > "${JOB_DIR}/vertex_name"
echo "${training_args}" > "${JOB_DIR}/training_args"

# Stream the logs from the job
gcloud ai custom-jobs stream-logs $(gcloud ai custom-jobs list --region=$REGION --filter="displayName:"$JOB_NAME --format="get(name)")