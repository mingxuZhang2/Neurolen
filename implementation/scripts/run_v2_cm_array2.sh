#!/bin/bash
#SBATCH --job-name=v2_xm2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/v2_xm2_%A_%a.log
#SBATCH --array=0-13%8

# 14 ROIs that completed in the first run without per-ROI incremental save
ROIS=(V1 V2 V3 V4 early_visual FFA PPA EBA VWFA ventral_stream STS AG TPOJ lateral_stream)
ROI=${ROIS[$SLURM_ARRAY_TASK_ID]}

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  Array index: $SLURM_ARRAY_TASK_ID  ROI: $ROI"
echo "Date: $(date)"

$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step cross_modal \
    --model llava \
    --subjects 1 \
    --pca-dim 512 \
    --rois $ROI \
    --output-root $PROJECT_DIR/results_v2 \
    --nsd-root $PROJECT_DIR/../nsd

echo "Done $ROI: $(date)"
