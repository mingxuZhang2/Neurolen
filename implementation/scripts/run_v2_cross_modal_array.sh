#!/bin/bash
#SBATCH --job-name=v2_xm_arr
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/v2_xm_arr_%A_%a.log
#SBATCH --array=0-4

# Map array index to a ROI
ROIS=(parietal_stream Broca IFG_extended auditory_assoc temporal_pole)
ROI=${ROIS[$SLURM_ARRAY_TASK_ID]}

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)"
echo "Array index: $SLURM_ARRAY_TASK_ID  ROI: $ROI"
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
