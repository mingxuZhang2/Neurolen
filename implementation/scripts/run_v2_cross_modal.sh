#!/bin/bash
#SBATCH --job-name=v2_xmodal
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/v2_xmodal_%j.log

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)"
echo "Date: $(date)"

COMMON="--model llava \
        --output-root $PROJECT_DIR/results_v2 \
        --nsd-root $PROJECT_DIR/../nsd"

echo "=== cross_modal step ==="
EXTRA=""
if [ -n "${ROIS:-}" ]; then
    EXTRA="--rois $ROIS"
fi
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step cross_modal \
    --subjects 1 \
    --pca-dim 512 \
    $EXTRA \
    $COMMON

echo "=== visualize step ==="
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step visualize \
    $COMMON

echo "Done: $(date)"
