#!/bin/bash
#SBATCH --job-name=raw_ext
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/raw_ext_%j.log

# Phase 1: Re-extract raw 4096-d activations for 9 representative layers.
# These will be used with fold-local PCA in the final evaluator.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  GPU: $(nvidia-smi -L | head -1)"
echo "Date: $(date)"

LLAVA_PATH=${LLAVA_PATH:-$PROJECT_DIR/models/llava-v1.5-7b}

# Extract with pca_dim=0 (no PCA, save raw 4096-d) for representative layers
$PY $PROJECT_DIR/implementation/scripts/run_token_pipeline.py \
    --step extract \
    --model llava \
    --model-path-override $LLAVA_PATH \
    --output-root $PROJECT_DIR/results_v2_raw \
    --pca-dim 0 \
    --top-k 10

echo "Done: $(date)"
