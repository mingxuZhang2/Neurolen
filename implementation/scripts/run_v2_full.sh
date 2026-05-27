#!/bin/bash
#SBATCH --job-name=v2_full
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/v2_full_%j.log

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
NSD_ROOT=$PROJECT_DIR/../nsd

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Date: $(date)"
echo "NSD root: $NSD_ROOT"

LLAVA_PATH=$PROJECT_DIR/models/llava-v1.5-7b
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

# Common args
COMMON="--model llava \
        --model-path-override $LLAVA_PATH \
        --output-root $PROJECT_DIR/results_v2 \
        --nsd-root $NSD_ROOT"

# Step 1: brain (CPU-bound mostly, can run while GPU idle)
echo "=== brain step ==="
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step brain \
    --subjects 1 \
    --max-sessions ${MAX_SESSIONS:-40} \
    $COMMON

# Step 2: encode
echo "=== encode step ==="
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step encode \
    --subjects 1 \
    --pca-dim 512 \
    --use-pca \
    $COMMON

# Step 3: sweep
echo "=== sweep step ==="
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step sweep \
    --sweep-n-stim ${SWEEP_N_STIM:-30} \
    --patch-method noise \
    --run-caption-task \
    $COMMON

# Step 4: visualize
echo "=== visualize step ==="
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step visualize \
    $COMMON

echo "Full pipeline done: $(date)"
