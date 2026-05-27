#!/bin/bash
#SBATCH --job-name=v2_sweep
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/v2_sweep_%j.log

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH

LLAVA_PATH=$PROJECT_DIR/models/llava-v1.5-7b
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Date: $(date)"

COMMON="--model llava \
        --model-path-override $LLAVA_PATH \
        --output-root $PROJECT_DIR/results_v2 \
        --nsd-root $PROJECT_DIR/../nsd"

echo "=== sweep step (noise method) ==="
EXTRA_FLAGS=""
if [ "${RUN_CAPTION:-0}" == "1" ]; then
    EXTRA_FLAGS="--run-caption-task --perplexity-model ${PERPLEXITY_MODEL:-/data/user/mzhang630/data/mllm/models/gpt2}"
fi
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step sweep \
    --sweep-n-stim ${SWEEP_N_STIM:-20} \
    --patch-method ${PATCH_METHOD:-noise} \
    $EXTRA_FLAGS \
    $COMMON

echo "=== visualize step ==="
$PY $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step visualize \
    $COMMON

echo "Done: $(date)"
