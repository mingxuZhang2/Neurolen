#!/bin/bash
#SBATCH --job-name=tok_extract
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/tok_extract_%j.log

# Token-level activation + logit-lens extraction for LLaVA-1.5-7B on shared1000.
# Output: results_v2/token_activations/llava/<modelname>/

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Date: $(date)"

LLAVA_PATH=${LLAVA_PATH:-$PROJECT_DIR/models/llava-v1.5-7b}
LIMIT=${LIMIT:-0}

$PY $PROJECT_DIR/implementation/scripts/run_token_pipeline.py \
    --step extract \
    --model llava \
    --model-path-override $LLAVA_PATH \
    --output-root $PROJECT_DIR/results_v2 \
    --nsd-root $PROJECT_DIR/../nsd \
    --pca-dim 256 \
    --top-k 10 \
    --limit $LIMIT

echo "Done: $(date)"
