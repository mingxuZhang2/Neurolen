#!/bin/bash
#SBATCH --job-name=tok_viz
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/tok_viz_%j.log

# Produce F1-F4 figures from spatial + semantic encoding outputs.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  Date: $(date)"

SUBJ=${SUBJ:-1}

$PY $PROJECT_DIR/implementation/scripts/run_token_pipeline.py \
    --step visualize \
    --model llava \
    --output-root $PROJECT_DIR/results_v2 \
    --nsd-root $PROJECT_DIR/../nsd \
    --subjects $SUBJ \
    --image-idx 0

echo "Done: $(date)"
