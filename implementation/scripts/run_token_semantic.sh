#!/bin/bash
#SBATCH --job-name=tok_sem
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/tok_sem_%j.log

# Token × layer logit-lens → language ROI semantic encoding.
# Needs the LLM word-embedding table -> uses GPU briefly to load LLaVA;
# main compute is Ridge.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)"
echo "Date: $(date)"

SUBJ=${SUBJ:-1}
LLAVA_PATH=${LLAVA_PATH:-$PROJECT_DIR/models/llava-v1.5-7b}
ROIS=${ROIS:-"Broca IFG_extended auditory_assoc temporal_pole"}

$PY $PROJECT_DIR/implementation/scripts/run_token_pipeline.py \
    --step semantic \
    --model llava \
    --model-path-override $LLAVA_PATH \
    --output-root $PROJECT_DIR/results_v2 \
    --nsd-root $PROJECT_DIR/../nsd \
    --subjects $SUBJ \
    --pca-dim 256 \
    --language-rois $ROIS

echo "Done: $(date)"
