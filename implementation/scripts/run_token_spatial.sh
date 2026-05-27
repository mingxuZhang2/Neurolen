#!/bin/bash
#SBATCH --job-name=tok_spat
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/tok_spat_%j.log

# Token × layer → V1..V4 spatial encoding.
# Ridge fit per (token, layer); ~576 × n_layers × n_rois fits.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)"
echo "Date: $(date)"

SUBJ=${SUBJ:-1}
ROIS=${ROIS:-"V1 V2 V3 V4"}

$PY $PROJECT_DIR/implementation/scripts/run_token_pipeline.py \
    --step spatial \
    --model llava \
    --output-root $PROJECT_DIR/results_v2 \
    --nsd-root $PROJECT_DIR/../nsd \
    --subjects $SUBJ \
    --pca-dim 256 \
    --visual-rois $ROIS

echo "Done: $(date)"
