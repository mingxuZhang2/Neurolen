#!/bin/bash
#SBATCH --job-name=v2_extract
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/v2_extract_%j.log

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Date: $(date)"

# Override the LLaVA model path to use the local copy on HPC3
LLAVA_PATH=$PROJECT_DIR/models/llava-v1.5-7b

/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python \
    $PROJECT_DIR/implementation/scripts/run_pipeline_v2.py \
    --step extract \
    --model llava \
    --model-path-override $LLAVA_PATH \
    --output-root $PROJECT_DIR/results_v2 \
    --nsd-root $PROJECT_DIR \
    --pca-dim 0

echo "Extract done: $(date)"
