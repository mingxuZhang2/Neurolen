#!/bin/bash
#SBATCH --job-name=neurolens
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/neurolens_%j.log

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=/data/user/mzhang630/data/mllm/implementation:$PYTHONPATH

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Date: $(date)"

/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python \
    /data/user/mzhang630/data/mllm/implementation/scripts/run_pipeline.py

echo "Done: $(date)"
