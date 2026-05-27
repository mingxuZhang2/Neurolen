#!/bin/bash
#SBATCH --job-name=neurolens_patch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/patching_%j.out
#SBATCH --error=logs/patching_%j.err

# NeuroLens: Activation Patching and Triple Dissociation
# Runs 3 stages x 3 task types causal validation.
# Requires 1x GPU (H100/A100 80GB).

set -euo pipefail

MODEL=${1:-"llava"}
NSD_ROOT=${NSD_ROOT:-"/path/to/nsd"}
COCO_ROOT=${COCO_ROOT:-"/path/to/coco"}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "NeuroLens: Activation Patching"
echo "Model: ${MODEL}"
echo "Project Dir: ${PROJECT_DIR}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p logs

source activate neurolens 2>/dev/null || conda activate neurolens

echo "[$(date)] Starting patching for ${MODEL}"
python run_experiment.py \
    --step patch \
    --model "${MODEL}" \
    --nsd-root "${NSD_ROOT}" \
    --coco-root "${COCO_ROOT}" \
    --log-level INFO

echo "[$(date)] Patching complete for ${MODEL}"
