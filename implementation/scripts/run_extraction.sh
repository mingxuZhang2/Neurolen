#!/bin/bash
#SBATCH --job-name=neurolens_extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# NeuroLens: Activation Extraction
# Extracts per-layer hidden states from MLLMs for NSD stimuli.
# Requires 1x GPU (H100/A100 80GB).

set -euo pipefail

# Configuration
MODEL=${1:-"llava"}
NSD_ROOT=${NSD_ROOT:-"/path/to/nsd"}
COCO_ROOT=${COCO_ROOT:-"/path/to/coco"}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "NeuroLens: Activation Extraction"
echo "Model: ${MODEL}"
echo "NSD Root: ${NSD_ROOT}"
echo "COCO Root: ${COCO_ROOT}"
echo "Project Dir: ${PROJECT_DIR}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

# Setup environment
cd "${PROJECT_DIR}"
mkdir -p logs

# Activate conda environment
source activate neurolens 2>/dev/null || conda activate neurolens

# Run extraction
echo "[$(date)] Starting extraction for ${MODEL}"
python run_experiment.py \
    --step extract \
    --model "${MODEL}" \
    --nsd-root "${NSD_ROOT}" \
    --coco-root "${COCO_ROOT}" \
    --log-level INFO

echo "[$(date)] Extraction complete for ${MODEL}"

# Also compute model RDMs
echo "[$(date)] Computing model RDMs for ${MODEL}"
python run_experiment.py \
    --step rdms \
    --model "${MODEL}" \
    --nsd-root "${NSD_ROOT}" \
    --coco-root "${COCO_ROOT}" \
    --log-level INFO

echo "[$(date)] Model RDMs complete for ${MODEL}"
