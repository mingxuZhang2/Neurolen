#!/bin/bash
#SBATCH --job-name=neurolens_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# NeuroLens: Brain-Model Alignment Analysis
# Runs RSA, CKA, and encoding models. CPU-intensive, no GPU needed.

set -euo pipefail

MODEL=${1:-"llava"}
NSD_ROOT=${NSD_ROOT:-"/path/to/nsd"}
COCO_ROOT=${COCO_ROOT:-"/path/to/coco"}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "NeuroLens: Alignment Analysis"
echo "Model: ${MODEL}"
echo "Project Dir: ${PROJECT_DIR}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-32}"
echo "============================================"

cd "${PROJECT_DIR}"
mkdir -p logs

source activate neurolens 2>/dev/null || conda activate neurolens

# Process brain data first (if not already done)
if [ ! -d "brain_data" ] || [ -z "$(ls -A brain_data 2>/dev/null)" ]; then
    echo "[$(date)] Processing brain data..."
    python run_experiment.py \
        --step brain \
        --model "${MODEL}" \
        --nsd-root "${NSD_ROOT}" \
        --coco-root "${COCO_ROOT}" \
        --log-level INFO
fi

# Run analysis (RSA + CKA + encoding)
echo "[$(date)] Starting analysis for ${MODEL}"
python run_experiment.py \
    --step analyze \
    --model "${MODEL}" \
    --nsd-root "${NSD_ROOT}" \
    --coco-root "${COCO_ROOT}" \
    --log-level INFO

echo "[$(date)] Analysis complete for ${MODEL}"

# Generate visualizations
echo "[$(date)] Generating figures..."
python run_experiment.py \
    --step visualize \
    --model "${MODEL}" \
    --nsd-root "${NSD_ROOT}" \
    --coco-root "${COCO_ROOT}" \
    --log-level INFO

echo "[$(date)] Visualization complete"
