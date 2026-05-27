#!/bin/bash
# NeuroLens: Master script to submit all jobs in dependency order.
#
# Usage:
#   ./scripts/run_all.sh [NSD_ROOT] [COCO_ROOT]
#
# Jobs are submitted with SLURM dependency chains:
#   1. extract_llava -> 2. extract_qwen2vl -> 3. extract_internvl2 (GPU, sequential)
#   4. brain_data (CPU, can run in parallel with extraction)
#   5. analysis (CPU, depends on extraction + brain_data)
#   6. patching (GPU, depends on extraction)
#   7. visualization (CPU, depends on analysis + patching)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

export NSD_ROOT=${1:-${NSD_ROOT:-"/path/to/nsd"}}
export COCO_ROOT=${2:-${COCO_ROOT:-"/path/to/coco"}}

echo "============================================"
echo "NeuroLens: Full Pipeline Submission"
echo "NSD Root: ${NSD_ROOT}"
echo "COCO Root: ${COCO_ROOT}"
echo "Project: ${PROJECT_DIR}"
echo "============================================"

mkdir -p logs

# Step 1: Extraction (GPU jobs, sequential per model to avoid OOM)
JOB_EXT_LLAVA=$(sbatch --parsable scripts/run_extraction.sh llava)
echo "Submitted extraction/llava: ${JOB_EXT_LLAVA}"

JOB_EXT_QWEN=$(sbatch --parsable --dependency=afterok:${JOB_EXT_LLAVA} \
    scripts/run_extraction.sh qwen2vl)
echo "Submitted extraction/qwen2vl: ${JOB_EXT_QWEN}"

JOB_EXT_INTERN=$(sbatch --parsable --dependency=afterok:${JOB_EXT_QWEN} \
    scripts/run_extraction.sh internvl2)
echo "Submitted extraction/internvl2: ${JOB_EXT_INTERN}"

# Step 2: Brain data processing (CPU, can start immediately)
JOB_BRAIN=$(sbatch --parsable scripts/run_analysis.sh llava)
echo "Submitted brain_data: ${JOB_BRAIN}"

# Note: run_analysis.sh handles brain data processing before analysis.
# For the full pipeline, we submit analysis jobs after extraction completes.

# Step 3: Analysis (depends on all extractions + brain data)
ALL_EXTRACT="${JOB_EXT_LLAVA}:${JOB_EXT_QWEN}:${JOB_EXT_INTERN}"

for MODEL in llava qwen2vl internvl2; do
    JOB_ANALYSIS=$(sbatch --parsable \
        --dependency=afterok:${ALL_EXTRACT},afterok:${JOB_BRAIN} \
        scripts/run_analysis.sh ${MODEL})
    echo "Submitted analysis/${MODEL}: ${JOB_ANALYSIS}"
done

# Step 4: Patching (GPU, depends on extraction only)
for MODEL in llava qwen2vl internvl2; do
    JOB_PATCH=$(sbatch --parsable \
        --dependency=afterok:${ALL_EXTRACT} \
        scripts/run_patching.sh ${MODEL})
    echo "Submitted patching/${MODEL}: ${JOB_PATCH}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs will be in: ${PROJECT_DIR}/logs/"
