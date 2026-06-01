#!/bin/bash
#SBATCH --job-name=sr_match
#SBATCH --partition=a128m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/mllm/logs/sr_match_%j.log

# SemReps matching-level analysis: high-power cross-modal TRANSFER test of whether the
# MLLM shares the brain's cross-modal code (the direct test of the core question).
# Login node OOMs (other users hold ~480/503 GB) -> dedicated compute node. Thread-capped.
#
# Override SR_SUBS / SR_NPERM from the submit line, e.g.
#   sbatch --export=ALL,SR_SUBS=sub-01,SR_NPERM=200 run_semreps_matching.sh   # smoke
#   sbatch run_semreps_matching.sh                                            # full N=6

set -uo pipefail
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export SR_SUBS="${SR_SUBS:-sub-01,sub-02,sub-03,sub-04,sub-05,sub-07}"
export SR_NPERM="${SR_NPERM:-1000}"
PY=/hpc2hdd/home/mzhang630/miniconda3/bin/python
SCRIPTS=/hpc2hdd/home/mzhang630/data/mllm/implementation/scripts

echo "=== node: $(hostname)  mem: $(free -g | awk '/Mem:/{print $2}')GB  date: $(date) ==="
echo "=== SR_SUBS=$SR_SUBS  SR_NPERM=$SR_NPERM ==="

cd "$SCRIPTS"
$PY "$SCRIPTS/run_semreps_matching.py"
echo "matching rc=$?"
echo "=== done: $(date) ==="
