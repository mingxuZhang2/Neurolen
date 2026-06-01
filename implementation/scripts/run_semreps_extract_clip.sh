#!/bin/bash
#SBATCH --job-name=sr_clip
#SBATCH --partition=i64m1tga40u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/mllm/logs/sr_clip_%j.log

# CLIP baseline extraction on an HPC2 A40 GPU. Model + images are local; compute
# nodes have no outbound internet so force offline HF.
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
PY=/hpc2hdd/home/mzhang630/miniconda3/bin/python
SCRIPTS=/hpc2hdd/home/mzhang630/data/mllm/implementation/scripts

echo "=== node: $(hostname)  date: $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no gpu visible)"
cd "$SCRIPTS"
$PY "$SCRIPTS/run_semreps_extract_clip.py"
echo "clip extract rc=$?"
echo "=== done: $(date) ==="
