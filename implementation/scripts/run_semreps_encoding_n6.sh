#!/bin/bash
#SBATCH --job-name=sr_enc_n6
#SBATCH --partition=a128m512u
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/hpc2hdd/home/mzhang630/data/mllm/logs/sr_enc_n6_%j.log

# N=6 SemReps encoding + noise-ceiling normalization on a DEDICATED HPC2 CPU node.
# The shared login node OOMs (other users hold ~480/503 GB); a compute node gives
# clean 512 GB. Betas/features/scripts all live on shared GPFS (/hpc2hdd), so the
# compute node sees them directly. Thread-capped (uncapped BLAS thrashes PCA).

set -uo pipefail
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export SR_SUBS="sub-01,sub-02,sub-03,sub-04,sub-05,sub-07"
PY=/hpc2hdd/home/mzhang630/miniconda3/bin/python
SCRIPTS=/hpc2hdd/home/mzhang630/data/mllm/implementation/scripts

echo "=== node: $(hostname)  mem: $(free -g | awk '/Mem:/{print $2}')GB  date: $(date) ==="
echo "=== SR_SUBS=$SR_SUBS ==="

echo "########## STEP 1/2: encoding (within/cross dissociation) ##########"
$PY "$SCRIPTS/run_semreps_encoding_multi.py"
echo "encoding rc=$?"

echo "########## STEP 2/2: noise-ceiling normalization ##########"
$PY "$SCRIPTS/run_semreps_noise_ceiling.py"
echo "ceiling rc=$?"

echo "=== done: $(date) ==="
