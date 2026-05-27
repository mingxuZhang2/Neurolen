#!/bin/bash
# Setup and run NeuroLens on HPC3
set -e

PROJECT_DIR=/data/user/mzhang630/data/mllm
CONDA_PREFIX=/data/user/mzhang630/miniconda3/envs/alphasteer
PYTHON=$CONDA_PREFIX/bin/python
PIP=$CONDA_PREFIX/bin/pip

echo "=== Installing packages ==="
cd $PROJECT_DIR/transfer/wheels311 2>/dev/null || cd $PROJECT_DIR/transfer/wheels
$PIP install --no-index --find-links=. *.whl 2>/dev/null || true

echo "=== Verifying ==="
$PYTHON -c "
import rsatoolbox; print('rsatoolbox OK')
import nibabel; print('nibabel OK')
import nilearn; print('nilearn OK')
" 2>/dev/null || echo "Some packages missing, continuing..."

echo "=== Setup done ==="
