#!/bin/bash
#SBATCH --job-name=clip_base
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/clip_base_%j.log

# Extract CLIP patch tokens + projector output as baselines.
# Then run spatial encoding on them.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  GPU: $(nvidia-smi -L | head -1)"
echo "Date: $(date)"

LLAVA_PATH=${LLAVA_PATH:-$PROJECT_DIR/models/llava-v1.5-7b}

echo "=== Step 1: CLIP baseline extraction ==="
$PY $PROJECT_DIR/implementation/scripts/run_token_pipeline.py \
    --step clip_baseline \
    --model llava \
    --model-path-override $LLAVA_PATH \
    --output-root $PROJECT_DIR/results_v2 \
    --pca-dim 256

echo "=== Step 2: Spatial encoding on CLIP patches ==="
$PY -c "
import sys, json, logging, numpy as np
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

from src.analysis.token_encoding import run_token_spatial_encoding
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels

clip_dir = '$PROJECT_DIR/results_v2/token_activations/clip_baseline'
with open(clip_dir + '/nsd_ids.json') as f:
    nsd_ids = np.array(json.load(f))

# Load CLIP patches and projector output as two 'layers'
clip_patches = np.load(clip_dir + '/clip_patches_vis_tokens.npy').astype(np.float32)
proj_out = np.load(clip_dir + '/projector_out_vis_tokens.npy').astype(np.float32)
print(f'clip_patches: {clip_patches.shape}, proj_out: {proj_out.shape}')

# Treat as 2 pseudo-layers: layer 0 = CLIP, layer 1 = projector
token_acts = {-2: clip_patches, -1: proj_out}

loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
brain = {}
for roi in ['V1', 'V2', 'V3', 'V4']:
    try:
        v, ids, nc = get_roi_voxels(loader, 0, roi)
        if v.shape[1] > 0:
            brain[roi] = {'voxels': v, 'nsd_ids': ids, 'ncsnr': nc}
    except Exception as e:
        print(f'ROI {roi} failed: {e}')

out_dir = '$PROJECT_DIR/results_v2/token_spatial_real/subj01_clip_baseline'
run_token_spatial_encoding(
    token_activations_per_layer=token_acts,
    activation_nsd_ids=nsd_ids,
    brain_per_roi=brain,
    n_folds=5,
    output_dir=out_dir,
)
print('CLIP baseline spatial encoding done')
"

echo "Done: $(date)"
