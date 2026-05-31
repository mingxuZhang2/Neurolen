#!/bin/bash
#SBATCH --job-name=final_raw
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/final_raw_%j.log

# Phase 1 FINAL: 5-fold CV with fold-local PCA from raw 4096-d features.
# Processes ONE layer at a time to avoid 84 GB RAM blow-up.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  Date: $(date)"

$PY -c "
import sys, json, logging, numpy as np, gc
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('final_raw')

from src.analysis.token_encoding import run_token_spatial_encoding
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from pathlib import Path

raw_dir = Path('$PROJECT_DIR/results_v2_raw/token_activations/llava/LLaVA_1.5_7B')
clip_dir = Path('$PROJECT_DIR/results_v2/token_activations/clip_baseline')
with open(raw_dir / 'nsd_ids.json') as f:
    act_nsd_ids = np.array(json.load(f))

# Brain data
loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
brain = {}
for roi in ['V1', 'V2', 'V3', 'V4']:
    try:
        v, ids, nc = get_roi_voxels(loader, 0, roi)
        if v.shape[1] > 0:
            brain[roi] = {'voxels': v, 'nsd_ids': ids, 'ncsnr': nc}
    except Exception as e:
        logger.warning(f'ROI {roi}: {e}')

out_dir = Path('$PROJECT_DIR/results_v2/token_spatial_final_raw/subj01')

# Process ONE layer at a time to keep memory under 80 GB
# Each raw layer = (1000, 576, 4096) fp16 = 4.7 GB → fp32 = 9.4 GB
all_layers = [-2, -1, 0, 4, 8, 12, 14, 16, 20, 24, 31]

for L in all_layers:
    logger.info(f'=== Processing layer {L} ===')
    if L == -2:
        arr = np.load(clip_dir / 'clip_patches_vis_tokens.npy')
    elif L == -1:
        arr = np.load(clip_dir / 'projector_out_vis_tokens.npy')
    else:
        # Load raw 4096-d as fp16 memmap (don't copy to fp32 yet — saves 9 GB)
        arr = np.load(raw_dir / f'layer_{L}_vis_tokens.npy', mmap_mode='r')
    logger.info(f'  Loaded: shape={arr.shape} dtype={arr.dtype}')

    # Only keep aligned images to save memory (602 of 1000)
    nsd_to_idx = {int(n): i for i, n in enumerate(act_nsd_ids)}
    sample_brain_ids = list(brain.values())[0]['nsd_ids']
    keep_act = []
    aligned_nsd_ids = []
    for nid in sample_brain_ids:
        a = nsd_to_idx.get(int(nid))
        if a is not None:
            keep_act.append(a)
            aligned_nsd_ids.append(int(nid))
    keep_act = np.array(keep_act)
    aligned_nsd_ids = np.array(aligned_nsd_ids)
    arr_aligned = np.array(arr[keep_act]).astype(np.float32)
    logger.info(f'  Aligned: {arr_aligned.shape}')
    del arr

    token_acts = {L: arr_aligned}
    result = run_token_spatial_encoding(
        token_activations_per_layer=token_acts,
        activation_nsd_ids=aligned_nsd_ids,
        brain_per_roi=brain,
        n_folds=5,
        pca_dim=256,
        final_mode=True,
        output_dir=str(out_dir) + f'/layer_{L}',
    )
    # Log summary
    for ri, roi in enumerate(result.roi_names):
        mr = result.mean_r[:, 0, ri].mean()
        br = result.best_r[:, 0, ri].mean()
        logger.info(f'  L{L} {roi}: mean_r={mr:.4f} best_r={br:.4f}')

    del token_acts, arr_aligned, result
    gc.collect()

logger.info('Phase 1 final encoding (raw + fold-local PCA, layer-by-layer) done')
"

echo "Done: $(date)"
