#!/bin/bash
#SBATCH --job-name=final_raw2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/final_raw2_%j.log

# Gold-standard: 5-fold CV with fold-local PCA from raw 4096-d features.
# v2: fixes segfault by limiting BLAS threads and joblib parallelism for 4096-d.
# Skips CLIP/projector (already done in v1).

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

# Limit BLAS threads to prevent segfault from 12 threads × PCA simultaneously
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "Node: $(hostname)  Date: $(date)"

$PY -c "
import sys, json, logging, numpy as np, gc
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('final_raw2')

from src.analysis.token_encoding import _ridge_final_voxelwise
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from pathlib import Path

raw_dir = Path('$PROJECT_DIR/results_v2_raw/token_activations/llava/LLaVA_1.5_7B')
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

# Pre-compute alignment indices (same for all layers)
nsd_to_idx = {int(n): i for i, n in enumerate(act_nsd_ids)}
sample_brain_ids = list(brain.values())[0]['nsd_ids']
keep_act, keep_brain_idx, aligned_nsd_ids = [], [], []
for brow, nid in enumerate(sample_brain_ids):
    a = nsd_to_idx.get(int(nid))
    if a is not None:
        keep_act.append(a)
        keep_brain_idx.append(brow)
        aligned_nsd_ids.append(int(nid))
keep_act = np.array(keep_act)
keep_brain_idx = np.array(keep_brain_idx)
n_aligned = len(keep_act)
logger.info(f'Aligned {n_aligned} images')

out_dir = Path('$PROJECT_DIR/results_v2/token_spatial_final_raw/subj01')

# Only decoder layers (CLIP/projector already done in v1)
decoder_layers = [0, 4, 8, 12, 14, 16, 20, 24, 31]
N_TOKENS = 576
BATCH_SIZE = 48  # process 48 tokens at a time
N_JOBS = 4       # limited parallelism for 4096-d PCA

for L in decoder_layers:
    logger.info(f'=== Processing layer {L} ===')
    # Load as memmap to avoid copying full array into RAM
    arr = np.load(raw_dir / f'layer_{L}_vis_tokens.npy', mmap_mode='r')
    logger.info(f'  Loaded memmap: shape={arr.shape} dtype={arr.dtype}')

    # Pre-align: only copy aligned rows (602 of 1000)
    arr_aligned = np.array(arr[keep_act]).astype(np.float32)
    logger.info(f'  Aligned: {arr_aligned.shape} ({arr_aligned.nbytes / 1e9:.1f} GB)')
    del arr

    layer_out = out_dir / f'layer_{L}'
    inc_dir = layer_out / 'per_roi_layer'
    inc_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in ['V1', 'V2', 'V3', 'V4']:
        Y_full = brain[roi_name]['voxels']
        Y = Y_full[keep_brain_idx].astype(np.float32)
        logger.info(f'  ROI {roi_name}: Y shape={Y.shape}')

        all_best_r = np.zeros(N_TOKENS, dtype=np.float32)
        all_mean_r = np.zeros(N_TOKENS, dtype=np.float32)
        all_best_voxel = np.zeros(N_TOKENS, dtype=np.int32)

        # Process tokens in batches to limit concurrent memory
        for batch_start in range(0, N_TOKENS, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, N_TOKENS)
            batch_toks = list(range(batch_start, batch_end))

            def _fit_one(tok):
                X_tok = arr_aligned[:, tok, :]  # (n_aligned, 4096)
                r_v = _ridge_final_voxelwise(X_tok, Y, n_folds=5, pca_dim=256)
                return tok, float(r_v.max()), int(np.argmax(r_v)), float(r_v.mean())

            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=N_JOBS, prefer='threads')(
                    delayed(_fit_one)(t) for t in batch_toks
                )
            except ImportError:
                results = [_fit_one(t) for t in batch_toks]

            for tok, br, bv, mr in results:
                all_best_r[tok] = br
                all_mean_r[tok] = mr
                all_best_voxel[tok] = bv

            logger.info(f'    batch {batch_start}-{batch_end}: mean_r={np.mean([r[3] for r in results]):.4f}')

        mr = float(all_mean_r.mean())
        br = float(all_best_r.mean())
        logger.info(f'  L{L} {roi_name}: mean_r={mr:.4f} best_r={br:.4f}')

        np.savez(inc_dir / f'{roi_name}_L{L}.npz',
            best_r=all_best_r, mean_r=all_mean_r, best_voxel=all_best_voxel)

        del Y
        gc.collect()

    del arr_aligned
    gc.collect()

logger.info('Gold-standard encoding (decoder layers, raw 4096-d + fold-local PCA) done')
"

echo "Done: $(date)"
