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
# v2: fixes segfault by using sequential processing (no joblib threads for 4096-d).
# Skips CLIP/projector (already done in v1).

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "Node: $(hostname)  Date: $(date)"

$PY -c "
import sys, json, logging, numpy as np, gc, time
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('final_raw2')

from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from pathlib import Path

raw_dir = Path('$PROJECT_DIR/results_v2_raw/token_activations/llava/LLaVA_1.5_7B')
with open(raw_dir / 'nsd_ids.json') as f:
    act_nsd_ids = np.array(json.load(f))

def pearson_r_per_voxel(y_pred, y_true, eps=1e-10):
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    num = (yp * yt).sum(axis=0)
    den = np.sqrt((yp ** 2).sum(axis=0) * (yt ** 2).sum(axis=0))
    r = num / np.maximum(den, eps)
    return np.where(np.isfinite(r), r, 0.0)

def ridge_fold_local_pca(X_raw, Y, n_folds=5, pca_dim=256, alpha=10000.0, seed=42):
    \"\"\"5-fold CV with fold-local PCA. Inline to avoid import issues.\"\"\"
    n_stim, raw_dim = X_raw.shape
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y_c)
    for train_idx, test_idx in kf.split(X_raw):
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        Y_tr = Y_c[train_idx]
        if raw_dim > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=seed, svd_solver='randomized')
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_tr, Y_tr)
        preds[test_idx] = ridge.predict(X_te)
    return pearson_r_per_voxel(preds, Y_c)

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

# Pre-compute alignment
nsd_to_idx = {int(n): i for i, n in enumerate(act_nsd_ids)}
sample_brain_ids = list(brain.values())[0]['nsd_ids']
keep_act, keep_brain_idx = [], []
for brow, nid in enumerate(sample_brain_ids):
    a = nsd_to_idx.get(int(nid))
    if a is not None:
        keep_act.append(a)
        keep_brain_idx.append(brow)
keep_act = np.array(keep_act)
keep_brain_idx = np.array(keep_brain_idx)
n_aligned = len(keep_act)
logger.info(f'Aligned {n_aligned} images')

# Smoke test: one token, one ROI to verify no segfault
logger.info('=== Smoke test: single token on V1 ===')
arr_test = np.load(raw_dir / 'layer_0_vis_tokens.npy', mmap_mode='r')
X_test = np.array(arr_test[keep_act, 0, :]).astype(np.float32)
Y_test = brain['V1']['voxels'][keep_brain_idx].astype(np.float32)
r_test = ridge_fold_local_pca(X_test, Y_test)
logger.info(f'  Smoke test OK: mean_r={r_test.mean():.4f}, max_r={r_test.max():.4f}')
del arr_test, X_test, Y_test, r_test
gc.collect()

out_dir = Path('$PROJECT_DIR/results_v2/token_spatial_final_raw/subj01')
decoder_layers = [0, 4, 8, 12, 14, 16, 20, 24, 31]
N_TOKENS = 576

for L in decoder_layers:
    t0 = time.time()
    logger.info(f'=== Processing layer {L} ===')
    arr = np.load(raw_dir / f'layer_{L}_vis_tokens.npy', mmap_mode='r')
    arr_aligned = np.array(arr[keep_act]).astype(np.float32)
    logger.info(f'  Aligned: {arr_aligned.shape} ({arr_aligned.nbytes / 1e9:.1f} GB)')
    del arr

    layer_out = out_dir / f'layer_{L}'
    inc_dir = layer_out / 'per_roi_layer'
    inc_dir.mkdir(parents=True, exist_ok=True)

    for roi_name in ['V1', 'V2', 'V3', 'V4']:
        Y = brain[roi_name]['voxels'][keep_brain_idx].astype(np.float32)
        best_r_arr = np.zeros(N_TOKENS, dtype=np.float32)
        mean_r_arr = np.zeros(N_TOKENS, dtype=np.float32)
        best_voxel_arr = np.zeros(N_TOKENS, dtype=np.int32)

        for tok in range(N_TOKENS):
            X_tok = arr_aligned[:, tok, :]
            r_v = ridge_fold_local_pca(X_tok, Y)
            best_r_arr[tok] = r_v.max()
            mean_r_arr[tok] = r_v.mean()
            best_voxel_arr[tok] = np.argmax(r_v)

            if tok % 96 == 95:
                elapsed = time.time() - t0
                eta_layer = elapsed / (tok + 1) * N_TOKENS - elapsed
                logger.info(f'    {roi_name} tok {tok+1}/{N_TOKENS} mean_r={mean_r_arr[:tok+1].mean():.4f} (ETA layer: {eta_layer/60:.0f}m)')

        mr = float(mean_r_arr.mean())
        br = float(best_r_arr.mean())
        logger.info(f'  L{L} {roi_name}: mean_r={mr:.4f} best_r={br:.4f}')

        np.savez(inc_dir / f'{roi_name}_L{L}.npz',
            best_r=best_r_arr, mean_r=mean_r_arr, best_voxel=best_voxel_arr)
        del Y
        gc.collect()

    del arr_aligned
    gc.collect()
    elapsed = time.time() - t0
    logger.info(f'  Layer {L} done in {elapsed/60:.1f} min')

logger.info('Gold-standard encoding (decoder layers, raw 4096-d + fold-local PCA) done')
"

echo "Done: $(date)"
