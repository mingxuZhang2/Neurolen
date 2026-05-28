#!/bin/bash
#SBATCH --job-name=exp_rand_proj
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/exp_rand_proj_%j.log

# Experiment 3: Random projection control.
# Takes raw CLIP 1024-d patch features, projects to 4096-d with a random orthogonal matrix.
# Tests whether the brain-alignment gain is just from higher dimensionality (1024->4096)
# or from the decoder's learned computation.
# Expected: if random projection does NOT improve over CLIP, dimension is not the explanation.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)  Date: $(date)"

$PY -c "
import sys, os, json, logging, numpy as np, gc, time, torch
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('exp_rand_proj')

from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels

# ========== Phase 1: Extract raw CLIP 1024-d features ==========
logger.info('=== Phase 1: Extracting raw CLIP 1024-d patch features ===')

LLAVA_DIR = '$PROJECT_DIR/models/llava-v1.5-7b'

from transformers import CLIPVisionModel, CLIPImageProcessor

with open(os.path.join(LLAVA_DIR, 'config.json')) as f:
    cfg = json.load(f)

clip_path = cfg.get('mm_vision_tower', '')
dtype = torch.float16
device = 'cuda'

vision_tower = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=dtype).to(device)
vision_tower.eval()
image_processor = CLIPImageProcessor.from_pretrained(clip_path)
logger.info(f'CLIP loaded from {clip_path}')

# Load stimuli
manifest_path = Path('$PROJECT_DIR/results_v2/stimuli_shared1000/manifest.json')
with open(manifest_path) as f:
    manifest = json.load(f)
image_paths = [str(Path('$PROJECT_DIR/results_v2/stimuli_shared1000') / p) for p in manifest['paths']]
n_images = len(image_paths)
N_VIS = 576
CLIP_DIM = 1024

logger.info(f'Extracting {n_images} images...')
clip_raw = np.zeros((n_images, N_VIS, CLIP_DIM), dtype=np.float32)

for idx in range(n_images):
    img = Image.open(image_paths[idx]).convert('RGB')
    pixel_values = image_processor(images=img, return_tensors='pt')['pixel_values'].to(device, dtype=dtype)
    with torch.no_grad():
        vision_out = vision_tower(pixel_values, output_hidden_states=True)
        patches = vision_out.hidden_states[-2][:, 1:, :]  # (1, 576, 1024)
    clip_raw[idx] = patches[0].cpu().float().numpy()
    if (idx + 1) % 200 == 0:
        logger.info(f'  {idx+1}/{n_images}')
        torch.cuda.empty_cache()

del vision_tower
torch.cuda.empty_cache()
logger.info(f'Raw CLIP features: {clip_raw.shape}')

# ========== Phase 1b: Random projection to 4096-d ==========
logger.info('=== Phase 1b: Random orthogonal projection 1024 -> 4096 ===')

TARGET_DIM = 4096
rng = np.random.default_rng(42)

# Generate random orthogonal matrix via QR decomposition
# Draw a random (4096, 1024) matrix, then orthogonalize
random_mat = rng.standard_normal((CLIP_DIM, TARGET_DIM)).astype(np.float32)
Q, R = np.linalg.qr(random_mat.T)  # Q: (4096, 1024)
proj_matrix = Q.T  # (1024, 4096) — maps 1024-d to 4096-d, rows are orthonormal
logger.info(f'Projection matrix: {proj_matrix.shape}, ortho check: {np.allclose(proj_matrix @ proj_matrix.T, np.eye(CLIP_DIM), atol=1e-5)}')

# Project: (n_images, 576, 1024) @ (1024, 4096) -> (n_images, 576, 4096)
clip_proj = np.zeros((n_images, N_VIS, TARGET_DIM), dtype=np.float16)
for i in range(n_images):
    clip_proj[i] = (clip_raw[i] @ proj_matrix).astype(np.float16)
logger.info(f'Projected features: {clip_proj.shape}')

# Save
out_dir = Path('$PROJECT_DIR/results_v2_raw/token_activations/random_proj')
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / 'clip_raw_1024d.npy', clip_raw.astype(np.float16))
np.save(out_dir / 'clip_randproj_4096d.npy', clip_proj)
np.save(out_dir / 'proj_matrix.npy', proj_matrix)

# Copy nsd_ids
import shutil
shutil.copy('$PROJECT_DIR/results_v2_raw/token_activations/llava/LLaVA_1.5_7B/nsd_ids.json',
            out_dir / 'nsd_ids.json')

del clip_raw
gc.collect()
logger.info('Phase 1 done')

# ========== Phase 2: Encoding ==========
logger.info('=== Phase 2: Encoding both raw CLIP 1024-d and random projection 4096-d ===')

with open(out_dir / 'nsd_ids.json') as f:
    act_nsd_ids = np.array(json.load(f))

def pearson_r_per_voxel(y_pred, y_true, eps=1e-10):
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    num = (yp * yt).sum(axis=0)
    den = np.sqrt((yp ** 2).sum(axis=0) * (yt ** 2).sum(axis=0))
    r = num / np.maximum(den, eps)
    return np.where(np.isfinite(r), r, 0.0)

def ridge_fold_local_pca(X_raw, Y, n_folds=5, pca_dim=256, alpha=10000.0, seed=42):
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

loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
brain = {}
for roi in ['V1', 'V2', 'V3', 'V4']:
    try:
        v, ids, nc = get_roi_voxels(loader, 0, roi)
        if v.shape[1] > 0:
            brain[roi] = {'voxels': v, 'nsd_ids': ids, 'ncsnr': nc}
    except Exception as e:
        logger.warning(f'ROI {roi}: {e}')

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
logger.info(f'Aligned {len(keep_act)} images')

N_TOKENS = 576

# Encode both conditions: raw CLIP 1024-d and random projection 4096-d
conditions = [
    ('clip_raw_1024d', out_dir / 'clip_raw_1024d.npy', '$PROJECT_DIR/results_v2/ablation_randproj/subj01/clip_raw'),
    ('clip_randproj_4096d', out_dir / 'clip_randproj_4096d.npy', '$PROJECT_DIR/results_v2/ablation_randproj/subj01/randproj_4096d'),
]

for cond_name, feat_path, enc_path in conditions:
    logger.info(f'=== Encoding condition: {cond_name} ===')
    t0 = time.time()
    arr = np.load(feat_path, mmap_mode='r')
    arr_aligned = np.array(arr[keep_act]).astype(np.float32)
    logger.info(f'  Shape: {arr_aligned.shape}')
    del arr

    layer_out = Path(enc_path) / 'per_roi_layer'
    layer_out.mkdir(parents=True, exist_ok=True)

    for roi_name in ['V1', 'V2', 'V3', 'V4']:
        Y = brain[roi_name]['voxels'][keep_brain_idx].astype(np.float32)
        mean_r_arr = np.zeros(N_TOKENS, dtype=np.float32)

        for tok in range(N_TOKENS):
            X_tok = arr_aligned[:, tok, :]
            r_v = ridge_fold_local_pca(X_tok, Y)
            mean_r_arr[tok] = r_v.mean()
            if tok % 96 == 95:
                logger.info(f'    {roi_name} tok {tok+1}/{N_TOKENS} mean_r={mean_r_arr[:tok+1].mean():.4f}')

        mr = float(mean_r_arr.mean())
        logger.info(f'  {cond_name} {roi_name}: mean_r={mr:.4f}')
        np.savez(layer_out / f'{roi_name}.npz', mean_r=mean_r_arr)
        del Y
        gc.collect()

    del arr_aligned
    gc.collect()
    logger.info(f'  {cond_name} done in {(time.time()-t0)/60:.1f} min')

logger.info('Experiment 3 (Random projection) complete')
"

echo "Done: $(date)"
