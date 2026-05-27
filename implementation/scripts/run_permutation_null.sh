#!/bin/bash
#SBATCH --job-name=perm_null
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/perm_null_%j.log

# Permutation null: shuffle image labels, recompute mean_r.
# Tests whether token-level brain encoding is significantly above chance.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  Date: $(date)"

$PY -c "
import sys, json, logging, numpy as np
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('perm_null')

from src.analysis.token_encoding import load_token_activations, _ridge_cv_voxelwise
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from pathlib import Path

act_dir = Path('$PROJECT_DIR/results_v2/token_activations/llava/LLaVA_1.5_7B')
with open(act_dir / 'nsd_ids.json') as f:
    act_nsd_ids = np.array(json.load(f))

# Load representative layers only (saves time)
rep_layers = [0, 4, 8, 12, 14, 16, 20, 24, 28, 31]
token_acts = load_token_activations(act_dir, rep_layers)
logger.info(f'Loaded {len(token_acts)} layers')

loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
visual_rois = ['V1', 'V2', 'V3', 'V4']
nsd_to_idx = {int(n): i for i, n in enumerate(act_nsd_ids)}

N_PERM = 200
N_TOKENS_SAMPLE = 50  # sample 50 tokens per permutation (not all 576)
rng = np.random.default_rng(42)

results = {}

for roi in visual_rois:
    voxels, brain_ids, ncsnr = get_roi_voxels(loader, 0, roi)
    if voxels.shape[1] == 0:
        continue
    keep_brain, keep_act = [], []
    for brow, nid in enumerate(brain_ids):
        a = nsd_to_idx.get(int(nid))
        if a is not None:
            keep_brain.append(brow)
            keep_act.append(a)
    keep_brain = np.array(keep_brain)
    keep_act = np.array(keep_act)
    Y = voxels[keep_brain].astype(np.float32)
    n_aligned = len(keep_brain)

    for L in sorted(token_acts.keys()):
        X_all = token_acts[L][keep_act].astype(np.float32)  # (n_aligned, 576, 256)

        # Real mean_r: avg over all tokens
        real_rs = []
        tok_sample = rng.choice(576, size=N_TOKENS_SAMPLE, replace=False)
        for tok in tok_sample:
            r_v = _ridge_cv_voxelwise(X_all[:, tok, :], Y, n_folds=5)
            real_rs.append(r_v.mean())
        real_mean_r = np.mean(real_rs)

        # Null distribution: shuffle image labels
        null_mean_rs = []
        for p in range(N_PERM):
            perm_idx = rng.permutation(n_aligned)
            Y_shuf = Y[perm_idx]
            perm_rs = []
            for tok in tok_sample[:10]:  # 10 tokens per perm for speed
                r_v = _ridge_cv_voxelwise(X_all[:, tok, :], Y_shuf, n_folds=5)
                perm_rs.append(r_v.mean())
            null_mean_rs.append(np.mean(perm_rs))

        null_arr = np.array(null_mean_rs)
        p_value = (null_arr >= real_mean_r).sum() / N_PERM
        results[(roi, L)] = {
            'real': real_mean_r,
            'null_mean': null_arr.mean(),
            'null_std': null_arr.std(),
            'p_value': p_value,
        }
        logger.info(f'{roi} L{L}: real={real_mean_r:.4f} null={null_arr.mean():.4f}+/-{null_arr.std():.4f} p={p_value:.4f}')

# Save
out_dir = Path('$PROJECT_DIR/results_v2/statistical_tests')
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    out_dir / 'permutation_null.npz',
    results_keys=np.array([(r, l) for r, l in results.keys()]),
    real_values=np.array([v['real'] for v in results.values()]),
    null_means=np.array([v['null_mean'] for v in results.values()]),
    null_stds=np.array([v['null_std'] for v in results.values()]),
    p_values=np.array([v['p_value'] for v in results.values()]),
)
logger.info(f'Saved to {out_dir}/permutation_null.npz')

# Print summary
print()
print('=== Permutation null summary ===')
for (roi, L), v in sorted(results.items()):
    sig = '***' if v['p_value'] < 0.001 else '**' if v['p_value'] < 0.01 else '*' if v['p_value'] < 0.05 else 'ns'
    print(f\"{roi} L{L:>2d}: real={v['real']:.4f} null={v['null_mean']:.4f} p={v['p_value']:.4f} {sig}\")
"

echo "Done: $(date)"
