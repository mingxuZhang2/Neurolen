#!/bin/bash
#SBATCH --job-name=seq_diss
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/seq_diss_%j.log

# Step 2: Sequence-position dissociation
# Extracts image/prompt/generated token hidden states, then encodes to V1-V4 + FFA/PPA/EBA/STS/AG

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  GPU: $(nvidia-smi -L | head -1)"
echo "Date: $(date)"

LLAVA_PATH=${LLAVA_PATH:-$PROJECT_DIR/models/llava-v1.5-7b}

$PY -c "
import sys, json, logging, numpy as np, gc
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('seq_diss')

# --- Phase 1: Extraction ---
logger.info('=== Phase 1: Sequence position extraction ===')
from src.models.sequence_extract import SequencePositionExtractor
import yaml

with open('$PROJECT_DIR/implementation/configs/models.yaml') as f:
    models_cfg = yaml.safe_load(f)
model_cfg = dict(models_cfg['models']['llava'])
model_cfg['hf_id'] = '$LLAVA_PATH'

manifest_path = '$PROJECT_DIR/results_v2/stimuli_shared1000/manifest.json'
with open(manifest_path) as f:
    manifest = json.load(f)
image_paths = [
    '$PROJECT_DIR/results_v2/stimuli_shared1000/' + p
    for p in manifest['paths']
]
nsd_ids = manifest['nsd_ids']

out_dir = '$PROJECT_DIR/results_v2/sequence_positions/llava'
rep_layers = [0, 4, 8, 12, 14, 16, 20, 24, 28, 31]

extractor = SequencePositionExtractor(model_cfg, device='cuda', dtype='float16',
                                       max_new_tokens=30)
extractor.load_model()
extractor.extract_all_sequence_positions(
    images=image_paths,
    texts=['Describe this image in detail.'] * len(image_paths),
    output_dir=out_dir,
    rep_layers=rep_layers,
)
extractor.cleanup()
gc.collect()

# Save nsd_ids
with open(out_dir + '/nsd_ids.json', 'w') as f:
    json.dump(nsd_ids, f)

# --- Phase 2: Encoding ---
logger.info('=== Phase 2: Encoding to brain ROIs ===')
from src.analysis.token_encoding import _ridge_final_voxelwise
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from pathlib import Path

loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
act_nsd_ids = np.array(nsd_ids)
nsd_to_idx = {int(n): i for i, n in enumerate(act_nsd_ids)}

# Expanded ROIs: visual + category-selective + semantic
all_rois = ['V1', 'V2', 'V3', 'V4', 'FFA', 'PPA', 'EBA', 'STS', 'AG',
            'Broca', 'IFG_extended']
brain = {}
for roi in all_rois:
    try:
        v, ids, nc = get_roi_voxels(loader, 0, roi)
        if v.shape[1] > 0:
            brain[roi] = {'voxels': v, 'nsd_ids': ids, 'ncsnr': nc}
    except Exception as e:
        logger.warning(f'ROI {roi}: {e}')

streams = ['image_mean', 'prompt_mean', 'generated_mean']
results = {}  # (stream, layer, roi) -> mean_r

for L in rep_layers:
    for stream in streams:
        fpath = Path(out_dir) / f'layer_{L}_{stream}.npy'
        if not fpath.exists():
            logger.warning(f'Missing {fpath}')
            continue
        X_all = np.load(fpath).astype(np.float32)  # (n_img, hidden_dim)

        for roi_name, roi_data in brain.items():
            Y_full = roi_data['voxels']
            brain_ids = roi_data['nsd_ids']
            keep_brain, keep_act = [], []
            for brow, nid in enumerate(brain_ids):
                a = nsd_to_idx.get(int(nid))
                if a is not None:
                    keep_brain.append(brow)
                    keep_act.append(a)
            if len(keep_brain) < 20:
                continue
            keep_brain = np.array(keep_brain)
            keep_act = np.array(keep_act)
            Y = Y_full[keep_brain].astype(np.float32)
            X = X_all[keep_act]

            r_v = _ridge_final_voxelwise(X, Y, n_folds=5, pca_dim=256)
            mr = float(r_v.mean())
            results[(stream, L, roi_name)] = mr

        logger.info(f'  L{L} {stream}: ' + ', '.join(
            f'{roi}={results.get((stream, L, roi), 0):.4f}' for roi in all_rois[:4]))

# Save results
res_dir = Path('$PROJECT_DIR/results_v2/sequence_dissociation')
res_dir.mkdir(parents=True, exist_ok=True)

# Build 3D array: (n_streams, n_layers, n_rois)
n_s, n_l, n_r = len(streams), len(rep_layers), len(all_rois)
mat = np.zeros((n_s, n_l, n_r), dtype=np.float32)
for si, s in enumerate(streams):
    for li, L in enumerate(rep_layers):
        for ri, roi in enumerate(all_rois):
            mat[si, li, ri] = results.get((s, L, roi), 0)

np.savez(
    res_dir / 'sequence_dissociation.npz',
    matrix=mat,
    streams=np.array(streams),
    layers=np.array(rep_layers),
    rois=np.array(all_rois),
)
logger.info(f'Saved to {res_dir}/sequence_dissociation.npz')

# Print summary table
print()
print('=== Sequence-Position Dissociation (mean_r) ===')
print(f'{\"\":>20s}', end='')
for roi in all_rois:
    print(f'{roi:>10s}', end='')
print()
for si, s in enumerate(streams):
    # Average across layers
    avg = mat[si].mean(axis=0)
    print(f'{s:>20s}', end='')
    for ri in range(n_r):
        print(f'{avg[ri]:>10.4f}', end='')
    print()
"

echo "Done: $(date)"
