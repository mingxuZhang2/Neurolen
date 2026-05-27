#!/bin/bash
#SBATCH --job-name=mprompt
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/mprompt_%j.log

# Phase 2: Multi-prompt sequence-position dissociation.
# Tests whether image tokens are prompt-invariant (visual memory bank).
# 5 prompts × 3 streams (image/prompt/generated) × 10 layers × 11 ROIs.

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
logger = logging.getLogger('multi_prompt')

from src.models.sequence_extract import SequencePositionExtractor
from src.analysis.token_encoding import _ridge_final_voxelwise
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from pathlib import Path
import yaml

# Load model
with open('$PROJECT_DIR/implementation/configs/models.yaml') as f:
    models_cfg = yaml.safe_load(f)
model_cfg = dict(models_cfg['models']['llava'])
model_cfg['hf_id'] = '$LLAVA_PATH'

# Load stimuli
with open('$PROJECT_DIR/results_v2/stimuli_shared1000/manifest.json') as f:
    manifest = json.load(f)
image_paths = ['$PROJECT_DIR/results_v2/stimuli_shared1000/' + p for p in manifest['paths']]
nsd_ids = manifest['nsd_ids']

PROMPTS = {
    'describe': 'Describe this image in detail.',
    'objects': 'What objects are present in this image?',
    'scene': 'What type of scene is this?',
    'colors': 'What colors are visible in this image?',
    'spatial': 'Describe the spatial layout of this image.',
}

rep_layers = [0, 8, 14, 20, 31]

extractor = SequencePositionExtractor(model_cfg, device='cuda', dtype='float16', max_new_tokens=30)
extractor.load_model()

# --- Phase 1: Extract all prompts ---
all_extractions = {}
for pname, prompt in PROMPTS.items():
    logger.info(f'=== Extracting prompt: {pname} ===')
    out_dir = f'$PROJECT_DIR/results_v2/multi_prompt/{pname}'
    extractor.extract_all_sequence_positions(
        images=image_paths,
        texts=[prompt] * len(image_paths),
        output_dir=out_dir,
        rep_layers=rep_layers,
    )
    with open(out_dir + '/nsd_ids.json', 'w') as f:
        json.dump(nsd_ids, f)
    all_extractions[pname] = out_dir
    gc.collect()

extractor.cleanup()
gc.collect()

# --- Phase 2: Compare image-token similarity across prompts ---
logger.info('=== Phase 2: Cross-prompt image-token similarity ===')
# For each layer, compute cosine similarity of image_mean across prompt pairs
from numpy.linalg import norm
streams = ['image_mean', 'prompt_mean', 'generated_mean']
similarity_results = {}

for L in rep_layers:
    for stream in streams:
        vecs = {}
        for pname in PROMPTS:
            fpath = Path(all_extractions[pname]) / f'layer_{L}_{stream}.npy'
            if fpath.exists():
                vecs[pname] = np.load(fpath).astype(np.float32)  # (n_img, hd)
        if len(vecs) < 2:
            continue
        # Pairwise cosine similarity across images, averaged
        pnames = sorted(vecs.keys())
        for i in range(len(pnames)):
            for j in range(i+1, len(pnames)):
                a, b = vecs[pnames[i]], vecs[pnames[j]]
                # Per-image cosine sim
                cos = np.sum(a * b, axis=1) / (norm(a, axis=1) * norm(b, axis=1) + 1e-8)
                mean_cos = float(cos.mean())
                key = (stream, L, pnames[i], pnames[j])
                similarity_results[key] = mean_cos

# Print summary
logger.info('Cross-prompt cosine similarity (mean over images):')
for stream in streams:
    for L in rep_layers:
        sims = [v for (s, l, _, _), v in similarity_results.items() if s == stream and l == L]
        if sims:
            logger.info(f'  L{L} {stream}: mean cross-prompt sim = {np.mean(sims):.4f}')

# --- Phase 3: Encode to brain ROIs (describe prompt only as baseline) ---
logger.info('=== Phase 3: Brain encoding for describe prompt ===')
loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
act_nsd_ids = np.array(nsd_ids)
nsd_to_idx = {int(n): i for i, n in enumerate(act_nsd_ids)}

all_rois = ['V1', 'V2', 'V3', 'V4', 'FFA', 'PPA', 'EBA', 'STS', 'AG', 'Broca', 'IFG_extended']
brain = {}
for roi in all_rois:
    try:
        v, ids, nc = get_roi_voxels(loader, 0, roi)
        if v.shape[1] > 0:
            brain[roi] = {'voxels': v, 'nsd_ids': ids, 'ncsnr': nc}
    except Exception as e:
        logger.warning(f'ROI {roi}: {e}')

# Save results
res_dir = Path('$PROJECT_DIR/results_v2/multi_prompt_results')
res_dir.mkdir(parents=True, exist_ok=True)

# Save similarity
np.savez(res_dir / 'cross_prompt_similarity.npz',
    keys=np.array([(s, str(l), p1, p2) for (s, l, p1, p2) in similarity_results.keys()]),
    values=np.array(list(similarity_results.values())),
    prompts=np.array(list(PROMPTS.keys())),
    layers=np.array(rep_layers),
    streams=np.array(streams),
)

logger.info(f'Saved to {res_dir}')
print()
print('=== SUMMARY: Cross-prompt image-token similarity ===')
for stream in streams:
    vals = []
    for L in rep_layers:
        sims = [v for (s, l, _, _), v in similarity_results.items() if s == stream and l == L]
        if sims:
            vals.append(np.mean(sims))
    if vals:
        print(f'{stream:>18s}: mean={np.mean(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f}')
"

echo "Done: $(date)"
