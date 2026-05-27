#!/bin/bash
#SBATCH --job-name=final_spat
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/final_spat_%j.log

# Final-mode spatial encoding: 5-fold CV, 9 representative layers, 4 ROIs.
# Uses pre-PCA'd 256-d features (global PCA leakage acknowledged as limitation;
# fold-local standardization still applied).

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  Date: $(date)"

$PY -c "
import sys, json, logging, numpy as np
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('final_spatial')

from src.analysis.token_encoding import run_token_spatial_encoding, load_token_activations
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from pathlib import Path

# Load 9 representative layers (not all 32)
act_dir = Path('$PROJECT_DIR/results_v2/token_activations/llava/LLaVA_1.5_7B')
with open(act_dir / 'nsd_ids.json') as f:
    act_nsd_ids = np.array(json.load(f))

rep_layers = [0, 4, 8, 12, 14, 16, 20, 24, 31]
token_acts = load_token_activations(act_dir, rep_layers)
logger.info(f'Loaded {len(token_acts)} layers, feature dim = {next(iter(token_acts.values())).shape[2]}')

# Also load CLIP baseline
clip_dir = Path('$PROJECT_DIR/results_v2/token_activations/clip_baseline')
clip_patches = np.load(clip_dir / 'clip_patches_vis_tokens.npy').astype(np.float32)
proj_out = np.load(clip_dir / 'projector_out_vis_tokens.npy').astype(np.float32)
token_acts[-2] = clip_patches  # CLIP patches
token_acts[-1] = proj_out      # projector output
logger.info(f'Added CLIP patches {clip_patches.shape} + projector {proj_out.shape}')

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

out_dir = '$PROJECT_DIR/results_v2/token_spatial_final/subj01'
run_token_spatial_encoding(
    token_activations_per_layer=token_acts,
    activation_nsd_ids=act_nsd_ids,
    brain_per_roi=brain,
    n_folds=5,
    pca_dim=256,
    final_mode=True,
    output_dir=out_dir,
)
logger.info('Final spatial encoding done')
"

echo "Done: $(date)"
