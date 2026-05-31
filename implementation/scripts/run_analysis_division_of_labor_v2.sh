#!/bin/bash
#SBATCH --job-name=div_labor2
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/div_labor2_%j.log

# Direction A v2: Fixed analysis.
# Bug fixes: (1) raw difference instead of normalized index,
# (2) prompt-vs-generated comparison to control token-count confound,
# (3) layer-specific results, excluding L0 from average.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python
export PYTHONUNBUFFERED=1

echo "Date: $(date)"

$PY -c "
import numpy as np, json, logging
from scipy import stats
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('div_labor2')

d = np.load('$PROJECT_DIR/results_v2/sequence_dissociation/sequence_dissociation.npz', allow_pickle=True)
matrix = d['matrix']      # (3 streams, 10 layers, 11 rois)
streams = list(d['streams'])
layers = list(d['layers'])
rois = list(d['rois'])

logger.info(f'Streams: {streams}, Layers: {layers}')
logger.info(f'ROIs: {rois}')

img_idx = streams.index('image_mean')
pmt_idx = streams.index('prompt_mean')
gen_idx = streams.index('generated_mean')

brain_visual_score = {
    'V1': 1.0, 'V2': 1.0, 'V3': 1.0, 'V4': 1.0,
    'FFA': 0.8, 'PPA': 0.8, 'EBA': 0.8,
    'STS': 0.0,
    'AG': -0.5,
    'Broca': -1.0,
    'IFG_extended': -0.8,
}
brain_scores = [brain_visual_score.get(r, 0) for r in rois]

# ========== Fix 1: Raw difference instead of normalized index ==========
logger.info('')
logger.info('='*60)
logger.info('FIX 1: Raw difference (image - generated), layer-averaged')
logger.info('='*60)

img_avg = matrix[img_idx].mean(axis=0)
gen_avg = matrix[gen_idx].mean(axis=0)
raw_diff = img_avg - gen_avg

logger.info(f'{\"ROI\":<15} {\"Image\":<10} {\"Generated\":<10} {\"Diff\":<10} {\"Brain\":<8}')
for i, roi in enumerate(rois):
    logger.info(f'{roi:<15} {img_avg[i]:<10.4f} {gen_avg[i]:<10.4f} {raw_diff[i]:<10.4f} {brain_scores[i]:<8.1f}')

rho, p = stats.spearmanr(brain_scores, raw_diff)
logger.info(f'Spearman rho={rho:.4f} p={p:.4f}')

# ========== Fix 2: Prompt vs Generated (controls token-count confound) ==========
logger.info('')
logger.info('='*60)
logger.info('FIX 2: Prompt vs Generated (matched token counts)')
logger.info('='*60)

pmt_avg = matrix[pmt_idx].mean(axis=0)
pmt_gen_diff = pmt_avg - gen_avg

logger.info(f'{\"ROI\":<15} {\"Prompt\":<10} {\"Generated\":<10} {\"Diff\":<10} {\"Brain\":<8} {\"Winner\":<12}')
for i, roi in enumerate(rois):
    winner = 'prompt' if pmt_gen_diff[i] > 0 else 'GENERATED'
    logger.info(f'{roi:<15} {pmt_avg[i]:<10.4f} {gen_avg[i]:<10.4f} {pmt_gen_diff[i]:<+10.4f} {brain_scores[i]:<8.1f} {winner}')

rho2, p2 = stats.spearmanr(brain_scores, pmt_gen_diff)
logger.info(f'Spearman rho={rho2:.4f} p={p2:.4f}')

# Double dissociation check
visual_rois = ['V1', 'V2', 'V3', 'V4', 'FFA', 'PPA', 'EBA']
lang_rois = ['STS', 'AG', 'Broca', 'IFG_extended']
vis_idx = [rois.index(r) for r in visual_rois]
lang_idx = [rois.index(r) for r in lang_rois]

vis_pmt_wins = sum(1 for i in vis_idx if pmt_gen_diff[i] > 0)
lang_gen_wins = sum(1 for i in lang_idx if pmt_gen_diff[i] < 0)
logger.info(f'Visual ROIs where prompt > gen: {vis_pmt_wins}/{len(vis_idx)}')
logger.info(f'Language ROIs where gen > prompt: {lang_gen_wins}/{len(lang_idx)}')
if vis_pmt_wins == len(vis_idx) and lang_gen_wins == len(lang_idx):
    logger.info('*** PERFECT DOUBLE DISSOCIATION ***')
elif vis_pmt_wins == len(vis_idx) or lang_gen_wins == len(lang_idx):
    logger.info('Partial dissociation')

# ========== Fix 3: Layer-specific analysis ==========
logger.info('')
logger.info('='*60)
logger.info('FIX 3: Layer-specific correlations (prompt - generated)')
logger.info('='*60)

logger.info(f'{\"Layer\":<8} {\"Spearman_rho\":<15} {\"p-value\":<12} {\"Dissoc?\":<12}')
best_rho = -999
best_layer = -1
for li, L in enumerate(layers):
    pmt_L = matrix[pmt_idx, li, :]
    gen_L = matrix[gen_idx, li, :]
    diff_L = pmt_L - gen_L
    rho_L, p_L = stats.spearmanr(brain_scores, diff_L)

    # Check dissociation at this layer
    vis_wins = sum(1 for i in vis_idx if diff_L[i] > 0)
    lang_wins = sum(1 for i in lang_idx if diff_L[i] < 0)
    if vis_wins == len(vis_idx) and lang_wins == len(lang_idx):
        dissoc = 'DOUBLE'
    elif vis_wins > len(vis_idx)//2 and lang_wins > len(lang_idx)//2:
        dissoc = 'partial'
    else:
        dissoc = '-'

    logger.info(f'L{L:<6} {rho_L:>12.4f}   {p_L:>10.4f}   {dissoc}')
    if rho_L > best_rho:
        best_rho = rho_L
        best_layer = L

logger.info(f'Peak correlation at L{best_layer} (rho={best_rho:.4f})')

# ========== Layer-averaged excluding L0 ==========
logger.info('')
logger.info('='*60)
logger.info('Layer-averaged EXCLUDING L0 (prompt - generated)')
logger.info('='*60)

mask_no_l0 = [i for i, L in enumerate(layers) if L > 0]
pmt_no_l0 = matrix[pmt_idx][mask_no_l0].mean(axis=0)
gen_no_l0 = matrix[gen_idx][mask_no_l0].mean(axis=0)
diff_no_l0 = pmt_no_l0 - gen_no_l0

rho3, p3 = stats.spearmanr(brain_scores, diff_no_l0)
logger.info(f'Spearman rho={rho3:.4f} p={p3:.4f}')

# ========== Interaction test ==========
logger.info('')
logger.info('='*60)
logger.info('Interaction test: prompt-gen difference × ROI category')
logger.info('='*60)

vis_diffs = [pmt_gen_diff[i] for i in vis_idx]
lang_diffs = [pmt_gen_diff[i] for i in lang_idx]
t_stat, t_p = stats.ttest_ind(vis_diffs, lang_diffs)
logger.info(f'Visual ROIs mean diff: {np.mean(vis_diffs):+.4f}')
logger.info(f'Language ROIs mean diff: {np.mean(lang_diffs):+.4f}')
logger.info(f't-test: t={t_stat:.3f}, p={t_p:.4f}')

# Save
out_dir = Path('$PROJECT_DIR/results_v2/division_of_labor_v2')
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(out_dir / 'analysis_v2.npz',
    rois=np.array(rois),
    brain_scores=np.array(brain_scores),
    raw_diff_img_gen=raw_diff,
    raw_diff_pmt_gen=pmt_gen_diff,
    spearman_raw=(rho, p),
    spearman_pmt_gen=(rho2, p2),
    spearman_no_l0=(rho3, p3))

logger.info(f'Saved to {out_dir}')
logger.info('Direction A v2 complete')
"

echo "Done: $(date)"
