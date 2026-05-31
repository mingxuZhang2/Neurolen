#!/bin/bash
#SBATCH --job-name=div_labor
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/div_labor_%j.log

# Direction A: Quantify MLLM's vision-language division of labor vs brain's.
# Uses existing sequence dissociation data (3 streams × 10 layers × 11 ROIs).
# Computes specialization indices and correlates with known brain functional organization.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python
export PYTHONUNBUFFERED=1

echo "Date: $(date)"

$PY -c "
import numpy as np, json, logging
from scipy import stats
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('div_labor')

# ========== Load sequence dissociation data ==========
d = np.load('$PROJECT_DIR/results_v2/sequence_dissociation/sequence_dissociation.npz', allow_pickle=True)
matrix = d['matrix']      # (3 streams, 10 layers, 11 rois)
streams = list(d['streams'])  # image_mean, prompt_mean, generated_mean
layers = list(d['layers'])
rois = list(d['rois'])

logger.info(f'Loaded: {streams} x {len(layers)} layers x {rois}')

# ========== Brain functional labels (ground truth) ==========
# Based on well-established neuroscience:
# Visual: V1, V2, V3, V4 (early visual), FFA, PPA, EBA (category-selective visual)
# Multimodal: STS (social/speech), AG (semantic integration)
# Language: Broca (speech production), IFG (language processing)
# Score: +1 = purely visual, -1 = purely linguistic, 0 = multimodal
brain_visual_score = {
    'V1': 1.0, 'V2': 1.0, 'V3': 1.0, 'V4': 1.0,
    'FFA': 0.8, 'PPA': 0.8, 'EBA': 0.8,
    'STS': 0.0,
    'AG': -0.5,
    'Broca': -1.0,
    'IFG_extended': -0.8,
}

# ========== MLLM specialization index ==========
# For each ROI: how much does image_tokens encoding exceed generated_tokens encoding?
# Index = (image - generated) / (image + generated + eps)
# High = visually specialized, Low/negative = linguistically specialized

img_idx = streams.index('image_mean')
gen_idx = streams.index('generated_mean')

# Average across layers for overall specialization
img_avg = matrix[img_idx].mean(axis=0)  # (11 rois,)
gen_avg = matrix[gen_idx].mean(axis=0)  # (11 rois,)

mllm_spec_index = (img_avg - gen_avg) / (img_avg + gen_avg + 1e-8)

logger.info('=== ROI Specialization Analysis ===')
logger.info(f'{\"ROI\":<15} {\"Image_enc\":<12} {\"Gen_enc\":<12} {\"MLLM_spec\":<12} {\"Brain_score\":<12}')
brain_scores = []
mllm_scores = []
for i, roi in enumerate(rois):
    bs = brain_visual_score.get(roi, 0)
    ms = mllm_spec_index[i]
    brain_scores.append(bs)
    mllm_scores.append(ms)
    logger.info(f'{roi:<15} {img_avg[i]:<12.4f} {gen_avg[i]:<12.4f} {ms:<12.4f} {bs:<12.1f}')

# Correlation between MLLM specialization and brain ground truth
rho, pval = stats.spearmanr(brain_scores, mllm_scores)
r_pearson, p_pearson = stats.pearsonr(brain_scores, mllm_scores)
logger.info(f'')
logger.info(f'MLLM spec vs Brain score: Spearman rho={rho:.4f} (p={pval:.4f}), Pearson r={r_pearson:.4f} (p={p_pearson:.4f})')

# ========== Layer-wise analysis ==========
# Does the MLLM-brain specialization correlation change across layers?
logger.info('')
logger.info('=== Layer-wise MLLM-Brain correlation ===')
logger.info(f'{\"Layer\":<8} {\"Spearman_rho\":<15} {\"p-value\":<12}')

layer_rhos = []
for li, L in enumerate(layers):
    img_L = matrix[img_idx, li, :]
    gen_L = matrix[gen_idx, li, :]
    spec_L = (img_L - gen_L) / (img_L + gen_L + 1e-8)
    rho_L, p_L = stats.spearmanr(brain_scores, spec_L)
    layer_rhos.append(rho_L)
    logger.info(f'L{L:<6} {rho_L:>12.4f}   {p_L:>10.4f}')

peak_layer = layers[np.argmax(layer_rhos)]
logger.info(f'Peak alignment at L{peak_layer} (rho={max(layer_rhos):.4f})')

# ========== Three-way stream comparison ==========
logger.info('')
logger.info('=== Stream dominance per ROI category ===')

visual_rois = ['V1', 'V2', 'V3', 'V4', 'FFA', 'PPA', 'EBA']
lang_rois = ['AG', 'Broca', 'IFG_extended']

for cat_name, cat_rois in [('Visual ROIs', visual_rois), ('Language ROIs', lang_rois)]:
    cat_idx = [rois.index(r) for r in cat_rois if r in rois]
    logger.info(f'{cat_name}:')
    for s, sname in enumerate(streams):
        vals = matrix[s].mean(axis=0)[cat_idx]
        logger.info(f'  {sname}: mean_r={vals.mean():.4f} (range {vals.min():.4f}-{vals.max():.4f})')

# ========== Dissociation strength ==========
# Double dissociation: image tokens favor visual ROIs, generated tokens favor language ROIs
vis_idx = [rois.index(r) for r in visual_rois if r in rois]
lang_idx = [rois.index(r) for r in lang_rois if r in rois]

img_vis = matrix[img_idx].mean(axis=0)[vis_idx].mean()
img_lang = matrix[img_idx].mean(axis=0)[lang_idx].mean()
gen_vis = matrix[gen_idx].mean(axis=0)[vis_idx].mean()
gen_lang = matrix[gen_idx].mean(axis=0)[lang_idx].mean()

logger.info('')
logger.info('=== Double dissociation test ===')
logger.info(f'Image tokens:     visual={img_vis:.4f}  language={img_lang:.4f}  ratio={img_vis/max(img_lang,1e-6):.1f}x')
logger.info(f'Generated tokens: visual={gen_vis:.4f}  language={gen_lang:.4f}  ratio={gen_vis/max(gen_lang,1e-6):.1f}x')
logger.info(f'Interaction: image_vis-gen_vis={img_vis-gen_vis:.4f}, gen_lang-img_lang={gen_lang-img_lang:.4f}')

if img_vis > gen_vis and gen_lang > img_lang:
    logger.info('DOUBLE DISSOCIATION: image→visual, generated→language (crossover)')
elif img_vis > gen_vis:
    logger.info('SINGLE DISSOCIATION: image→visual only (no crossover)')
else:
    logger.info('NO DISSOCIATION')

# Save results
out_dir = Path('$PROJECT_DIR/results_v2/division_of_labor')
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(out_dir / 'specialization_analysis.npz',
    rois=np.array(rois),
    mllm_spec_index=np.array(mllm_scores),
    brain_visual_score=np.array(brain_scores),
    spearman_rho=rho, spearman_p=pval,
    pearson_r=r_pearson, pearson_p=p_pearson,
    layer_rhos=np.array(layer_rhos),
    layers=np.array(layers))

logger.info(f'Saved to {out_dir}')
logger.info('Direction A (division of labor) complete')
"

echo "Done: $(date)"
