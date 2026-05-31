"""RSA pilot: Does MLLM process images like the brain does during viewing vs reading?

For 70 COCO test images, computes:
1. MLLM RDM from image token features (how similar are images in MLLM space)
2. Brain RDM from "see image" betas (how similar are images in brain space)
3. Brain RDM from "read caption" betas (how similar are captions in brain space)
4. Correlates MLLM RDM with each brain RDM across brain regions

Hypothesis: MLLM image features should correlate more with brain "see" patterns
in visual areas, and less in language areas (where "read" might dominate).
"""

import numpy as np
import nibabel as nib
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json, os, sys, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('rsa_pilot')

SEMREPS_DIR = Path('/hpc2hdd/home/mzhang630/data/semreps')
FEATURES_PATH = SEMREPS_DIR / 'semreps_test_features.npz'

# ========== Load MLLM features ==========
logger.info('Loading MLLM features...')
feat_data = np.load(FEATURES_PATH)
coco_ids = feat_data['coco_ids']
n_images = len(coco_ids)
logger.info(f'{n_images} test images')

# ========== Load brain betas ==========
def load_betas(sub, hemi, condition, coco_ids):
    """Load fsaverage surface betas for given condition, ordered by coco_ids."""
    betas = []
    base = SEMREPS_DIR / 'surface' / hemi / sub / condition
    for cid in coco_ids:
        path = base / f'beta_{cid:06d}.gii'
        if not path.exists():
            logger.warning(f'Missing: {path}')
            betas.append(None)
            continue
        gii = nib.load(str(path))
        betas.append(gii.darrays[0].data)
    valid = [b for b in betas if b is not None]
    if not valid:
        return None
    return np.stack(valid)  # (n_images, n_vertices)

def compute_rdm(X):
    """Compute RDM (1 - correlation) from feature matrix (n_images, n_features)."""
    return pdist(X, metric='correlation')

def roi_mask_from_annot(hemi):
    """Load Desikan-Killiany ROI labels from FreeSurfer fsaverage."""
    fs_dir = os.environ.get('FREESURFER_HOME', '/usr/local/freesurfer')
    annot_path = f'{fs_dir}/subjects/fsaverage/label/{hemi}.aparc.annot'
    if not os.path.exists(annot_path):
        for try_path in [
            f'/hpc2hdd/home/mzhang630/data/nsd/nsddata/freesurfer/fsaverage/label/{hemi}.aparc.annot',
            f'/data/user/mzhang630/freesurfer/subjects/fsaverage/label/{hemi}.aparc.annot',
        ]:
            if os.path.exists(try_path):
                annot_path = try_path
                break
    if not os.path.exists(annot_path):
        return None, None
    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels, names

logger.info('Loading brain betas for all subjects...')
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']

# Define ROI groups based on Desikan-Killiany atlas
visual_roi_names = ['pericalcarine', 'cuneus', 'lingual', 'lateraloccipital', 'fusiform', 'parahippocampal']
language_roi_names = ['parsopercularis', 'parstriangularis', 'parsorbitalis', 'superiortemporal', 'middletemporal', 'supramarginal']

# ========== Main RSA analysis ==========
logger.info('Running RSA analysis...')

# Compute MLLM RDMs at different layers
mllm_rdms = {}
for key in feat_data.files:
    if key == 'coco_ids':
        continue
    X = feat_data[key]  # (70, dim)
    mllm_rdms[key] = compute_rdm(X)
    logger.info(f'MLLM RDM {key}: {X.shape[1]}-d, {len(mllm_rdms[key])} pairs')

# For each subject, compute brain RDMs and correlate with MLLM RDMs
results = {key: {'see_corr': [], 'read_corr': []} for key in mllm_rdms}
results_roi = {}

for sub in subjects:
    logger.info(f'Processing {sub}...')

    # Load betas for both hemispheres
    see_lh = load_betas(sub, 'left', 'betas_test_image', coco_ids)
    see_rh = load_betas(sub, 'left', 'betas_test_caption', coco_ids)
    read_lh = load_betas(sub, 'left', 'betas_test_caption', coco_ids)

    if see_lh is None:
        logger.warning(f'Skipping {sub}')
        continue

    # Get ROI labels
    roi_labels, roi_names = roi_mask_from_annot('lh')

    if roi_labels is not None:
        # ROI-level RSA
        for roi_group_name, roi_list in [('visual', visual_roi_names), ('language', language_roi_names)]:
            roi_key = f'{sub}_{roi_group_name}'
            # Get vertex indices for this ROI group
            roi_idx = []
            for rname in roi_list:
                if rname in roi_names:
                    ridx = roi_names.index(rname)
                    roi_idx.extend(np.where(roi_labels == ridx)[0].tolist())

            if len(roi_idx) < 10:
                continue

            roi_idx = np.array(roi_idx)
            see_roi = see_lh[:, roi_idx]
            read_roi = read_lh[:, roi_idx]

            brain_see_rdm = compute_rdm(see_roi)
            brain_read_rdm = compute_rdm(read_roi)

            for mllm_key, mllm_rdm in mllm_rdms.items():
                r_see, _ = stats.spearmanr(mllm_rdm, brain_see_rdm)
                r_read, _ = stats.spearmanr(mllm_rdm, brain_read_rdm)

                result_key = f'{mllm_key}_{roi_group_name}'
                if result_key not in results_roi:
                    results_roi[result_key] = {'see': [], 'read': []}
                results_roi[result_key]['see'].append(r_see)
                results_roi[result_key]['read'].append(r_read)

    # Whole-brain RSA (LH only for speed)
    brain_see_rdm = compute_rdm(see_lh)
    brain_read_rdm = compute_rdm(read_lh)

    for mllm_key, mllm_rdm in mllm_rdms.items():
        r_see, _ = stats.spearmanr(mllm_rdm, brain_see_rdm)
        r_read, _ = stats.spearmanr(mllm_rdm, brain_read_rdm)
        results[mllm_key]['see_corr'].append(r_see)
        results[mllm_key]['read_corr'].append(r_read)

# ========== Report ==========
logger.info('')
logger.info('='*70)
logger.info('WHOLE-BRAIN RSA: MLLM RDM vs Brain RDM (6 subjects, LH)')
logger.info('='*70)
logger.info(f'{"Feature":<20} {"See(mean±std)":<20} {"Read(mean±std)":<20} {"See>Read?":<10}')

for key in sorted(results.keys()):
    see = results[key]['see_corr']
    read = results[key]['read_corr']
    if len(see) == 0:
        continue
    see_m, see_s = np.mean(see), np.std(see)
    read_m, read_s = np.mean(read), np.std(read)
    winner = 'YES' if see_m > read_m else 'no'
    logger.info(f'{key:<20} {see_m:>6.4f}±{see_s:.4f}     {read_m:>6.4f}±{read_s:.4f}     {winner}')

logger.info('')
logger.info('='*70)
logger.info('ROI-LEVEL RSA: Visual ROIs vs Language ROIs')
logger.info('='*70)

for mllm_key in ['clip_features', 'layer_0', 'layer_14', 'layer_31']:
    logger.info(f'\n--- {mllm_key} ---')
    for roi_group in ['visual', 'language']:
        rk = f'{mllm_key}_{roi_group}'
        if rk in results_roi and len(results_roi[rk]['see']) > 0:
            see = results_roi[rk]['see']
            read = results_roi[rk]['read']
            see_m = np.mean(see)
            read_m = np.mean(read)
            diff = see_m - read_m
            # Paired t-test
            if len(see) > 1:
                t, p = stats.ttest_rel(see, read)
            else:
                t, p = 0, 1
            logger.info(f'  {roi_group:<10}: see={see_m:.4f}  read={read_m:.4f}  diff={diff:+.4f}  t={t:.2f} p={p:.4f}')

logger.info('')
logger.info('RSA pilot complete')
