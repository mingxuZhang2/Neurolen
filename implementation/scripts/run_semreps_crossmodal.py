"""SemReps cross-modal analysis: does MLLM's image<->text convergence mirror
the brain's see<->read convergence?

Runs on HPC2 (betas live here). Needs both feature npz copied here:
  results_semreps/semreps_test_features.npz          (image side: image_L)
  results_semreps/semreps_test_caption_features.npz  (caption side: caption_L)

Outputs a printed report + saves results_semreps/crossmodal_results.json

Design
------
MLLM side (per decoder layer L):
  - convergence_rdm[L]   = Spearman( RDM(image_L), RDM(caption_L) )   -> shared geometry
  - match_acc[L]         = top-1 cross-modal retrieval acc (img_i -> its own caption)
Brain side (per ROI group, averaged over subjects/hemis):
  - convergence_rdm[ROI] = Spearman( RDM(see), RDM(read) )
Two legs (per layer x ROI, averaged over subjects/hemis):
  - see_align[L,ROI]  = Spearman( RDM(image_L), RDM(see) )
  - read_align[L,ROI] = Spearman( RDM(caption_L), RDM(read) )
Correspondence:
  - For each ROI, which layer maximizes see_align / read_align.
  - Does the brain see-read convergence gradient track MLLM layer convergence.
"""
import numpy as np
import nibabel as nib
from scipy import stats
from scipy.spatial.distance import pdist
from pathlib import Path
import json, os, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('crossmodal')

SEMREPS_DIR = Path('/hpc2hdd/home/mzhang630/data/semreps')
RESULTS_DIR = SEMREPS_DIR
IMG_NPZ = RESULTS_DIR / 'semreps_test_features.npz'
CAP_NPZ = RESULTS_DIR / 'semreps_test_caption_features.npz'

LAYERS = [0, 4, 8, 12, 14, 16, 20, 24, 31]
SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']
HEMIS = ['left', 'right']

# Desikan-Killiany ROI groups along a visual -> semantic gradient
ROI_GROUPS = {
    'early_visual':    ['pericalcarine', 'cuneus', 'lingual'],
    'ventral_visual':  ['lateraloccipital', 'fusiform', 'parahippocampal', 'inferiortemporal'],
    'lateral_temporal':['superiortemporal', 'middletemporal', 'bankssts'],
    'language_ifg':    ['parsopercularis', 'parstriangularis', 'parsorbitalis'],
    'parietal_assoc':  ['supramarginal', 'inferiorparietal'],
}
GRADIENT_ORDER = ['early_visual', 'ventral_visual', 'lateral_temporal', 'parietal_assoc', 'language_ifg']


def zscore_rows(X):
    """z-score each item (row) across features; guards zero-var."""
    X = X.astype(np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def rdm(X):
    """1 - correlation RDM (condensed) on z-scored items."""
    return pdist(zscore_rows(X), metric='correlation')


def rdm_corr(a, b):
    r, _ = stats.spearmanr(a, b)
    return r


def crossmodal_match_acc(img, cap):
    """Top-1 retrieval: for each image, is its own caption the nearest (cosine)?"""
    A = zscore_rows(img); B = zscore_rows(cap)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    sim = A @ B.T            # (n, n) image x caption
    n = sim.shape[0]
    top1 = (sim.argmax(axis=1) == np.arange(n)).mean()
    # chance = 1/n
    return float(top1)


def load_annot(hemi):
    h = 'lh' if hemi == 'left' else 'rh'
    for p in [
        f'/hpc2hdd/home/mzhang630/data/nsd/nsddata/freesurfer/fsaverage/label/{h}.aparc.annot',
        f'{os.environ.get("FREESURFER_HOME","/usr/local/freesurfer")}/subjects/fsaverage/label/{h}.aparc.annot',
    ]:
        if os.path.exists(p):
            labels, ctab, names = nib.freesurfer.read_annot(p)
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            return labels, names
    raise FileNotFoundError(f'no aparc.annot for {hemi}')


def load_betas(sub, hemi, condition, coco_ids):
    base = SEMREPS_DIR / 'surface' / hemi / sub / condition
    if not base.exists():
        return None
    out = []
    for cid in coco_ids:
        p = base / f'beta_{cid:06d}.gii'
        if not p.exists():
            return None
        out.append(nib.load(str(p)).darrays[0].data)
    return np.stack(out)


# ===================== Load MLLM features =====================
logger.info('Loading MLLM image + caption features...')
img = np.load(IMG_NPZ)
cap = np.load(CAP_NPZ)
coco_ids = [int(c) for c in img['coco_ids']]
assert [int(c) for c in cap['coco_ids']] == coco_ids, 'coco_id order mismatch between image and caption npz!'
n = len(coco_ids)
logger.info(f'{n} items aligned')

img_rdm = {L: rdm(img[f'layer_{L}']) for L in LAYERS}
cap_rdm = {L: rdm(cap[f'layer_{L}']) for L in LAYERS}

# ---- MLLM convergence + matching ----
logger.info('')
logger.info('=' * 70)
logger.info('MLLM IMAGE<->CAPTION CONVERGENCE (per layer)')
logger.info('=' * 70)
logger.info(f'{"layer":<8}{"RDM_corr(img,cap)":<20}{"xmodal_match_acc":<18}(chance={1/n:.3f})')
mllm_conv = {}
for L in LAYERS:
    c = rdm_corr(img_rdm[L], cap_rdm[L])
    acc = crossmodal_match_acc(img[f'layer_{L}'], cap[f'layer_{L}'])
    mllm_conv[L] = {'rdm_corr': float(c), 'match_acc': acc}
    logger.info(f'L{L:<7}{c:<20.4f}{acc:<18.4f}')

# ===================== Brain RDMs =====================
logger.info('')
logger.info('Loading brain betas + building ROI RDMs (6 subj x 2 hemi)...')

# accumulate per ROI: list of brain see/read RDMs across subj x hemi
brain_see_rdms = {g: [] for g in ROI_GROUPS}
brain_read_rdms = {g: [] for g in ROI_GROUPS}

for hemi in HEMIS:
    labels, names = load_annot(hemi)
    name2idx = {nm: i for i, nm in enumerate(names)}
    roi_vertices = {}
    for g, rlist in ROI_GROUPS.items():
        idx = []
        for rn in rlist:
            if rn in name2idx:
                idx.extend(np.where(labels == name2idx[rn])[0].tolist())
        roi_vertices[g] = np.array(idx)

    for sub in SUBJECTS:
        see = load_betas(sub, hemi, 'betas_test_image', coco_ids)
        read = load_betas(sub, hemi, 'betas_test_caption', coco_ids)
        if see is None or read is None:
            logger.warning(f'  skip {sub} {hemi} (missing betas)')
            continue
        for g in ROI_GROUPS:
            v = roi_vertices[g]
            if len(v) < 20:
                continue
            brain_see_rdms[g].append(rdm(see[:, v]))
            brain_read_rdms[g].append(rdm(read[:, v]))
        logger.info(f'  {sub} {hemi} done')

# ---- Brain convergence per ROI ----
logger.info('')
logger.info('=' * 70)
logger.info('BRAIN SEE<->READ CONVERGENCE (per ROI, mean over subj x hemi)')
logger.info('=' * 70)
brain_conv = {}
for g in GRADIENT_ORDER:
    convs = [rdm_corr(s, r) for s, r in zip(brain_see_rdms[g], brain_read_rdms[g])]
    if not convs:
        continue
    m, sd = np.mean(convs), np.std(convs)
    brain_conv[g] = {'mean': float(m), 'std': float(sd), 'n': len(convs)}
    logger.info(f'  {g:<18} see-read RDM corr = {m:+.4f} ± {sd:.4f}  (n={len(convs)})')

# ===================== Two legs: layer x ROI alignment =====================
logger.info('')
logger.info('=' * 70)
logger.info('TWO LEGS: image_L -> SEE  and  caption_L -> READ  (mean over subj x hemi)')
logger.info('=' * 70)
see_align = {g: {} for g in ROI_GROUPS}   # [roi][L] = mean spearman
read_align = {g: {} for g in ROI_GROUPS}
for g in GRADIENT_ORDER:
    if not brain_see_rdms[g]:
        continue
    for L in LAYERS:
        s_vals = [rdm_corr(img_rdm[L], b) for b in brain_see_rdms[g]]
        r_vals = [rdm_corr(cap_rdm[L], b) for b in brain_read_rdms[g]]
        see_align[g][L] = float(np.mean(s_vals))
        read_align[g][L] = float(np.mean(r_vals))

# print see-align table (image -> see)
hdr = 'ROI \\ layer      ' + ''.join([f'L{L:<6}' for L in LAYERS])
logger.info('image_L -> SEE alignment:')
logger.info(hdr)
for g in GRADIENT_ORDER:
    if g not in see_align or not see_align[g]:
        continue
    row = ''.join([f'{see_align[g][L]:<7.3f}' for L in LAYERS])
    best = max(see_align[g], key=see_align[g].get)
    logger.info(f'{g:<17}{row}  peak@L{best}')
logger.info('')
logger.info('caption_L -> READ alignment:')
logger.info(hdr)
for g in GRADIENT_ORDER:
    if g not in read_align or not read_align[g]:
        continue
    row = ''.join([f'{read_align[g][L]:<7.3f}' for L in LAYERS])
    best = max(read_align[g], key=read_align[g].get)
    logger.info(f'{g:<17}{row}  peak@L{best}')

# ===================== Correspondence test =====================
logger.info('')
logger.info('=' * 70)
logger.info('CORRESPONDENCE: does brain see-read convergence track the visual->semantic gradient?')
logger.info('=' * 70)
grad_idx = list(range(len(GRADIENT_ORDER)))
conv_vals = [brain_conv[g]['mean'] for g in GRADIENT_ORDER if g in brain_conv]
present = [g for g in GRADIENT_ORDER if g in brain_conv]
if len(conv_vals) >= 3:
    rho, p = stats.spearmanr(list(range(len(conv_vals))), conv_vals)
    logger.info(f'  gradient(early_visual..language_ifg) vs see-read convergence: rho={rho:+.3f} p={p:.3f}')
    logger.info(f'  order: {present}')
    logger.info(f'  convs: {[round(v,3) for v in conv_vals]}')

# Save
out = {
    'coco_ids': coco_ids,
    'mllm_convergence': mllm_conv,
    'brain_convergence': brain_conv,
    'see_align': see_align,
    'read_align': read_align,
}
with open(RESULTS_DIR / 'crossmodal_results.json', 'w') as f:
    json.dump(out, f, indent=2)
logger.info('')
logger.info(f'Saved {RESULTS_DIR / "crossmodal_results.json"}')
logger.info('Cross-modal analysis complete')
