"""Validate the SemReps two-leg dissociation before believing it.

(1) Permutation test: shuffle the 70 concept labels on the brain side N times,
    rebuild the null distribution of layer x ROI alignment, get p-values for the
    observed image_L->SEE and caption_L->READ peaks.
(2) Per-subject consistency: report each subject's peak layer (do they agree?).
(3) Brain reliability / noise ceiling: leave-one-subject-out inter-subject RDM
    correlation for SEE and READ per ROI. If READ reliability ~0, the see-read
    null and weak caption->read alignment are just noise, not a real finding.

Runs on HPC2.
"""
import numpy as np
import nibabel as nib
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import json, os, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('validate')

SEMREPS_DIR = Path('/hpc2hdd/home/mzhang630/data/semreps')
IMG_NPZ = SEMREPS_DIR / 'semreps_test_features.npz'
CAP_NPZ = SEMREPS_DIR / 'semreps_test_caption_features.npz'

LAYERS = [0, 4, 8, 12, 14, 16, 20, 24, 31]
SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']
HEMIS = ['left', 'right']
N_PERM = 1000
RNG_SEEDS = list(range(N_PERM))  # deterministic (no Math.random / Date in this env-safe way)

ROI_GROUPS = {
    'early_visual':    ['pericalcarine', 'cuneus', 'lingual'],
    'ventral_visual':  ['lateraloccipital', 'fusiform', 'parahippocampal', 'inferiortemporal'],
    'lateral_temporal':['superiortemporal', 'middletemporal', 'bankssts'],
    'language_ifg':    ['parsopercularis', 'parstriangularis', 'parsorbitalis'],
    'parietal_assoc':  ['supramarginal', 'inferiorparietal'],
}
GRADIENT_ORDER = ['early_visual', 'ventral_visual', 'lateral_temporal', 'parietal_assoc', 'language_ifg']


def zscore_rows(X):
    X = X.astype(np.float64)
    mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    return (X - mu) / sd

def rdm_condensed(X):
    return pdist(zscore_rows(X), metric='correlation')

def load_annot(hemi):
    h = 'lh' if hemi == 'left' else 'rh'
    p = f'/hpc2hdd/home/mzhang630/data/nsd/nsddata/freesurfer/fsaverage/label/{h}.aparc.annot'
    labels, ctab, names = nib.freesurfer.read_annot(p)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels, names

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

# ---- load features ----
logger.info('Loading MLLM features...')
img = np.load(IMG_NPZ); cap = np.load(CAP_NPZ)
coco_ids = [int(c) for c in img['coco_ids']]
assert [int(c) for c in cap['coco_ids']] == coco_ids
n = len(coco_ids)

# model RDMs (condensed) + ranks for fast spearman-as-pearson-on-ranks
img_rank = {L: stats.rankdata(rdm_condensed(img[f'layer_{L}'])) for L in LAYERS}
cap_rank = {L: stats.rankdata(rdm_condensed(cap[f'layer_{L}'])) for L in LAYERS}

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum() / d) if d > 0 else 0.0

# ---- load brain, build per (subj,hemi,roi,condition) square RDMs ----
logger.info('Loading brain betas + building RDMs...')
# brain_sq[condition][roi] = list of (70x70) squareform RDMs over subj x hemi
brain_sq = {'see': {g: [] for g in ROI_GROUPS}, 'read': {g: [] for g in ROI_GROUPS}}
subj_tag = {'see': {g: [] for g in ROI_GROUPS}, 'read': {g: [] for g in ROI_GROUPS}}

for hemi in HEMIS:
    labels, names = load_annot(hemi)
    name2idx = {nm: i for i, nm in enumerate(names)}
    roi_vert = {}
    for g, rlist in ROI_GROUPS.items():
        idx = []
        for rn in rlist:
            if rn in name2idx:
                idx.extend(np.where(labels == name2idx[rn])[0].tolist())
        roi_vert[g] = np.array(idx)
    for sub in SUBJECTS:
        see = load_betas(sub, hemi, 'betas_test_image', coco_ids)
        read = load_betas(sub, hemi, 'betas_test_caption', coco_ids)
        if see is None or read is None:
            continue
        for g in ROI_GROUPS:
            v = roi_vert[g]
            if len(v) < 20:
                continue
            brain_sq['see'][g].append(squareform(rdm_condensed(see[:, v])))
            brain_sq['read'][g].append(squareform(rdm_condensed(read[:, v])))
            subj_tag['see'][g].append(f'{sub}/{hemi}')
            subj_tag['read'][g].append(f'{sub}/{hemi}')
        logger.info(f'  {sub} {hemi} done')

iu = np.triu_indices(n, k=1)  # to re-condense permuted squareforms

# =========================================================
# (3) BRAIN RELIABILITY / NOISE CEILING (leave-one-out)
# =========================================================
logger.info('')
logger.info('='*70)
logger.info('BRAIN RELIABILITY (leave-one-subject-out inter-subject RDM corr)')
logger.info('  -> upper bound on how alignable each condition is. ~0 = noise.')
logger.info('='*70)
logger.info(f'{"ROI":<18}{"SEE reliab":<16}{"READ reliab":<16}')
reliability = {}
for g in GRADIENT_ORDER:
    rg = {}
    for cond in ['see', 'read']:
        mats = brain_sq[cond][g]
        if len(mats) < 3:
            rg[cond] = None; continue
        condensed = [squareform(m, checks=False) for m in mats]
        ranks = [stats.rankdata(c) for c in condensed]
        loo = []
        for i in range(len(ranks)):
            others = np.mean([ranks[j] for j in range(len(ranks)) if j != i], axis=0)
            loo.append(pearson(ranks[i], others))
        rg[cond] = (float(np.mean(loo)), float(np.std(loo)))
    reliability[g] = rg
    s = f'{rg["see"][0]:+.3f}±{rg["see"][1]:.3f}' if rg['see'] else 'NA'
    r = f'{rg["read"][0]:+.3f}±{rg["read"][1]:.3f}' if rg['read'] else 'NA'
    logger.info(f'{g:<18}{s:<16}{r:<16}')

# =========================================================
# (1) PERMUTATION TEST for the two legs (all layers x ROIs)
# =========================================================
logger.info('')
logger.info('='*70)
logger.info(f'PERMUTATION TEST ({N_PERM} label shuffles) — p = P(null mean >= observed)')
logger.info('='*70)

def observed_and_null(model_rank_by_L, cond):
    """For a condition, return observed mean alignment and null dist per (roi,L)."""
    obs = {g: {} for g in ROI_GROUPS}
    nul = {g: {L: np.zeros(N_PERM) for L in LAYERS} for g in ROI_GROUPS}
    # precompute per (roi) the list of brain ranks (unpermuted) for observed
    for g in GRADIENT_ORDER:
        mats = brain_sq[cond][g]
        if len(mats) < 3:
            continue
        brain_ranks = [stats.rankdata(squareform(m, checks=False)) for m in mats]
        for L in LAYERS:
            mr = model_rank_by_L[L]
            obs[g][L] = float(np.mean([pearson(mr, br) for br in brain_ranks]))
    # null: one shared label permutation per iteration, applied to all subjects
    for pi in range(N_PERM):
        rng = np.random.RandomState(RNG_SEEDS[pi])
        P = rng.permutation(n)
        for g in GRADIENT_ORDER:
            mats = brain_sq[cond][g]
            if len(mats) < 3:
                continue
            perm_ranks = []
            for m in mats:
                pm = m[np.ix_(P, P)]
                perm_ranks.append(stats.rankdata(pm[iu]))
            for L in LAYERS:
                mr = model_rank_by_L[L]
                nul[g][L][pi] = np.mean([pearson(mr, pr) for pr in perm_ranks])
        if (pi+1) % 250 == 0:
            logger.info(f'  perm {pi+1}/{N_PERM} ({cond})')
    return obs, nul

logger.info('Leg 1: image_L -> SEE')
obs_see, nul_see = observed_and_null(img_rank, 'see')
logger.info('Leg 2: caption_L -> READ')
obs_read, nul_read = observed_and_null(cap_rank, 'read')

def report_leg(name, obs, nul):
    logger.info('')
    logger.info(f'--- {name}: observed r  (p_perm) ---')
    logger.info('ROI \\ layer    ' + ''.join([f'L{L:<10}' for L in LAYERS]))
    for g in GRADIENT_ORDER:
        if g not in obs or not obs[g]:
            continue
        cells = []
        for L in LAYERS:
            o = obs[g][L]
            p = float((nul[g][L] >= o).mean())
            star = '*' if p < 0.05 else ' '
            cells.append(f'{o:+.3f}({p:.3f}){star}')
        logger.info(f'{g:<15}' + ''.join([f'{c:<12}' for c in cells]))

report_leg('image_L -> SEE', obs_see, nul_see)
report_leg('caption_L -> READ', obs_read, nul_read)

# =========================================================
# (2) PER-SUBJECT consistency of peak layer
# =========================================================
logger.info('')
logger.info('='*70)
logger.info('PER-SUBJECT peak layer (image->SEE in ventral_visual; caption->READ in language_ifg)')
logger.info('='*70)
def per_subject_peaks(model_rank_by_L, cond, roi):
    mats = brain_sq[cond][roi]; tags = subj_tag[cond][roi]
    out = []
    for m, tag in zip(mats, tags):
        br = stats.rankdata(squareform(m, checks=False))
        vals = {L: pearson(model_rank_by_L[L], br) for L in LAYERS}
        peak = max(vals, key=vals.get)
        out.append((tag, peak, vals[peak]))
    return out

for cond, mdl, roi in [('see', img_rank, 'ventral_visual'), ('read', cap_rank, 'language_ifg')]:
    logger.info(f'\n{cond} -> {roi}:')
    peaks = per_subject_peaks(mdl, cond, roi)
    from collections import Counter
    cnt = Counter(p[1] for p in peaks)
    for tag, peak, val in peaks:
        logger.info(f'  {tag:<14} peak@L{peak:<3} (r={val:+.3f})')
    logger.info(f'  peak-layer distribution: {dict(cnt)}')

# save
out = {
    'reliability': reliability,
    'see_obs': {g: obs_see.get(g, {}) for g in GRADIENT_ORDER},
    'read_obs': {g: obs_read.get(g, {}) for g in GRADIENT_ORDER},
    'see_p': {g: {L: float((nul_see[g][L] >= obs_see[g][L]).mean()) for L in LAYERS} for g in GRADIENT_ORDER if obs_see.get(g)},
    'read_p': {g: {L: float((nul_read[g][L] >= obs_read[g][L]).mean()) for L in LAYERS} for g in GRADIENT_ORDER if obs_read.get(g)},
}
with open(SEMREPS_DIR / 'crossmodal_validation.json', 'w') as f:
    json.dump(out, f, indent=2)
logger.info('')
logger.info(f'Saved {SEMREPS_DIR / "crossmodal_validation.json"}')
logger.info('Validation complete')
