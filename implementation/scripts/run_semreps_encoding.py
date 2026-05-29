"""SemReps encoding model (sub-01 LH): fit MLLM features -> brain on the ~4.9k
train items, evaluate on the 70 high-SNR test items.

Per layer L x ROI, two encoders:
  image_L  -> SEE  brain   (does image rep predict image-viewing response?)
  caption_L-> READ brain   (does caption rep predict caption-reading response?)
Reports test mean_r (mean over vertices of Pearson(pred, actual)).

This is the higher-power complement to RSA: ridge can find the predictive
subspace even in the noisier READ betas, so it should rescue the weak leg if real.

Runs on HPC2.
"""
import numpy as np
import nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json, os, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('encoding')

SEMREPS_DIR = Path('/hpc2hdd/home/mzhang630/data/semreps')
RES = SEMREPS_DIR
LAYERS = [0, 4, 8, 12, 14, 16, 20, 24, 31]
SUB = 'sub-01'; HEMI = 'left'
N_PCA = 512
ALPHAS = [1e2, 1e3, 1e4, 1e5]

ROI_GROUPS = {
    'early_visual':    ['pericalcarine', 'cuneus', 'lingual'],
    'ventral_visual':  ['lateraloccipital', 'fusiform', 'parahippocampal', 'inferiortemporal'],
    'lateral_temporal':['superiortemporal', 'middletemporal', 'bankssts'],
    'language_ifg':    ['parsopercularis', 'parstriangularis', 'parsorbitalis'],
    'parietal_assoc':  ['supramarginal', 'inferiorparietal'],
}
GRAD = ['early_visual', 'ventral_visual', 'lateral_temporal', 'parietal_assoc', 'language_ifg']


def load_annot():
    p = '/hpc2hdd/home/mzhang630/data/nsd/nsddata/freesurfer/fsaverage/label/lh.aparc.annot'
    labels, ctab, names = nib.freesurfer.read_annot(p)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels, names

def load_betas(condition, coco_ids):
    base = SEMREPS_DIR / 'surface' / HEMI / SUB / condition
    out = []
    keep = []
    for j, cid in enumerate(coco_ids):
        pth = base / f'beta_{cid:06d}.gii'
        if pth.exists():
            out.append(nib.load(str(pth)).darrays[0].data); keep.append(j)
    return np.stack(out), np.array(keep)

# ---- load features ----
logger.info('Loading MLLM features (train + test, image + caption)...')
tr_img = np.load(RES / 'semreps_train_image_features.npz')
tr_cap = np.load(RES / 'semreps_train_caption_features.npz')
te_img = np.load(RES / 'semreps_test_features.npz')
te_cap = np.load(RES / 'semreps_test_caption_features.npz')
tr_ids = [int(c) for c in tr_img['coco_ids']]
assert [int(c) for c in tr_cap['coco_ids']] == tr_ids
te_ids = [int(c) for c in te_img['coco_ids']]
logger.info(f'train items={len(tr_ids)}, test items={len(te_ids)}')

# ---- brain ----
labels, names = load_annot()
name2idx = {nm: i for i, nm in enumerate(names)}
roi_vert = {}
for g, rl in ROI_GROUPS.items():
    idx = []
    for rn in rl:
        if rn in name2idx:
            idx.extend(np.where(labels == name2idx[rn])[0].tolist())
    roi_vert[g] = np.array(idx)

logger.info('Loading sub-01 LH betas (train + test, see + read)...')
see_tr, see_tr_keep = load_betas('betas_train_image', tr_ids)
read_tr, read_tr_keep = load_betas('betas_train_caption', tr_ids)
see_te, see_te_keep = load_betas('betas_test_image', te_ids)
read_te, read_te_keep = load_betas('betas_test_caption', te_ids)
logger.info(f'train see={see_tr.shape}, read={read_tr.shape}; test see={see_te.shape}, read={read_te.shape}')


def encode_eval(Xtr, Ytr, Xte, Yte):
    """Fit PCA+Ridge on train, return test mean_r over vertices (CV alpha by train R)."""
    sx = StandardScaler().fit(Xtr)
    Xtr_s = sx.transform(Xtr); Xte_s = sx.transform(Xte)
    ncomp = min(N_PCA, Xtr_s.shape[0]-1, Xtr_s.shape[1])
    pca = PCA(n_components=ncomp).fit(Xtr_s)
    Ztr = pca.transform(Xtr_s); Zte = pca.transform(Xte_s)
    # simple alpha pick: split train 80/20
    ns = Ztr.shape[0]; cut = int(ns*0.8)
    best_a, best_s = ALPHAS[0], -1e9
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Ztr[:cut], Ytr[:cut])
        pv = m.predict(Ztr[cut:])
        # mean corr over a sample of vertices
        s = np.mean([stats.pearsonr(pv[:,k], Ytr[cut:,k])[0]
                     for k in range(0, Ytr.shape[1], max(1, Ytr.shape[1]//200))])
        if s > best_s: best_s, best_a = s, a
    m = Ridge(alpha=best_a).fit(Ztr, Ytr)
    pred = m.predict(Zte)
    rs = np.array([stats.pearsonr(pred[:,k], Yte[:,k])[0] for k in range(Yte.shape[1])])
    rs = rs[~np.isnan(rs)]
    return float(rs.mean()), best_a

logger.info('')
logger.info('='*70)
logger.info('ENCODING: test mean_r  (image_L -> SEE)')
logger.info('='*70)
logger.info('ROI \\ layer   ' + ''.join([f'L{L:<7}' for L in LAYERS]))
see_curve = {}
for g in GRAD:
    v = roi_vert[g]
    if len(v) < 20: continue
    Ytr = see_tr[:, v]; Yte = see_te[:, v]
    row = {}
    for L in LAYERS:
        r, a = encode_eval(tr_img[f'layer_{L}'], Ytr, te_img[f'layer_{L}'], Yte)
        row[L] = r
    see_curve[g] = row
    peak = max(row, key=row.get)
    logger.info(f'{g:<14}' + ''.join([f'{row[L]:<8.3f}' for L in LAYERS]) + f' peak@L{peak}')

logger.info('')
logger.info('='*70)
logger.info('ENCODING: test mean_r  (caption_L -> READ)')
logger.info('='*70)
logger.info('ROI \\ layer   ' + ''.join([f'L{L:<7}' for L in LAYERS]))
read_curve = {}
for g in GRAD:
    v = roi_vert[g]
    if len(v) < 20: continue
    Ytr = read_tr[:, v]; Yte = read_te[:, v]
    row = {}
    for L in LAYERS:
        r, a = encode_eval(tr_cap[f'layer_{L}'], Ytr, te_cap[f'layer_{L}'], Yte)
        row[L] = r
    read_curve[g] = row
    peak = max(row, key=row.get)
    logger.info(f'{g:<14}' + ''.join([f'{row[L]:<8.3f}' for L in LAYERS]) + f' peak@L{peak}')

# cross controls: image->READ and caption->SEE (should be weaker if dissociation real)
logger.info('')
logger.info('='*70)
logger.info('CROSS CONTROLS (mean over ROIs): image->READ vs caption->READ ; caption->SEE vs image->SEE')
logger.info('='*70)
for L in [8, 14]:
    rows = []
    for g in GRAD:
        v = roi_vert[g]
        if len(v) < 20: continue
        i2r,_ = encode_eval(tr_img[f'layer_{L}'], read_tr[:,v], te_img[f'layer_{L}'], read_te[:,v])
        c2s,_ = encode_eval(tr_cap[f'layer_{L}'], see_tr[:,v], te_cap[f'layer_{L}'], see_te[:,v])
        rows.append((g, i2r, c2s))
    logger.info(f'L{L}:')
    for g,i2r,c2s in rows:
        logger.info(f'  {g:<16} image->READ={i2r:+.3f}  caption->SEE={c2s:+.3f}')

out = {'see_curve': see_curve, 'read_curve': read_curve}
json.dump(out, open(RES / 'encoding_results.json', 'w'), indent=2)
logger.info('')
logger.info(f'Saved {RES/"encoding_results.json"}')
logger.info('Encoding complete')
