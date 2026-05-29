"""SemReps encoding model (sub-01 LH): fit MLLM features -> brain on the train
items, evaluate on the 70 high-SNR test items (which carry BOTH see and read betas).

IMPORTANT: train IMAGE and train CAPTION are DISJOINT stimulus sets
(image-trial ids ~4897 vs caption-trial ids ~4889, only 211 overlap), so each
leg is fit on its own id set:
  SEE  leg: imgtrial features  -> betas_train_image  (image-trial ids)
  READ leg: caption  features  -> betas_train_caption (caption-trial ids)

Evaluation is on the shared 70 test items, where every item has see+read betas
and image+caption features. For each fitted encoder we report:
  within  = corr(pred, matched test betas)      e.g. see-encoder vs test SEE
  cross   = corr(pred, other   test betas)      e.g. see-encoder vs test READ
A real dissociation => within > cross, in the predicted layer/ROI cells.

Runs on HPC2.
"""
import numpy as np
import nibabel as nib
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json, logging

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

def load_betas_aligned(condition, coco_ids):
    """Load betas for exactly coco_ids (in order). Returns (array, kept_index)."""
    base = SEMREPS_DIR / 'surface' / HEMI / SUB / condition
    out, keep = [], []
    for j, cid in enumerate(coco_ids):
        pth = base / f'beta_{cid:06d}.gii'
        if pth.exists():
            out.append(nib.load(str(pth)).darrays[0].data); keep.append(j)
    return np.stack(out), np.array(keep)

# ---- features ----
logger.info('Loading MLLM features...')
imgtrial = np.load(RES / 'semreps_train_imgtrial_features.npz')   # SEE-leg train X (image ids)
captrain = np.load(RES / 'semreps_train_caption_features.npz')    # READ-leg train X (caption ids)
te_img = np.load(RES / 'semreps_test_features.npz')               # test image features (70)
te_cap = np.load(RES / 'semreps_test_caption_features.npz')       # test caption features (70)
img_ids = [int(c) for c in imgtrial['coco_ids']]
cap_ids = [int(c) for c in captrain['coco_ids']]
te_ids = [int(c) for c in te_img['coco_ids']]
assert [int(c) for c in te_cap['coco_ids']] == te_ids
logger.info(f'train img-trial={len(img_ids)}, train caption={len(cap_ids)}, test={len(te_ids)}')

# ---- brain ----
labels, names = load_annot()
name2idx = {nm: i for i, nm in enumerate(names)}
roi_vert = {g: np.array([v for rn in rl if rn in name2idx
                         for v in np.where(labels == name2idx[rn])[0]]) for g, rl in ROI_GROUPS.items()}

logger.info('Loading betas (train see/read aligned to their own id sets; test see+read)...')
see_tr, see_keep = load_betas_aligned('betas_train_image', img_ids)
read_tr, read_keep = load_betas_aligned('betas_train_caption', cap_ids)
see_te, _ = load_betas_aligned('betas_test_image', te_ids)
read_te, _ = load_betas_aligned('betas_test_caption', te_ids)
logger.info(f'see_tr={see_tr.shape} read_tr={read_tr.shape} see_te={see_te.shape} read_te={read_te.shape}')

# align train features to the betas that were actually found
def feat_stack(npz, keep):
    return {L: npz[f'layer_{L}'][keep] for L in LAYERS}
see_X = feat_stack(imgtrial, see_keep)
read_X = feat_stack(captrain, read_keep)


def fit_encoder(Xtr, Ytr):
    sx = StandardScaler().fit(Xtr)
    Xs = sx.transform(Xtr)
    ncomp = min(N_PCA, Xs.shape[0]-1, Xs.shape[1])
    pca = PCA(n_components=ncomp).fit(Xs)
    Z = pca.transform(Xs)
    cut = int(Z.shape[0]*0.8)
    best_a, best_s = ALPHAS[0], -1e9
    step = max(1, Ytr.shape[1]//200)
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Z[:cut], Ytr[:cut])
        pv = m.predict(Z[cut:])
        s = np.nanmean([stats.pearsonr(pv[:, k], Ytr[cut:, k])[0] for k in range(0, Ytr.shape[1], step)])
        if s > best_s: best_s, best_a = s, a
    m = Ridge(alpha=best_a).fit(Z, Ytr)
    return (sx, pca, m)

def predict(enc, X):
    sx, pca, m = enc
    return m.predict(pca.transform(sx.transform(X)))

def mean_r(pred, Y):
    rs = np.array([stats.pearsonr(pred[:, k], Y[:, k])[0] for k in range(Y.shape[1])])
    rs = rs[~np.isnan(rs)]
    return float(rs.mean()) if len(rs) else float('nan')

# ---- SEE leg: image_L -> see ; eval within(SEE) vs cross(READ) on test ----
logger.info('')
logger.info('='*78)
logger.info('SEE-ENCODER  image_L -> SEE.  test within=SEE  cross=READ   (within>cross = visual-specific)')
logger.info('='*78)
logger.info('ROI \\ layer   ' + ''.join([f'L{L:<10}' for L in LAYERS]))
see_res = {}
for g in GRAD:
    v = roi_vert[g]
    if len(v) < 20: continue
    row = {}
    for L in LAYERS:
        enc = fit_encoder(see_X[L], see_tr[:, v])
        pred = predict(enc, te_img[f'layer_{L}'])
        w = mean_r(pred, see_te[:, v]); c = mean_r(pred, read_te[:, v])
        row[L] = (w, c)
    see_res[g] = row
    peak = max(row, key=lambda L: row[L][0])
    logger.info(f'{g:<14}' + ''.join([f'{row[L][0]:+.3f}/{row[L][1]:+.3f} ' for L in LAYERS]) + f' peak@L{peak}')

# ---- READ leg: caption_L -> read ; eval within(READ) vs cross(SEE) on test ----
logger.info('')
logger.info('='*78)
logger.info('READ-ENCODER caption_L -> READ. test within=READ cross=SEE   (within>cross = language-specific)')
logger.info('='*78)
logger.info('ROI \\ layer   ' + ''.join([f'L{L:<10}' for L in LAYERS]))
read_res = {}
for g in GRAD:
    v = roi_vert[g]
    if len(v) < 20: continue
    row = {}
    for L in LAYERS:
        enc = fit_encoder(read_X[L], read_tr[:, v])
        pred = predict(enc, te_cap[f'layer_{L}'])
        w = mean_r(pred, read_te[:, v]); c = mean_r(pred, see_te[:, v])
        row[L] = (w, c)
    read_res[g] = row
    peak = max(row, key=lambda L: row[L][0])
    logger.info(f'{g:<14}' + ''.join([f'{row[L][0]:+.3f}/{row[L][1]:+.3f} ' for L in LAYERS]) + f' peak@L{peak}')

out = {
    'see_leg':  {g: {str(L): see_res[g][L] for L in see_res[g]} for g in see_res},
    'read_leg': {g: {str(L): read_res[g][L] for L in read_res[g]} for g in read_res},
    'note': 'each cell = (within_r, cross_r). see_leg within=SEE cross=READ; read_leg within=READ cross=SEE',
}
json.dump(out, open(RES / 'encoding_results.json', 'w'), indent=2)
logger.info('')
logger.info(f'Saved {RES/"encoding_results.json"}')
logger.info('Encoding complete')
