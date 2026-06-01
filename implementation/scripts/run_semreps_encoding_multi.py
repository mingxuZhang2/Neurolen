"""Multi-subject SemReps encoding (vectorized).

For each subject we fit two encoders on that subject's OWN train stimuli/betas
and evaluate on the shared 70 test items (which carry BOTH see+read betas):

  SEE  leg: image  features -> betas_train_image   ; test within=SEE  cross=READ
  READ leg: caption features -> betas_train_caption ; test within=READ cross=SEE

within > cross in the predicted cells == a real modality-specific dissociation.

Subjects have DISJOINT train stimuli, so MLLM features are gathered per id from
a union of feature sources (sub-01 already-extracted + newly-extracted s23).

Speed: per-vertex Pearson and the alpha search are fully vectorized (column-wise
correlation via matrix ops), replacing the old python pearsonr loops
(~2.5 h/subject -> a few min/subject).

Runs on HPC2.
"""
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json, logging, gc, os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('enc_multi')

SEMREPS_DIR = Path('/hpc2hdd/home/mzhang630/data/semreps')
RES = SEMREPS_DIR
# Feature set / hemisphere are env-overridable so the same pipeline runs on LLaVA
# (default), a CLIP baseline (SR_LAYERS=0 + clip_*.npz), or the right hemisphere.
LAYERS = [int(x) for x in os.environ.get('SR_LAYERS', '0,4,8,12,14,16,20,24,31').split(',')]
HEMI = os.environ.get('SR_HEMI', 'left')
SUBJECTS = os.environ.get('SR_SUBS', 'sub-01,sub-02,sub-03').split(',')
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

# image-feature and caption-feature sources (each: coco_ids + layer_{L})
IMG_SRC = os.environ.get('SR_IMG_SRC', 'semreps_train_imgtrial_features.npz,'
                         'semreps_s23_image_features.npz,'
                         'semreps_s457_image_features.npz').split(',')
CAP_SRC = os.environ.get('SR_CAP_SRC', 'semreps_train_caption_features.npz,'
                         'semreps_s23_caption_features.npz,'
                         'semreps_s457_caption_features.npz').split(',')
TEST_IMG = os.environ.get('SR_TEST_IMG', 'semreps_test_features.npz')
TEST_CAP = os.environ.get('SR_TEST_CAP', 'semreps_test_caption_features.npz')


def load_annot():
    p = '/hpc2hdd/home/mzhang630/data/nsd/nsddata/freesurfer/fsaverage/label/lh.aparc.annot'
    labels, ctab, names = nib.freesurfer.read_annot(p)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels, names


def build_feat_index(sources):
    """Return {layer: (ids_array, mat)} and id->row dict, concatenated over sources
    (later sources do not override earlier; we keep first occurrence)."""
    seen = {}
    per_layer_rows = {L: [] for L in LAYERS}
    ids_order = []
    for src in sources:
        p = RES / src
        if not p.exists():
            logger.warning(f'feature source missing: {src}')
            continue
        z = np.load(p)
        cids = [int(c) for c in z['coco_ids']]
        # load each layer array ONCE into memory (npz access re-reads whole array)
        layer_arr = {L: np.asarray(z[f'layer_{L}']) for L in LAYERS}
        for i, cid in enumerate(cids):
            if cid in seen:
                continue
            seen[cid] = len(ids_order)
            ids_order.append(cid)
            for L in LAYERS:
                per_layer_rows[L].append(layer_arr[L][i])
    mats = {L: np.asarray(per_layer_rows[L], dtype=np.float32) for L in LAYERS}
    logger.info(f'  feature index built: {len(ids_order)} ids from {sources}')
    return seen, mats


def gather(seen, mats, ids):
    """Return (X_by_layer dict, keep_mask) for ids present in the index."""
    rows = [seen[i] for i in ids if i in seen]
    keep = np.array([j for j, i in enumerate(ids) if i in seen])
    X = {L: mats[L][rows] for L in LAYERS}
    return X, keep


def load_betas(condition, sub, coco_ids):
    base = SEMREPS_DIR / 'surface' / HEMI / sub / condition
    out, keep = [], []
    for j, cid in enumerate(coco_ids):
        pth = base / f'beta_{cid:06d}.gii'
        if pth.exists():
            out.append(nib.load(str(pth)).darrays[0].data); keep.append(j)
    if not out:
        return np.zeros((0, 163842), np.float32), np.array([], int)
    return np.stack(out).astype(np.float32), np.array(keep)


def col_corr_mean(P, Y):
    """Mean over vertices of column-wise Pearson r between P and Y (both n x V)."""
    Pc = P - P.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = (Pc * Yc).sum(0)
    den = np.sqrt((Pc ** 2).sum(0) * (Yc ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    r = r[np.isfinite(r)]
    return float(r.mean()) if r.size else float('nan')


def best_alpha(Ztr, Ytr):
    cut = int(Ztr.shape[0] * 0.8)
    best_a, best_s = ALPHAS[0], -1e9
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Ztr[:cut], Ytr[:cut])
        s = col_corr_mean(m.predict(Ztr[cut:]), Ytr[cut:])
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def run_leg(name, feat_seen, feat_mats, test_feat, train_cond, sub,
            roi_vert, within_te, cross_te):
    """Fit feat->train_cond betas for `sub`, eval within/cross on shared test.

    Speed: scaler+PCA fit ONCE per layer (independent of ROI); a single
    multi-output Ridge per layer over all ROI vertices concatenated, then split
    back per ROI. Returns {roi: {layer: (within_r, cross_r)}}.
    """
    base = SEMREPS_DIR / 'surface' / HEMI / sub / train_cond
    ids = sorted(int(p.stem.split('_')[1]) for p in base.glob('beta_*.gii'))
    Ytr_full, keep_b = load_betas(train_cond, sub, ids)
    ids_kept = [ids[j] for j in keep_b]
    Xtr, keep_f = gather(feat_seen, feat_mats, ids_kept)
    Ytr_full = Ytr_full[keep_f]  # align betas to features found
    logger.info(f'  [{sub}/{name}] train items={Ytr_full.shape[0]}')

    valid = [g for g in GRAD if len(roi_vert[g]) >= 20]
    # concatenated vertex index + per-ROI slices into the concatenation
    allv, slices, off = [], {}, 0
    for g in valid:
        v = roi_vert[g]
        allv.append(v); slices[g] = slice(off, off + len(v)); off += len(v)
    allv = np.concatenate(allv)
    Ytr_all = Ytr_full[:, allv]

    res = {g: {} for g in valid}
    ncomp = min(N_PCA, Ytr_full.shape[0] - 1, 4096)
    for L in LAYERS:
        sx = StandardScaler().fit(Xtr[L])
        pca = PCA(n_components=ncomp, svd_solver='randomized', random_state=0).fit(sx.transform(Xtr[L]))
        Ztr = pca.transform(sx.transform(Xtr[L]))
        Zte = pca.transform(sx.transform(test_feat[f'layer_{L}']))
        a = best_alpha(Ztr, Ytr_all)
        m = Ridge(alpha=a).fit(Ztr, Ytr_all)
        pred_all = m.predict(Zte)  # (70, len(allv))
        for g in valid:
            sl = slices[g]; v = roi_vert[g]
            w = col_corr_mean(pred_all[:, sl], within_te[:, v])
            c = col_corr_mean(pred_all[:, sl], cross_te[:, v])
            res[g][L] = (w, c)
    del Ytr_full, Ytr_all; gc.collect()
    return res


def fmt(res, title):
    logger.info('=' * 78)
    logger.info(title)
    logger.info('=' * 78)
    logger.info('ROI \\ layer   ' + ''.join([f'L{L:<10}' for L in LAYERS]))
    for g in GRAD:
        if g not in res:
            continue
        row = res[g]
        peak = max(row, key=lambda L: row[L][0])
        logger.info(f'{g:<14}' + ''.join([f'{row[L][0]:+.3f}/{row[L][1]:+.3f} ' for L in LAYERS]) + f' peak@L{peak}')


def avg_results(per_sub_list):
    """Average (within,cross) across subjects per roi/layer."""
    out = {}
    for g in GRAD:
        rows = [d[g] for d in per_sub_list if g in d]
        if not rows:
            continue
        out[g] = {}
        for L in LAYERS:
            ws = [r[L][0] for r in rows if L in r]
            cs = [r[L][1] for r in rows if L in r]
            out[g][L] = (float(np.nanmean(ws)), float(np.nanmean(cs)))
    return out


def main():
    logger.info('Building feature indices...')
    img_seen, img_mats = build_feat_index(IMG_SRC)
    cap_seen, cap_mats = build_feat_index(CAP_SRC)
    test_img = np.load(RES / TEST_IMG)
    test_cap = np.load(RES / TEST_CAP)
    te_ids = [int(c) for c in test_img['coco_ids']]
    assert [int(c) for c in test_cap['coco_ids']] == te_ids

    labels, names = load_annot()
    name2idx = {nm: i for i, nm in enumerate(names)}
    roi_vert = {g: np.array([v for rn in rl if rn in name2idx
                             for v in np.where(labels == name2idx[rn])[0]])
                for g, rl in ROI_GROUPS.items()}

    all_see, all_read = [], []
    out = {'per_subject': {}, 'note': 'cell=(within,cross). see:within=SEE cross=READ; read:within=READ cross=SEE'}
    for sub in SUBJECTS:
        logger.info('')
        logger.info(f'################## {sub} ##################')
        see_te, _ = load_betas('betas_test_image', sub, te_ids)
        read_te, _ = load_betas('betas_test_caption', sub, te_ids)
        if see_te.shape[0] != len(te_ids) or read_te.shape[0] != len(te_ids):
            logger.warning(f'  {sub}: test betas incomplete (see={see_te.shape[0]} read={read_te.shape[0]}), skipping')
            continue
        see_res = run_leg('SEE', img_seen, img_mats, test_img, 'betas_train_image',
                          sub, roi_vert, within_te=see_te, cross_te=read_te)
        read_res = run_leg('READ', cap_seen, cap_mats, test_cap, 'betas_train_caption',
                           sub, roi_vert, within_te=read_te, cross_te=see_te)
        fmt(see_res, f'{sub} SEE-ENCODER image_L->SEE  (within=SEE cross=READ)')
        fmt(read_res, f'{sub} READ-ENCODER caption_L->READ (within=READ cross=SEE)')
        all_see.append(see_res); all_read.append(read_res)
        out['per_subject'][sub] = {
            'see':  {g: {str(L): see_res[g][L] for L in see_res[g]} for g in see_res},
            'read': {g: {str(L): read_res[g][L] for L in read_res[g]} for g in read_res},
        }
        del see_te, read_te; gc.collect()

    logger.info('')
    logger.info('@@@@@@@@@@@@@@@@@@ MEAN ACROSS SUBJECTS @@@@@@@@@@@@@@@@@@')
    msee, mread = avg_results(all_see), avg_results(all_read)
    fmt(msee, f'MEAN(n={len(all_see)}) SEE-ENCODER (within=SEE cross=READ)')
    fmt(mread, f'MEAN(n={len(all_read)}) READ-ENCODER (within=READ cross=SEE)')
    out['mean'] = {
        'n': len(all_see),
        'see':  {g: {str(L): msee[g][L] for L in msee[g]} for g in msee},
        'read': {g: {str(L): mread[g][L] for L in mread[g]} for g in mread},
    }
    json.dump(out, open(RES / 'encoding_results_multi.json', 'w'), indent=2)
    logger.info(f'Saved {RES/"encoding_results_multi.json"}')
    logger.info('Done')


if __name__ == '__main__':
    main()
