"""Noise-ceiling normalized encoding scores (SEE + READ legs, sub-01/02/03 LH).

Raw mean-over-ROI Pearson r underestimates how good an fMRI encoder is, because
(a) it isn't normalized by the reproducible-signal ceiling and (b) it averages
over unresponsive vertices. This script reports the field-standard view:

  ceiling[v]  = leave-one-subject-out inter-subject reliability of the 70-item
                test response at vertex v (corr of subject vs mean of others).
                Estimates the reproducible (explainable) signal at v.
  model_r[v]  = within-subject encoder prediction-vs-actual on the 70 test items.
  reliable    = vertices with ceiling > THRESH (real signal present).
  normalized  = mean(model_r over reliable) / mean(ceiling over reliable)
              = fraction of explainable signal the MLLM features capture.

Reuses the validated encoding machinery. Runs on HPC2 (thread-capped).
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os, json, logging, gc, sys

sys.path.insert(0, str(Path(__file__).parent))
import run_semreps_encoding_multi as E  # validated helpers + constants

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('noiseceil')

SUBJECTS = os.environ.get('SR_SUBS', 'sub-01,sub-02,sub-03,sub-04,sub-05,sub-07').split(',')
THRESH = 0.15           # reliable-vertex cutoff on the ceiling
LAYERS = E.LAYERS
GRAD = E.GRAD
N_PCA = E.N_PCA
ALPHAS = E.ALPHAS


def col_corr_vec(P, Y):
    """Per-column Pearson r between P and Y (both n x V) -> length-V vector."""
    Pc = P - P.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = (Pc * Yc).sum(0)
    den = np.sqrt((Pc ** 2).sum(0) * (Yc ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    return r  # may contain nan where den==0


def loo_ceiling(te_by_sub, vidx):
    """Leave-one-subject-out inter-subject reliability per vertex, averaged over
    subjects. te_by_sub: dict sub -> (70 x V) test betas. Returns length-len(vidx)."""
    subs = list(te_by_sub)
    rs = []
    for i, s in enumerate(subs):
        others = [te_by_sub[o][:, vidx] for o in subs if o != s]
        mean_other = np.mean(others, axis=0)
        rs.append(col_corr_vec(te_by_sub[s][:, vidx], mean_other))
    return np.nanmean(rs, axis=0)


def main():
    logger.info('Building feature indices...')
    img_seen, img_mats = E.build_feat_index(E.IMG_SRC)
    cap_seen, cap_mats = E.build_feat_index(E.CAP_SRC)
    test_img = np.load(E.RES / E.TEST_IMG)
    test_cap = np.load(E.RES / E.TEST_CAP)
    te_ids = [int(c) for c in test_img['coco_ids']]

    labels, names = E.load_annot()
    name2idx = {nm: i for i, nm in enumerate(names)}
    roi_vert = {g: np.array([v for rn in rl if rn in name2idx
                             for v in np.where(labels == name2idx[rn])[0]])
                for g, rl in E.ROI_GROUPS.items()}

    # ---- load all subjects' test betas (see + read) once ----
    see_te, read_te = {}, {}
    for s in SUBJECTS:
        a, _ = E.load_betas('betas_test_image', s, te_ids)
        b, _ = E.load_betas('betas_test_caption', s, te_ids)
        see_te[s], read_te[s] = a, b
        logger.info(f'  {s}: see_te={a.shape} read_te={b.shape}')

    # ---- per-ROI ceilings (computed once, layer-independent) ----
    ceil_see = {g: loo_ceiling(see_te, roi_vert[g]) for g in GRAD}
    ceil_read = {g: loo_ceiling(read_te, roi_vert[g]) for g in GRAD}
    for g in GRAD:
        nrel_s = int((ceil_see[g] > THRESH).sum()); nrel_r = int((ceil_read[g] > THRESH).sum())
        logger.info(f'  ceiling[{g}]: SEE mean={np.nanmean(ceil_see[g]):.3f} reliable={nrel_s}/{len(ceil_see[g])} '
                    f'| READ mean={np.nanmean(ceil_read[g]):.3f} reliable={nrel_r}/{len(ceil_read[g])}')

    # ---- fit encoder per subject, collect raw + normalized per ROI/layer ----
    def run(train_cond, test_feat, feat_seen, feat_mats, target_te, ceil):
        agg = {g: {L: {'raw': [], 'rawrel': [], 'ceilrel': []} for L in LAYERS} for g in GRAD}
        for sub in SUBJECTS:
            base = E.SEMREPS_DIR / 'surface' / E.HEMI / sub / train_cond
            ids = sorted(int(p.stem.split('_')[1]) for p in base.glob('beta_*.gii'))
            Ytr_full, keep_b = E.load_betas(train_cond, sub, ids)
            ids_kept = [ids[j] for j in keep_b]
            Xtr, keep_f = E.gather(feat_seen, feat_mats, ids_kept)
            Ytr_full = Ytr_full[keep_f]
            valid = [g for g in GRAD if len(roi_vert[g]) >= 20]
            allv, sl, off = [], {}, 0
            for g in valid:
                v = roi_vert[g]; allv.append(v); sl[g] = slice(off, off + len(v)); off += len(v)
            allv = np.concatenate(allv); Yall = Ytr_full[:, allv]
            ncomp = min(N_PCA, Ytr_full.shape[0] - 1, 4096)
            for L in LAYERS:
                sx = StandardScaler().fit(Xtr[L])
                pca = PCA(n_components=ncomp, svd_solver='randomized', random_state=0).fit(sx.transform(Xtr[L]))
                Ztr = pca.transform(sx.transform(Xtr[L]))
                Zte = pca.transform(sx.transform(test_feat[f'layer_{L}']))
                a = E.best_alpha(Ztr, Yall)
                m = Ridge(alpha=a).fit(Ztr, Yall)
                pred = m.predict(Zte)
                for g in valid:
                    mr = col_corr_vec(pred[:, sl[g]], target_te[sub][:, roi_vert[g]])
                    c = ceil[g]
                    mask = (c > THRESH) & np.isfinite(mr)
                    agg[g][L]['raw'].append(np.nanmean(mr))
                    if mask.sum() >= 10:
                        agg[g][L]['rawrel'].append(np.nanmean(mr[mask]))
                        agg[g][L]['ceilrel'].append(np.nanmean(c[mask]))
            del Ytr_full, Yall; gc.collect()
            logger.info(f'  done {sub} ({train_cond})')
        # collapse across subjects
        res = {}
        for g in GRAD:
            res[g] = {}
            for L in LAYERS:
                raw = np.mean(agg[g][L]['raw']) if agg[g][L]['raw'] else float('nan')
                rr = np.mean(agg[g][L]['rawrel']) if agg[g][L]['rawrel'] else float('nan')
                cc = np.mean(agg[g][L]['ceilrel']) if agg[g][L]['ceilrel'] else float('nan')
                norm = rr / cc if (cc and np.isfinite(cc) and cc > 0) else float('nan')
                res[g][L] = {'raw_all': float(raw), 'raw_reliable': float(rr),
                             'ceiling': float(cc), 'normalized': float(norm)}
        return res

    logger.info('=== SEE leg ===')
    see = run('betas_train_image', test_img, img_seen, img_mats, see_te, ceil_see)
    logger.info('=== READ leg ===')
    read = run('betas_train_caption', test_cap, cap_seen, cap_mats, read_te, ceil_read)

    def show(res, title):
        logger.info('=' * 70); logger.info(title); logger.info('=' * 70)
        logger.info(f'{"ROI":<16}{"rawALL":>8}{"rawREL":>8}{"ceiling":>9}{"NORM":>8}  (best layer)')
        for g in GRAD:
            best = max(res[g], key=lambda L: (res[g][L]['normalized'] if np.isfinite(res[g][L]['normalized']) else -9))
            c = res[g][best]
            logger.info(f'{g:<16}{c["raw_all"]:>8.3f}{c["raw_reliable"]:>8.3f}{c["ceiling"]:>9.3f}{c["normalized"]:>8.3f}  @L{best}')

    show(see, 'SEE leg (image->SEE): raw vs noise-ceiling-normalized')
    show(read, 'READ leg (caption->READ): raw vs noise-ceiling-normalized')
    json.dump({'thresh': THRESH, 'see': see, 'read': read,
               'ceiling_means': {g: {'see': float(np.nanmean(ceil_see[g])),
                                      'read': float(np.nanmean(ceil_read[g]))} for g in GRAD}},
              open(E.RES / 'noise_ceiling_results.json', 'w'), indent=2)
    logger.info(f'Saved {E.RES/"noise_ceiling_results.json"}')


if __name__ == '__main__':
    main()
