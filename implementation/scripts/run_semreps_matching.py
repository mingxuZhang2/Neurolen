"""SemReps matching-level analysis — does the MLLM share the BRAIN's cross-modal code?

This is the direct test of the core question ("does an MLLM process image+text the
way the human brain does cross-modal matching"), upgraded from the earlier low-power
70-item RSA to a high-power *cross-modal transfer* design that mirrors the original
SemReps paper's modality-agnostic decoding (Nikolaus et al. 2025, eLife).

Per subject, per layer L (LLaVA/LLaMA residual-stream hidden states — image tokens and
caption tokens live in the SAME space, which is what makes transfer well-defined):

  1. JOINT scaler+PCA fit on POOLED train features (image trials + caption trials)
     -> a shared subspace Z that both modalities are projected into.
  2. Fit two readouts on that subject's OWN train betas:
       W_img : Z(image feat)   -> betas_train_image  (the "seeing readout")
       W_cap : Z(caption feat) -> betas_train_caption (the "reading readout")
     ~4000-5000 trials each -> high power (vs 70-item RSA).
  3. On the shared 70 test items (which carry BOTH see+read betas) compute a 2x2:
       within_V  = W_img(test image feat)   vs SEE  test betas   [matched, should be high]
       within_L  = W_cap(test caption feat) vs READ test betas   [matched]
       xfer_VtoL = W_img(test caption feat)  vs READ test betas   *** modality-agnostic test
       xfer_LtoV = W_cap(test image feat)    vs SEE  test betas   *** modality-agnostic test
     transfer > 0 in a region  => brain+MLLM share ONE cross-modal readout there
                                  (image<->text converge; core question = YES, locally).
     transfer ~ 0 while within high => modality-SPECIFIC; no shared code (core question = NO).
  4. Permutation null: shuffle the 70 test scene labels, recompute the transfer terms
     -> per-subject p-value that transfer exceeds chance.

Across-subject random effects (paired within-vs-transfer, one-sample transfer-vs-0) are
computed at each ROI's peak-within layer. Runs on HPC2 (login node OOMs -> use SLURM).
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import json, logging, gc, os

import run_semreps_encoding_multi as E  # reuse data/feature/ROI utilities

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('matching')

SUBJECTS = os.environ.get('SR_SUBS', 'sub-01,sub-02,sub-03,sub-04,sub-05,sub-07').split(',')
NPERM = int(os.environ.get('SR_NPERM', '1000'))
LAYERS = E.LAYERS
GRAD = E.GRAD
N_PCA = E.N_PCA
SEMREPS_DIR = E.SEMREPS_DIR
RES = E.RES
HEMI = E.HEMI

try:
    from scipy import stats as _st
except Exception:
    _st = None


def perm_null(pred, target, nperm, rng):
    """Observed mean-over-vertices column correlation + permutation p (shuffle scene rows).

    pred, target: (n_items, V). Null = match each prediction row to a shuffled target row.
    p = fraction of null r >= observed (one-sided: transfer above chance)."""
    Pc = pred - pred.mean(0, keepdims=True)
    Tc = target - target.mean(0, keepdims=True)
    Pn = Pc / (np.sqrt((Pc ** 2).sum(0)) + 1e-12)
    Tn = Tc / (np.sqrt((Tc ** 2).sum(0)) + 1e-12)
    obs = float(np.nanmean((Pn * Tn).sum(0)))
    n = pred.shape[0]
    null = np.empty(nperm)
    for k in range(nperm):
        perm = rng.permutation(n)
        null[k] = np.nanmean((Pn * Tn[perm]).sum(0))
    p = (1 + int(np.sum(null >= obs))) / (nperm + 1)
    return obs, float(p)


def _load_train(sub, cond, feat_seen, feat_mats):
    """Load one subject's train betas + aligned MLLM features for a condition."""
    base = SEMREPS_DIR / 'surface' / HEMI / sub / cond
    ids = sorted(int(p.stem.split('_')[1]) for p in base.glob('beta_*.gii'))
    Y, keep_b = E.load_betas(cond, sub, ids)
    ids_kept = [ids[j] for j in keep_b]
    X, keep_f = E.gather(feat_seen, feat_mats, ids_kept)
    Y = Y[keep_f]
    return X, Y


def run_subject(sub, img_seen, img_mats, cap_seen, cap_mats, test_img, test_cap,
                te_ids, roi_vert, rng):
    see_te, _ = E.load_betas('betas_test_image', sub, te_ids)
    read_te, _ = E.load_betas('betas_test_caption', sub, te_ids)
    if see_te.shape[0] != len(te_ids) or read_te.shape[0] != len(te_ids):
        logger.warning(f'  {sub}: test betas incomplete, skipping')
        return None, None

    Ximg, Ysee = _load_train(sub, 'betas_train_image', img_seen, img_mats)
    Xcap, Yread = _load_train(sub, 'betas_train_caption', cap_seen, cap_mats)
    logger.info(f'  [{sub}] train_img={Ysee.shape[0]} train_cap={Yread.shape[0]}')

    valid = [g for g in GRAD if len(roi_vert[g]) >= 20]
    allv, slices, off = [], {}, 0
    for g in valid:
        v = roi_vert[g]
        allv.append(v); slices[g] = slice(off, off + len(v)); off += len(v)
    allv = np.concatenate(allv)
    Ysee_all, Yread_all = Ysee[:, allv], Yread[:, allv]

    res = {g: {} for g in valid}
    preds = {}  # L -> (p_img, p_imgFromCap, p_cap, p_capFromImg) over allv
    for L in LAYERS:
        # shared subspace: joint scaler+PCA on pooled image+caption train features
        pooled = np.vstack([Ximg[L], Xcap[L]])
        sx = StandardScaler().fit(pooled)
        ncomp = min(N_PCA, pooled.shape[0] - 1, 4096)
        pca = PCA(n_components=ncomp, svd_solver='randomized', random_state=0).fit(sx.transform(pooled))
        Z = lambda M: pca.transform(sx.transform(M))
        Zimg_tr, Zcap_tr = Z(Ximg[L]), Z(Xcap[L])
        Zte_img, Zte_cap = Z(test_img[f'layer_{L}']), Z(test_cap[f'layer_{L}'])
        del pooled; gc.collect()

        a_img = E.best_alpha(Zimg_tr, Ysee_all)
        Wimg = Ridge(alpha=a_img).fit(Zimg_tr, Ysee_all)
        a_cap = E.best_alpha(Zcap_tr, Yread_all)
        Wcap = Ridge(alpha=a_cap).fit(Zcap_tr, Yread_all)

        p_img = Wimg.predict(Zte_img)          # seeing readout, image input
        p_imgFromCap = Wimg.predict(Zte_cap)   # seeing readout, CAPTION input -> xfer V->L
        p_cap = Wcap.predict(Zte_cap)          # reading readout, caption input
        p_capFromImg = Wcap.predict(Zte_img)   # reading readout, IMAGE input -> xfer L->V
        preds[L] = (p_img, p_imgFromCap, p_cap, p_capFromImg)

        for g in valid:
            sl, v = slices[g], roi_vert[g]
            withinV = E.col_corr_mean(p_img[:, sl], see_te[:, v])
            xferVL = E.col_corr_mean(p_imgFromCap[:, sl], read_te[:, v])
            withinL = E.col_corr_mean(p_cap[:, sl], read_te[:, v])
            xferLV = E.col_corr_mean(p_capFromImg[:, sl], see_te[:, v])
            res[g][L] = (withinV, xferVL, withinL, xferLV)

    # permutation null at each ROI's peak-within-V layer
    perm = {}
    for g in valid:
        peakL = max(res[g], key=lambda L: res[g][L][0])
        sl, v = slices[g], roi_vert[g]
        _, p_imgFromCap, _, p_capFromImg = preds[peakL]
        oVL, pVL = perm_null(p_imgFromCap[:, sl], read_te[:, v], NPERM, rng)
        oLV, pLV = perm_null(p_capFromImg[:, sl], see_te[:, v], NPERM, rng)
        perm[g] = {'peakL': int(peakL), 'xferVL': [oVL, pVL], 'xferLV': [oLV, pLV]}

    del Ximg, Xcap, Ysee, Yread, Ysee_all, Yread_all, see_te, read_te, preds
    gc.collect()
    return res, perm


def fmt(res, title):
    logger.info('=' * 92)
    logger.info(title)
    logger.info('cells = withinV / xferV->L | withinL / xferL->V   (xfer = modality-agnostic transfer)')
    logger.info('=' * 92)
    for g in GRAD:
        if g not in res:
            continue
        row = res[g]
        peak = max(row, key=lambda L: row[L][0])
        s = ''.join([f'{row[L][0]:+.2f}/{row[L][1]:+.2f}|{row[L][2]:+.2f}/{row[L][3]:+.2f}  ' for L in LAYERS])
        logger.info(f'{g:<16}' + s + f' peakV@L{peak}')


def across_subject_stats(per_sub):
    """Paired within-vs-transfer and one-sample transfer-vs-0, at each ROI peak-within layer."""
    out = {}
    for g in GRAD:
        rows = [d[g] for d in per_sub if g in d]
        if len(rows) < 2:
            continue
        # choose layer maximizing mean withinV across subjects
        layers = sorted(set().union(*[set(r.keys()) for r in rows]))
        meanV = {L: np.nanmean([r[L][0] for r in rows if L in r]) for L in layers}
        L = max(meanV, key=lambda k: meanV[k])
        wV = np.array([r[L][0] for r in rows if L in r])
        xVL = np.array([r[L][1] for r in rows if L in r])
        wL = np.array([r[L][2] for r in rows if L in r])
        xLV = np.array([r[L][3] for r in rows if L in r])
        rec = {'layer': int(L), 'n': int(len(wV)),
               'withinV': float(wV.mean()), 'xferVL': float(xVL.mean()),
               'withinL': float(wL.mean()), 'xferLV': float(xLV.mean())}
        if _st is not None:
            # transfer vs 0 (is there ANY shared code?) and within vs transfer (specificity)
            rec['xferVL_vs0_t'], rec['xferVL_vs0_p'] = [float(x) for x in _st.ttest_1samp(xVL, 0)]
            rec['xferLV_vs0_t'], rec['xferLV_vs0_p'] = [float(x) for x in _st.ttest_1samp(xLV, 0)]
            rec['V_within_vs_xfer_t'], rec['V_within_vs_xfer_p'] = [float(x) for x in _st.ttest_rel(wV, xVL)]
        out[g] = rec
    return out


def main():
    logger.info(f'SUBJECTS={SUBJECTS}  NPERM={NPERM}')
    logger.info('Building feature indices (image + caption)...')
    img_seen, img_mats = E.build_feat_index(E.IMG_SRC)
    cap_seen, cap_mats = E.build_feat_index(E.CAP_SRC)
    test_img = np.load(RES / E.TEST_IMG)
    test_cap = np.load(RES / E.TEST_CAP)
    te_ids = [int(c) for c in test_img['coco_ids']]
    assert [int(c) for c in test_cap['coco_ids']] == te_ids

    labels, names = E.load_annot()
    name2idx = {nm: i for i, nm in enumerate(names)}
    roi_vert = {g: np.array([v for rn in rl if rn in name2idx
                             for v in np.where(labels == name2idx[rn])[0]])
                for g, rl in E.ROI_GROUPS.items()}

    rng = np.random.default_rng(0)
    per_sub, perms = [], {}
    out = {'note': 'cell=(withinV,xferVtoL,withinL,xferLtoV). xfer>0 => shared cross-modal code',
           'nperm': NPERM, 'per_subject': {}, 'permutation': {}}
    # resume: reuse already-computed subjects from a prior (possibly cancelled) run
    ckpt = RES / 'matching_results.json'
    done = set()
    if ckpt.exists():
        prev = json.load(open(ckpt))
        if prev.get('nperm') == NPERM:
            for s, d in prev.get('per_subject', {}).items():
                res = {g: {int(L): tuple(d[g][L]) for L in d[g]} for g in d}
                per_sub.append(res); done.add(s)
                out['per_subject'][s] = d
                out['permutation'][s] = prev.get('permutation', {}).get(s, {})
            if done:
                logger.info(f'RESUME: reusing {sorted(done)} from checkpoint')
    for sub in SUBJECTS:
        if sub in done:
            logger.info(f'\n########## {sub} (already done, skip) ##########')
            continue
        logger.info(f'\n################## {sub} ##################')
        res, perm = run_subject(sub, img_seen, img_mats, cap_seen, cap_mats,
                                test_img, test_cap, te_ids, roi_vert, rng)
        if res is None:
            continue
        fmt(res, f'{sub} 2x2 transfer')
        for g in perm:
            logger.info(f'  perm[{sub}/{g}] peakV@L{perm[g]["peakL"]}  '
                        f'xferV->L r={perm[g]["xferVL"][0]:+.3f} p={perm[g]["xferVL"][1]:.3f} | '
                        f'xferL->V r={perm[g]["xferLV"][0]:+.3f} p={perm[g]["xferLV"][1]:.3f}')
        per_sub.append(res)
        perms[sub] = perm
        out['per_subject'][sub] = {g: {str(L): list(res[g][L]) for L in res[g]} for g in res}
        out['permutation'][sub] = perm
        # checkpoint after EACH subject (cluster scancel-resilient): persist partial results
        out['across_subject'] = across_subject_stats(per_sub)
        json.dump(out, open(RES / 'matching_results.json', 'w'), indent=2)
        logger.info(f'  [checkpoint] saved {len(per_sub)} subject(s) -> matching_results.json')

    stats = across_subject_stats(per_sub)
    out['across_subject'] = stats
    logger.info('\n@@@@@@@@@@@@@@@@ ACROSS-SUBJECT (random effects) @@@@@@@@@@@@@@@@')
    for g, rec in stats.items():
        logger.info(f'{g:<16} L{rec["layer"]}  withinV={rec["withinV"]:+.3f} xferV->L={rec["xferVL"]:+.3f} '
                    f'(vs0 p={rec.get("xferVL_vs0_p", float("nan")):.3f}) | '
                    f'withinL={rec["withinL"]:+.3f} xferL->V={rec["xferLV"]:+.3f} '
                    f'(vs0 p={rec.get("xferLV_vs0_p", float("nan")):.3f})')

    json.dump(out, open(RES / 'matching_results.json', 'w'), indent=2)
    logger.info(f'\nSaved {RES / "matching_results.json"}')
    logger.info('Done')


if __name__ == '__main__':
    main()
