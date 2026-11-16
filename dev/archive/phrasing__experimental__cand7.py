"""cand7.py (wip37): candidates on EVERY bar, same normalised features as cand2.py, + cand%4/%8/%16 flags.
Usage (from this folder, after build.py made ds.pkl):
    python cand7.py                 # features -> c7_feat.npy (cached), 5-fold grouped CV -> c7_os.npy c7_oe.npy c7_meta.pkl
    python cand7.py --final         # additionally fits final start/end models on ALL tracks -> c7_model_start.pkl c7_model_end.pkl
Options: --neg 0.3 (fraction of negative rows used for training), --iter 150, --maxtracks N (quick test)
Memory: ~220k rows x ~1250 float32 = ~1.1 GB (feature cache is a memmap). Uses all CPU cores via HGB/OpenMP.
"""
import pickle, sys, os, time, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--neg', type=float, default=0.3)
ap.add_argument('--iter', type=int, default=150)
ap.add_argument('--maxtracks', type=int, default=0)
ap.add_argument('--final', action='store_true')
ap.add_argument('--folds', type=int, default=5)
a = ap.parse_args()
src = open('bars.py').read().split('X=[];Y=[]')[0]; exec(src)       # out, drop_ends, barfeat
from labels2 import labels
from beatfeats import beat_feats
from bandfeats import band_feats
from adfeats import ad_feats
GEN=['Riddim','Tearout','DnB','2-Step','UKG (4x4 only)','140','Jungle','Liquid Riddim','Dubstep']
from sklearn.ensemble import HistGradientBoostingClassifier as HG
from sklearn.model_selection import GroupKFold
OFF = [-8, -4, -1, 0, 1, 4, 8]
COLS = list(range(0, 12)) + list(range(36, 96)) + list(range(120, 132))

def norm_feats(t):
    BE = t['BE']; nb = len(BE) // 4
    X = BE[:nb*4].reshape(nb, 4, 4, 3); m = X.mean(1).reshape(nb, 12)
    lm = np.log1p(m)
    med = np.median(lm, 0); rng = np.percentile(lm, 90, 0) - np.percentile(lm, 10, 0) + 1e-3
    Z = (lm - med) / rng
    q = np.percentile(lm, [20, 50, 80], 0)
    lvl = np.stack([(lm > q[2]), (lm < q[0])], 0).astype(np.float32)
    F = [Z, lvl[0], lvl[1]]
    c = np.cumsum(np.vstack([np.zeros((1, 12)), Z]), 0)
    def mean(a_, b_):
        a_ = np.clip(a_, 0, nb); b_ = np.clip(b_, 0, nb); ok = b_ > a_
        return np.where(ok[:, None], (c[np.maximum(b_, a_+1).clip(0, nb)] - c[a_]) / np.maximum(b_ - a_, 1)[:, None], 0)
    i = np.arange(nb)
    for w in (2, 4, 8, 16):
        bef = mean(i-w, i); aft = mean(i, i+w)
        F += [aft-bef, np.linalg.norm(aft-bef, axis=1, keepdims=True)]
    hi = lvl[0][:, [0, 1]].mean(1, keepdims=True); ch = np.cumsum(np.vstack([np.zeros((1, 1)), hi]), 0)
    for w in (4, 8, 16):
        F += [(ch[np.clip(i+w, 0, nb)] - ch[i]) / w, (ch[i] - ch[np.clip(i-w, 0, nb)]) / w]
    on = X[:, :, :, 1].reshape(nb, 16); F.append(on.mean(1, keepdims=True) / (on.mean() + 1e-6))
    # END features (new): level fall after bar vs level inside previous 16 bars (low/mid band cols 0..5 of Z)
    lowZ = Z[:, :6].mean(1, keepdims=True); cl = np.cumsum(np.vstack([np.zeros((1, 1)), lowZ]), 0)
    def m1(a_, b_):
        a_ = np.clip(a_, 0, nb); b_ = np.clip(b_, 0, nb); n_ = np.maximum(b_ - a_, 1)[:, None]
        return (cl[b_] - cl[a_]) / n_
    for w in (4, 8, 16):
        F += [m1(i, i+w) - m1(i-16, i), m1(i, i+w) - m1(i-w, i)]
    return np.concatenate(F, 1).astype(np.float32)


def phase_feats(t):
    BE = t['BE']; nb = len(BE) // 4
    X = BE[:nb*4].reshape(nb, 4, 4, 3); lm = np.log1p(X.mean(1).reshape(nb, 12))
    Z = (lm - np.median(lm, 0)) / (np.percentile(lm, 90, 0) - np.percentile(lm, 10, 0) + 1e-3)
    c = np.cumsum(np.vstack([np.zeros((1, 12)), Z]), 0)
    def mean(a_, b_):
        a_ = np.clip(a_, 0, nb); b_ = np.clip(b_, 0, nb); return (c[b_] - c[a_]) / np.maximum(b_ - a_, 1)[:, None]
    i = np.arange(nb + 1); out_ = []
    for w in (4, 8):
        nov = np.linalg.norm(mean(i, i + w) - mean(i - w, i), axis=1); nov[:2] = 0
        tot = nov.sum() + 1e-6
        for m_ in (8, 4):
            sh = np.array([nov[p::m_].sum() for p in range(m_)]) / tot
            ph = i % m_; out_ += [sh[ph], sh[ph] - sh.max(), (sh[ph] >= sh.max() - 1e-9).astype(float), (sh[None, :] > sh[ph][:, None]).sum(1)]
        loc = np.stack([nov[np.clip(i + d, 0, nb)] for d in range(-3, 4)], 1)
        out_ += [nov / (loc.max(1) + 1e-6), loc.argmax(1) - 3.0, nov / (nov.max() + 1e-6)]
    lowhi = (Z[:, :6].mean(1) > 0.0); fb = int(np.argmax(lowhi)) if lowhi.any() else 0
    d = i - fb; out_ += [d % 8 == 0, d % 4 == 0, d % 16 == 0, np.full(nb + 1, fb % 8), np.full(nb + 1, fb)]
    return np.stack(out_, 1).astype(np.float32)

# ---- tracks + labels
tr = []
for t in out:
    D, e1, H, e2 = drop_ends(t)
    if e1 is None or (H is not None and e2 is None): continue
    L = labels(t, D, H)
    if L['s1'] != 'ok' or (H is not None and L['s2'] not in ('ok', 'unk2')): continue   # ambiguous / no-end tracks dropped
    if t.get('BX') is None: continue          # needs bandx features
    tr.append((t, D, L['e1'], H, L['e2']))
if a.maxtracks: tr = tr[:a.maxtracks]
print(len(tr), 'tracks', flush=True)

# ---- features (every bar candidate)
rows = []; nrow = 0
for t, D, e1, H, e2 in tr:
    nb = len(t['BE']) // 4; nrow += nb + 1
nF = None
if os.path.exists('c7_feat.npy') and os.path.exists('c7_meta.pkl') and not a.maxtracks:
    meta = pickle.load(open('c7_meta.pkl', 'rb')); nF = np.load('c7_feat.npy', mmap_mode='r'); print('feature cache', nF.shape)
if nF is None:
    meta = []; chunks = []
    ncol = None; Fa = None; p = 0; t0 = time.time()
    for k, (t, D, e1, H, e2) in enumerate(tr):
        x, nb = barfeat(t); xn = norm_feats(t); x = np.concatenate([x[:, COLS], xn], 1)
        cands = np.arange(0, nb + 1)
        F = np.concatenate([x[np.clip(cands + o, 0, nb - 1)] for o in OFF], 1)
        gf = np.array([g_ in t['pl'] for g_ in GEN], np.float32)
        F = np.concatenate([F, beat_feats(t), band_feats(t), ad_feats(t), np.tile(gf, (len(cands), 1)), phase_feats(t), np.stack([cands % 4 == 0, cands % 8 == 0, cands % 16 == 0, nb - cands, cands, cands / nb, np.full(len(cands), t['bpm'])], 1)], 1).astype(np.float32)
        if Fa is None:
            ncol = F.shape[1]; Fa = np.lib.format.open_memmap('c7_feat.npy', mode='w+', dtype=np.float32, shape=(nrow, ncol))
        Fa[p:p+len(F)] = F; p += len(F)
        meta.append((cands, nb, (D, e1, H, e2), t['id'], t['name'], p - len(F)))
        if k % 200 == 0: print('feat', k, '%.0fs' % (time.time() - t0), flush=True)
    Fa.flush(); nF = Fa[:p] if p != nrow else Fa
    if not a.maxtracks: pickle.dump(meta, open('c7_meta.pkl', 'wb'))
Fa = nF; N = len(Fa); print(Fa.shape, flush=True)

# ---- targets (start: D,H ; end: e1,e2), every bar
ys = np.zeros(N, np.int8); ye = np.zeros(N, np.int8); g = np.zeros(N, np.int32)
for i, m in enumerate(meta):
    cands, nb, (D, e1, H, e2), _, _, o = m
    g[o:o+len(cands)] = i
    for s, y in ((D, ys), (H, ys), (e1, ye), (e2, ye)):
        if s is None: continue
        k = int(round(s))
        if abs(s - k) < 0.13 and 0 <= k < len(cands): y[o + k] = 1
rng = np.random.RandomState(0)
keep = (ys == 1) | (ye == 1) | (rng.rand(N) < a.neg)       # training subset
def mk(): return HG(max_iter=a.iter, learning_rate=0.08, max_leaf_nodes=16, l2_regularization=2.0, min_samples_leaf=30)
os_ = np.zeros(N, np.float32); oe = np.zeros(N, np.float32)
t0 = time.time()
for f, (tri, tei) in enumerate(GroupKFold(a.folds).split(np.arange(N), groups=g)):
    tk = tri[keep[tri]]; Xtr = np.asarray(Fa[tk])
    for y, o in ((ys, os_), (ye, oe)):
        mdl = mk().fit(Xtr, y[tk])
        for s in range(0, len(tei), 20000): o[tei[s:s+20000]] = mdl.predict_proba(np.asarray(Fa[tei[s:s+20000]]))[:, 1]
    del Xtr
    print('fold', f, '%.0fs' % (time.time() - t0), flush=True)
np.save('c7_os.npy', os_); np.save('c7_oe.npy', oe); print('OOF saved', flush=True)
if a.final:
    tk = np.where(keep)[0]; Xtr = np.asarray(Fa[tk])
    pickle.dump(mk().fit(Xtr, ys[tk]), open('c7_model_start.pkl', 'wb'))
    pickle.dump(mk().fit(Xtr, ye[tk]), open('c7_model_end.pkl', 'wb')); print('final models saved')
print('DONE')
