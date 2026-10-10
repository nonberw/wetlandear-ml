"""How well can the app recognise a voice it was taught from a few recordings? Simulation on Xeno-canto fingerprints:
teach 5 recordings (6 strongest fragments each), test on the other recordings of that species (should be found) and on other species (should not).
Compares the current rule (nearest sample, similarity >= 0.72) with rules that adapt to the species."""
import glob, os, sys
import numpy as np
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
rng = np.random.default_rng(3)
data = {}
for f in glob.glob(BASE + r'\feats2\*.npz'):
    z = np.load(f, allow_pickle=True)
    Y, G, X = z['Y'], z['G'], z['XB']
    m = Y != 'background'
    if m.sum() == 0: continue
    sp = str(Y[m][0]); recs = {}
    for g in set(G[m]): recs[g] = X[(G == g) & m]
    if len(recs) >= 10: data[sp] = recs
species = sorted(data)
print(len(species), 'species with >= 10 recordings')
all_other = {sp: np.concatenate(list(data[sp].values())) for sp in species}


def loo_sims(S):
    """Each sample against the others of the same voice: how similar do two fragments of this voice usually look?"""
    M = S @ S.T; np.fill_diagonal(M, -1); return M.max(1)


def rules(S, q):
    out = {}
    out['A nearest>=0.72 (now)'] = (q @ S.T).max(1) >= .72
    s1 = (q @ S.T).max(1)
    ls = loo_sims(S)
    out['B nearest>=LOO q20'] = s1 >= np.clip(np.quantile(ls, .20), .55, .9)
    c = S.mean(0); c /= np.linalg.norm(c)
    cs = S @ c; thr_c = np.quantile(cs, .05)
    out['C centroid>=q05 of samples'] = (q @ c) >= max(.5, thr_c - .05)
    # D: mean of the 3 nearest samples (less sensitive to one odd sample), threshold from LOO of the same statistic
    k = min(3, len(S) - 1)
    def knn3(Q, excl=False):
        M = Q @ S.T
        if excl: np.fill_diagonal(M, -1)
        return np.sort(M, 1)[:, -k:].mean(1)
    thr3 = np.quantile(knn3(S, True), .20)
    out['D mean of 3 nearest>=LOO q20'] = knn3(q) >= max(.5, thr3)
    out['E D and margin 0.0'] = out['D mean of 3 nearest>=LOO q20']
    return out


names = None; tp = {}; fp = {}; n_tp = n_fp = 0
for sp in rng.choice(species, 60, replace=False):
    recs = data[sp]; keys = list(recs); rng.shuffle(keys)
    train = np.concatenate([recs[k][:6] for k in keys[:5]])
    pos = np.concatenate([recs[k] for k in keys[5:]])
    others = [o for o in species if o != sp]
    neg = np.concatenate([all_other[o][rng.choice(len(all_other[o]), 6, replace=False)] for o in rng.choice(others, 60, replace=False)])
    rp, rn = rules(train, pos), rules(train, neg)
    for k in rp:
        tp[k] = tp.get(k, 0) + rp[k].sum(); fp[k] = fp.get(k, 0) + rn[k].sum()
    n_tp += len(pos); n_fp += len(neg)
print('taught with 5 recordings; found = fraction of fragments of other recordings of the same species, false = fraction of other species wrongly matched')
for k in tp: print('%-34s found %5.1f%%   false alarms %5.1f%%' % (k, tp[k] / n_tp * 100, fp[k] / n_fp * 100))
