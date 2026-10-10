"""Few-shot voices: a tiny logistic classifier "this voice vs everything else" trained on your samples plus a shipped set of negative fingerprints.
Fair test: shipped negatives come from one half of the species, test negatives from the OTHER half (species the app has never seen)."""
import glob, sys
import numpy as np
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
rng = np.random.default_rng(5)
data, allx = {}, {}
for f in glob.glob(BASE + r'\feats2\*.npz'):
    z = np.load(f, allow_pickle=True); Y, G, X = z['Y'], z['G'], z['XB']; m = Y != 'background'
    if m.sum() == 0: continue
    sp = str(Y[m][0]); data[sp] = {g: X[(G == g) & m] for g in set(G[m])}; allx[sp] = X[m]
species = sorted(data); rng.shuffle(species)
half = len(species) // 2; A_sp, B_sp = species[:half], species[half:]
def pick(sps, per):
    return np.concatenate([allx[s][rng.choice(len(allx[s]), min(per, len(allx[s])), replace=False)] for s in sps])
NEG_TEST_BASE = {s: allx[s][rng.choice(len(allx[s]), min(8, len(allx[s])), replace=False)] for s in B_sp}


def train_lr(P, N, iters=300, lr=2.0, l2=1e-3):
    X = np.concatenate([P, N]).astype(np.float32); y = np.r_[np.ones(len(P)), np.zeros(len(N))].astype(np.float32)
    sw = (np.where(y == 1, 0.5 / len(P), 0.5 / len(N)) * len(y)).astype(np.float32)
    w = np.zeros(X.shape[1], np.float32); b = np.float32(0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = (p - y) * sw
        w -= lr * (X.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return w, b


targets = list(rng.choice(species, 36, replace=False))
for n_neg in (400, 900):
    per = max(1, n_neg // len(A_sp)); NEG = pick(A_sp, per); print('shipped negatives:', len(NEG), flush=True)
    for n_rec in (3, 5, 10):
        hits = {t: [0, 0] for t in (.3, .5, .7)}; n_tp = n_fp = 0
        for sp in targets:
            recs = data[sp]; keys = list(recs); rng.shuffle(keys)
            if len(keys) <= n_rec + 2: continue
            train = np.concatenate([recs[k][:6] for k in keys[:n_rec]]); pos = np.concatenate([recs[k] for k in keys[n_rec:]])
            w, b = train_lr(train, NEG)
            neg_test = np.concatenate([v for s, v in NEG_TEST_BASE.items() if s != sp][:40])
            sp_ = 1 / (1 + np.exp(-(pos @ w + b))); sn_ = 1 / (1 + np.exp(-(neg_test @ w + b)))
            for t in hits: hits[t][0] += (sp_ >= t).sum(); hits[t][1] += (sn_ >= t).sum()
            n_tp += len(pos); n_fp += len(neg_test)
        print('  taught with %2d recordings:' % n_rec, '   '.join('p>=%.1f found %4.1f%% / false %4.1f%%' % (t, h[0] / n_tp * 100, h[1] / n_fp * 100) for t, h in hits.items()), flush=True)
