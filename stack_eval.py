import glob, os, sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
dirs = sorted(glob.glob(BASE + r'\train_audio\*'))
cls_all = [SYN.get(os.path.basename(d).replace('_', ' '), os.path.basename(d).replace('_', ' ')) for d in dirs]
XB, Y, G, LB, BS, BC = [], [], [], [], [], []
for d, c in zip(dirs, cls_all):
    n = os.path.basename(d); f2 = BASE + '\\feats2\\' + n + '.npz'; f3 = BASE + '\\feats3\\' + n + '.npy'
    if not (os.path.exists(f2) and os.path.exists(f3)): continue
    z = np.load(f2, allow_pickle=True); l = np.load(f3).astype('float32')
    if len(z['Y']) != len(l) or len(l) == 0: continue
    XB.append(z['XB']); Y.append(z['Y']); G.append(z['G']); LB.append(l); BS.append(z['BS']); BC.append(z['BC'])
XB, Y, G, LB, BS, BC = [np.concatenate(a) for a in (XB, Y, G, LB, BS, BC)]
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y) if sp != 'background'}
keep = sorted(sp for sp, n in rec_per.items() if n >= 4)
mask = np.isin(Y, keep + ['background']); XB, Y, G, LB, BS, BC = XB[mask], Y[mask], G[mask], LB[mask], BS[mask], BC[mask]
col = {c: i for i, c in enumerate(cls_all)}
print('windows', len(Y), 'species', len(keep), flush=True)
sel = Y != 'background'


def run(Xs, C):
    P = None; classes = None
    for tr, te in GroupKFold(n_splits=5).split(Xs, Y, G):
        m = LogisticRegression(C=C, max_iter=400, class_weight='balanced').fit(Xs[tr], Y[tr])
        if classes is None:
            classes = np.array(m.classes_); P = np.zeros((len(Y), len(classes)), dtype='float32')
        P[te] = m.predict_proba(Xs[te])
    return P, classes


def rep(name, P, classes):
    top = classes[np.argmax(P, 1)]; acc = (top[sel] == Y[sel]).mean()
    ok = n = 0
    for g in set(G[sel]):
        m = (G == g) & sel; ok += classes[np.argmax(P[m].mean(0))] == Y[m][0]; n += 1
    print('%-40s window %.1f%% | per-recording %.1f%%' % (name, acc * 100, ok / n * 100), flush=True)
    return acc


best = None
for C in (1.0, 2.0, 5.0):
    t = time.time(); P, classes = run(XB, C); a = rep('fingerprint, LogReg C=%.0f' % C, P, classes)
    if C == 2.0: P2, classes2 = P, classes
    if best is None or a > best[0]: best = (a, C, P, classes)
print('best C', best[1], flush=True)
P, classes = best[2], best[3]
# late fusion with BirdNET's own scores for our species
lbcols = np.array([col.get(c, -1) for c in classes])
bn = np.zeros((len(Y), len(classes)), dtype='float32'); ok = lbcols >= 0; bn[:, ok] = LB[:, lbcols[ok]]
for w in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75):
    z = np.log(np.clip(P, 1e-9, 1)) + w * bn; z -= z.max(1, keepdims=True); Pf = np.exp(z); Pf /= Pf.sum(1, keepdims=True)
    rep('late fusion w=%.2f' % w, Pf, classes)
# stacked features: fingerprint + sigmoid(BirdNET scores)
S = 1 / (1 + np.exp(-LB))
for scale in (0.5, 1.0):
    Xs = np.hstack([XB, S * scale]); P3, cl3 = run(Xs, 2.0); rep('stacked [fingerprint, BirdNET scores x%.1f]' % scale, P3, cl3)
