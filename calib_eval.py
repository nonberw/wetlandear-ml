import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
d = np.load(r'C:\Users\Timur\Documents\wetlandear-models\feats.npz', allow_pickle=True)
X, Y, G, BS, BC = d['X'], d['Y'], d['G'], d['BS'], d['BC']
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y)}
keep = [sp for sp, n in rec_per.items() if n >= 2 and sp != 'background']
mask = np.isin(Y, keep + ['background']); X, Y, G, BS, BC = X[mask], Y[mask], G[mask], BS[mask], BC[mask]
C = 2.0
rows = []   # (maxprob, correct, bn_correct)
for tr, te in GroupKFold(n_splits=5).split(X, Y, G):
    clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(X[tr], Y[tr]); P = clf.predict_proba(X[te]); cl = np.array(clf.classes_)
    for p, y, b, c in zip(P, Y[te], BS[te], BC[te]):
        if y == 'background': continue
        k = int(np.argmax(p)); rows.append((p[k], cl[k] == y, (b == y) and c >= .2, cl[k] == 'background'))
r = np.array([(a, b, c) for a, b, c, _ in rows], dtype=float)
print('closed set (species the head knows), by head confidence')
bins = [0, .15, .25, .35, .5, .7, 1.01]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (r[:, 0] >= lo) & (r[:, 0] < hi)
    if m.sum(): print('  p %.2f-%.2f: %4d windows (%.0f%%)  head right %.1f%%  | BirdNET right on the same windows %.1f%%' % (lo, hi, m.sum(), m.mean() * 100, r[m, 1].mean() * 100, r[m, 2].mean() * 100))
print('\nopen set (head never saw the species): share of windows where head claims a species at confidence >= t')
opn = []
for sp in keep:
    if rec_per[sp] < 3: continue
    tr = Y != sp; te = Y == sp
    clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(X[tr], Y[tr]); P = clf.predict_proba(X[te]); cl = np.array(clf.classes_)
    k = np.argmax(P, 1); hp = P[np.arange(len(P)), k]; hc = cl[k]
    opn.append((hp, hc != 'background'))
hp = np.concatenate([a for a, _ in opn]); nb = np.concatenate([b for _, b in opn])
for t in (.15, .25, .35, .5):
    print('  t=%.2f: wrong claims %.1f%%' % (t, ((hp >= t) & nb).mean() * 100))
for t in (.15, .25, .35, .5):
    m = r[:, 0] >= t
    print('  closed-set at t=%.2f: covers %.0f%% of windows, precision %.1f%%' % (t, m.mean() * 100, r[m, 1].mean() * 100))
