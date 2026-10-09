"""Honest evaluation of head + BirdNET fusion on cached features. All splits are by recording."""
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

d = np.load(r'C:\Users\Timur\Documents\wetlandear-models\feats.npz', allow_pickle=True)
X, Y, G, BS, BC = d['X'], d['Y'], d['G'], d['BS'], d['BC']
SYN = {'Anthropoides virgo': 'Anthropoides virgo'}
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y)}
keep = [sp for sp, n in rec_per.items() if n >= 2 and sp != 'background']
mask = np.isin(Y, keep + ['background'])
X, Y, G, BS, BC = X[mask], Y[mask], G[mask], BS[mask], BC[mask]
print('species', len(keep), 'windows', len(X), 'recordings', len(set(G)))


def head_fit(Xtr, Ytr, C):
    return LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(Xtr, Ytr)


def fuse(proba, classes, bs, bc, th, bn_th):
    """Return predicted species per window."""
    out = []
    cset = set(classes)
    for p, b, c in zip(proba, bs, bc):
        k = int(np.argmax(p)); hc = classes[k]; hp = p[k]
        bn_ok = c >= .2
        if bn_ok and c >= bn_th and b not in cset:
            out.append(b)                      # BirdNET is sure about a species the head does not know
        elif hc != 'background' and hp >= th:
            out.append(hc)
        else:
            out.append(b if bn_ok else 'background')
    return np.array(out)


for C in (2.0, 8.0, 20.0):
    folds = GroupKFold(n_splits=5)
    stats = {'bn': 0, 'head': 0, 'fused': 0, 'n': 0}
    for tr, te in folds.split(X, Y, G):
        clf = head_fit(X[tr], Y[tr], C); P = clf.predict_proba(X[te]); cl = list(clf.classes_)
        sel = Y[te] != 'background'
        yt = Y[te][sel]
        bn = np.where(BC[te] >= .2, BS[te], 'background')[sel]
        hd = np.array(cl)[np.argmax(P, 1)][sel]
        fz = fuse(P[sel], cl, BS[te][sel], BC[te][sel], .5, .6)
        stats['n'] += sel.sum(); stats['bn'] += (bn == yt).sum(); stats['head'] += (hd == yt).sum(); stats['fused'] += (fz == yt).sum()
    n = stats['n']
    print('C=%4.1f  BirdNET %.1f%%  head %.1f%%  fused %.1f%%  (n=%d)' % (C, stats['bn'] / n * 100, stats['head'] / n * 100, stats['fused'] / n * 100, n))

# open-set: how often does the head wrongly claim a species it does not know? train without one species, test on it
print('\nOPEN-SET (species the head was NOT trained on). Lower false-claim rate is better; BirdNET is the reference.')
tot = bad_head = bad_fused = ok_bn = 0
for sp in keep:
    if rec_per[sp] < 3: continue
    tr = Y != sp; te = Y == sp
    clf = head_fit(X[tr], Y[tr], 8.0); P = clf.predict_proba(X[te]); cl = list(clf.classes_)
    hp = P.max(1); hc = np.array(cl)[np.argmax(P, 1)]
    claim = (hc != 'background') & (hp >= .5)
    fz = fuse(P, cl, BS[te], BC[te], .5, .6)
    tot += te.sum(); bad_head += claim.sum(); bad_fused += ((fz != 'background') & (fz != sp) & (np.isin(fz, cl))).sum(); ok_bn += ((BS[te] == sp) & (BC[te] >= .2)).sum()
print('windows of unknown species: %d | head claims a wrong known species at p>=0.5: %.1f%% | fused returns a wrong head-class: %.1f%% | BirdNET names it correctly: %.1f%%' % (tot, bad_head / tot * 100, bad_fused / tot * 100, ok_bn / tot * 100))
