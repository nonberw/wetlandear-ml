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


def head_first(P, cl, bs, bc, th, bth):
    k = np.argmax(P, 1); hc = np.array(cl)[k]; hp = P[np.arange(len(P)), k]
    use = (hc != 'background') & (hp >= th)
    fb = np.where(bc >= bth, bs, 'background')
    return np.where(use, hc, fb)


res = {}
for th in (.3, .4, .5, .6, .7):
    ok = n = ok_bn = 0; wrong_claims = 0
    for tr, te in GroupKFold(n_splits=5).split(X, Y, G):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(X[tr], Y[tr]); P = clf.predict_proba(X[te]); cl = list(clf.classes_)
        sel = Y[te] != 'background'
        pr = head_first(P[sel], cl, BS[te][sel], BC[te][sel], th, .3)
        n += sel.sum(); ok += (pr == Y[te][sel]).sum()
        ok_bn += (np.where(BC[te][sel] >= .2, BS[te][sel], 'background') == Y[te][sel]).sum()
    print('head-first th=%.1f: %.1f%%   (BirdNET alone %.1f%%)' % (th, ok / n * 100, ok_bn / n * 100))

print('\nOPEN-SET with head-first th=0.5: unseen species')
tot = ok = wrong = nothing = 0
for sp in keep:
    if rec_per[sp] < 3: continue
    tr = Y != sp; te = Y == sp
    clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(X[tr], Y[tr]); P = clf.predict_proba(X[te]); cl = list(clf.classes_)
    pr = head_first(P, cl, BS[te], BC[te], .5, .3)
    tot += te.sum(); ok += (pr == sp).sum(); wrong += ((pr != sp) & (pr != 'background')).sum(); nothing += (pr == 'background').sum()
print('unseen-species windows %d: correct %.1f%% | wrong name %.1f%% | no suggestion %.1f%%' % (tot, ok / tot * 100, wrong / tot * 100, nothing / tot * 100))
