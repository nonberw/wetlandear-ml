"""Compare feature sets and classifiers with cross-validation BY RECORDING, then calibrate. Prints honest numbers."""
import glob, sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
XB, XY, Y, G, BS, BC = [], [], [], [], [], []
for f in sorted(glob.glob(BASE + r'\feats2\*.npz')):
    z = np.load(f, allow_pickle=True)
    if len(z['Y']) == 0: continue
    XB.append(z['XB']); XY.append(z['XY']); Y.append(z['Y']); G.append(z['G']); BS.append(z['BS']); BC.append(z['BC'])
XB, XY, Y, G, BS, BC = [np.concatenate(a) for a in (XB, XY, Y, G, BS, BC)]
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y) if sp != 'background'}
keep = sorted(sp for sp, n in rec_per.items() if n >= 4)
mask = np.isin(Y, keep + ['background'])
XB, XY, Y, G, BS, BC = XB[mask], XY[mask], Y[mask], G[mask], BS[mask], BC[mask]
print('species with >=4 recordings:', len(keep), '| windows', len(Y), '| recordings', len(set(G)), flush=True)
XBY = np.hstack([XB, XY])

sets = {'BirdNET fingerprint': XB, 'BirdNET + YAMNet': XBY}
models = {
    'LogReg C=2': lambda: LogisticRegression(C=2.0, max_iter=400, class_weight='balanced'),
    'MLP 512': lambda: MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3, max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=0),
}


def cv(Xs, make, folds=5):
    P = np.zeros((len(Y), len(keep) + 1), dtype='float32'); classes = None
    for tr, te in GroupKFold(n_splits=folds).split(Xs, Y, G):
        m = make().fit(Xs[tr], Y[tr])
        if classes is None: classes = list(m.classes_)
        pp = np.zeros((len(te), len(classes)), dtype='float32')
        idx = [classes.index(c) for c in m.classes_]
        pp[:, idx] = m.predict_proba(Xs[te])
        P[te] = pp
    return P, np.array(classes)


def metrics(P, classes, name):
    sel = Y != 'background'
    top = classes[np.argmax(P, 1)]
    acc = (top[sel] == Y[sel]).mean()
    t3 = np.mean([Y[i] in classes[np.argsort(-P[i])[:3]] for i in np.where(sel)[0]])
    # recording level: mean probability over a recording's windows
    ok = n = 0
    for g in set(G[sel]):
        m = (G == g) & sel
        ok += classes[np.argmax(P[m].mean(0))] == Y[m][0]; n += 1
    print('%-34s window top1 %.1f%% | top3 %.1f%% | per-recording %.1f%% (%d rec)' % (name, acc * 100, t3 * 100, ok / n * 100, n), flush=True)
    return acc


sel = Y != 'background'
bn = np.where(BC >= .2, BS, 'background')
print('BirdNET alone (species it can name, same windows): window top1 %.1f%%' % ((bn[sel] == Y[sel]).mean() * 100), flush=True)
res = {}
for sname, Xs in sets.items():
    for mname, make in models.items():
        t = time.time(); P, classes = cv(Xs, make)
        res[(sname, mname)] = (metrics(P, classes, sname + ' / ' + mname), P, classes)
        print('   (%.0f s)' % (time.time() - t), flush=True)
best = max(res, key=lambda k: res[k][0]); acc, P, classes = res[best]
print('\nBEST:', best, '%.1f%%' % (acc * 100))

# calibration: temperature scaling on out-of-fold probabilities
lg = np.log(np.clip(P, 1e-9, 1)); yi = np.array([list(classes).index(y) for y in Y])
def nll(T):
    z = lg / T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -np.mean(np.log(p[np.arange(len(Y)), yi] + 1e-9))
Ts = np.linspace(.5, 3, 26); T = Ts[int(np.argmin([nll(t) for t in Ts]))]
z = lg / T; z = z - z.max(1, keepdims=True); Pc = np.exp(z); Pc /= Pc.sum(1, keepdims=True)
print('temperature %.2f' % T)
mx = Pc.max(1); top = classes[np.argmax(Pc, 1)]; right = top == Y
for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    m = sel & (mx >= lo) & (mx < hi) & (top != 'background')
    if m.sum(): print('  confidence %.2f-%.2f: %5d windows (%.0f%%), right %.1f%%' % (lo, hi, m.sum(), m.sum() / sel.sum() * 100, right[m].mean() * 100))
for t in (.3, .4, .5, .6):
    m = sel & (mx >= t) & (top != 'background')
    print('  at confidence >= %.1f: covers %.0f%%, precision %.1f%%' % (t, m.sum() / sel.sum() * 100, right[m].mean() * 100))
# per species recall at 0.4
print('\nweakest species (recall of top1 across windows):')
rec = {sp: (top[Y == sp] == sp).mean() for sp in keep}
for sp in sorted(rec, key=rec.get)[:12]: print('  %-30s %.0f%%  (%d recordings)' % (sp, rec[sp] * 100, rec_per[sp]))
np.savez(BASE + r'\best_cfg.npz', set=best[0], model=best[1], T=T)
