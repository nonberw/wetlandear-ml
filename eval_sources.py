"""Per-source comparison (see _make_eval_sources.py)."""
import csv, datetime, glob, json, os, sys, warnings
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
APP = r'C:\Users\Timur\Documents\wetlandear-app'
EXPORT = '--export' in sys.argv
CREG = float(os.environ.get('CREG', '2.0'))
W = float(os.environ.get('FW', '0.15'))
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
dirs = sorted(glob.glob(BASE + r'\train_audio\*'))
cls_all = [SYN.get(os.path.basename(d).replace('_', ' '), os.path.basename(d).replace('_', ' ')) for d in dirs]
col = {c: i for i, c in enumerate(cls_all)}
labels = {}
for l in open(BASE + r'\x\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels.txt', encoding='utf-8'):
    s, _, c = l.strip().partition('_'); labels[s] = c
xc = json.load(open(BASE + r'\xc_names.json', encoding='utf-8'))
try:
    js = open(APP + r'\js\species-ru.js', encoding='utf-8').read(); ru = json.loads(js[js.index('{'):js.rindex('}') + 1])
except Exception:
    ru = {}

XB, Y, G, LB, BS, BC = [], [], [], [], [], []
XN, YN, GN, LN, TN, SN = [], [], [], [], [], []
for d in dirs:
    n = os.path.basename(d); f2 = BASE + '\\feats2\\' + n + '.npz'; f3 = BASE + '\\feats3\\' + n + '.npy'; fn = BASE + '\\feats_noisy\\' + n + '.npz'
    if not (os.path.exists(f2) and os.path.exists(f3)): continue
    z = np.load(f2, allow_pickle=True); l = np.load(f3).astype('float32')
    if len(z['Y']) != len(l) or len(l) == 0: continue
    XB.append(z['XB']); Y.append(z['Y']); G.append(z['G']); LB.append(l); BS.append(z['BS']); BC.append(z['BC'])
    if os.path.exists(fn):
        q = np.load(fn, allow_pickle=True)
        if len(q['Y']): XN.append(q['XB']); YN.append(q['Y']); GN.append(q['G']); LN.append(q['LB'].astype('float32')); TN.append(q['T']); SN.append(q['S'])
XB, Y, G, LB, BS, BC = [np.concatenate(a) for a in (XB, Y, G, LB, BS, BC)]
XN, YN, GN, LN, TN, SN = [np.concatenate(a) for a in (XN, YN, GN, LN, TN, SN)]
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y) if sp != 'background'}
use = sorted(sp for sp, n in rec_per.items() if n >= 2)
m = np.isin(Y, use + ['background']); XB, Y, G, LB, BS, BC = XB[m], Y[m], G[m], LB[m], BS[m], BC[m]
m = np.isin(YN, use); XN, YN, GN, LN, TN, SN = XN[m], YN[m], GN[m], LN[m], TN[m], SN[m]
nc, nn = len(Y), len(YN)
XA = np.concatenate([XB, XN]); YA = np.concatenate([Y, YN]); GA = np.concatenate([G, GN]); LA = np.concatenate([LB, LN])
isn = np.r_[np.zeros(nc, bool), np.ones(nn, bool)]
sel = YA != 'background'
print('species %d | clean windows %d | noisy windows %d | recordings %d' % (len(use), nc, nn, len(set(GA))), flush=True)

classes = np.array(sorted(set(YA)))
in_bn = np.array([c in labels for c in classes]); cidx = np.array([col.get(c, -1) for c in classes])
raw = np.zeros((len(YA), len(classes)), dtype='float32'); ok = cidx >= 0; raw[:, ok] = LA[:, cidx[ok]]
raw[:, ~in_bn] = 0.0; raw[:, classes == 'background'] = 0.0
yi = np.array([list(classes).index(y) for y in YA])
folds = list(GroupKFold(n_splits=5).split(XA, YA, GA))

import hashlib
gid_src = {}
for d in dirs:
    for f in glob.glob(d + r'\*'):
        b = os.path.basename(f); gid_src[int(hashlib.md5(f.encode()).hexdigest()[:8], 16)] = 'inat' if b.startswith('INAT') else 'xc' if b.startswith('XC') else 'other'
SRC = np.array([gid_src.get(int(g), 'other') for g in GA])
print('recordings by source:', {s: len(set(GA[SRC == s])) for s in ('xc', 'inat', 'other')}, flush=True)


def fit_pred2(tr, te, only_xc):
    t = tr[SRC[tr] != 'inat'] if only_xc else tr
    mdl = LogisticRegression(C=CREG, max_iter=500, class_weight='balanced').fit(XA[t], YA[t])
    P = np.zeros((len(te), len(classes)), dtype='float32'); P[:, np.searchsorted(classes, mdl.classes_)] = mdl.predict_proba(XA[te])
    return P


jobs = [(k, ox) for k in range(5) for ox in (False, True)]
res = Parallel(n_jobs=int(os.environ.get('NJOBS', '5')), verbose=0)(delayed(fit_pred2)(folds[k][0], folds[k][1], ox) for k, ox in jobs)
PALL = np.zeros((len(YA), len(classes)), dtype='float32'); PXC = PALL.copy()
for (k, ox), P in zip(jobs, res): (PXC if ox else PALL)[folds[k][1]] = P
print('cross-validation done', flush=True)

lg = lambda P: np.log(np.clip(P, 1e-9, 1))
top_all = classes[(lg(PALL) + W * raw).argmax(1)]; top_xc = classes[(lg(PXC) + W * raw).argmax(1)]
bnl = raw.copy(); bnl[:, ~in_bn] = -99; bnl[:, classes == 'background'] = -99
bn_top = np.where(1 / (1 + np.exp(-bnl.max(1))) >= .2, classes[bnl.argmax(1)], 'background')
out = {}
for name, cl in (('clean', ~isn), ('noisy', isn)):
    for s in ('xc', 'inat'):
        m = cl & sel & (SRC == s)
        if m.sum() == 0: continue
        r = {'windows': int(m.sum()), 'recordings': len(set(GA[m])), 'birdnet_restricted': float((bn_top[m] == YA[m]).mean()), 'trained_on_everything': float((top_all[m] == YA[m]).mean()), 'trained_without_inat': float((top_xc[m] == YA[m]).mean())}
        out['%s_%s' % (name, s)] = r
        print('%-12s windows %6d recordings %5d | BirdNET %.1f%% | head trained on everything %.1f%% | head trained without iNaturalist %.1f%%' % (name + '/' + s, r['windows'], r['recordings'], r['birdnet_restricted'] * 100, r['trained_on_everything'] * 100, r['trained_without_inat'] * 100), flush=True)
# per recording (clean windows): average probability of the recording, vote
for s in ('xc', 'inat'):
    ok_a = ok_x = nrec = 0
    for g in set(GA[(SRC == s) & ~isn & sel]):
        m = (GA == g) & ~isn & sel
        ya = YA[m][0]; ok_a += classes[(lg(PALL[m]) + W * raw[m]).mean(0).argmax()] == ya; ok_x += classes[(lg(PXC[m]) + W * raw[m]).mean(0).argmax()] == ya; nrec += 1
    out['per_recording_' + s] = {'recordings': nrec, 'trained_on_everything': ok_a / nrec, 'trained_without_inat': ok_x / nrec}
    print('per recording %-5s n=%d | everything %.1f%% | without iNaturalist %.1f%%' % (s, nrec, ok_a / nrec * 100, ok_x / nrec * 100), flush=True)
json.dump(out, open(BASE + r'\eval_sources.json', 'w', encoding='utf-8'), indent=1)
print('done', flush=True)
