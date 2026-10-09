"""Honest metrics for the public page: cross-validation by recording, per-species table, calibration. Writes metrics.json (copied to the site)."""
import glob, json, os, sys, warnings, datetime
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
OUT = sys.argv[1] if len(sys.argv) > 1 else BASE + r'\metrics.json'
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
dirs = sorted(glob.glob(BASE + r'\train_audio\*'))
cls_all = [SYN.get(os.path.basename(d).replace('_', ' '), os.path.basename(d).replace('_', ' ')) for d in dirs]
labels = {}
for l in open(BASE + r'\x\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels.txt', encoding='utf-8'):
    s, _, c = l.strip().partition('_'); labels[s] = c
xc = json.load(open(BASE + r'\xc_names.json', encoding='utf-8'))
ru = {}
try:
    import re
    js = open(r'C:\Users\Timur\Documents\wetlandear-app\js\species-ru.js', encoding='utf-8').read()
    ru = json.loads(js[js.index('{'):js.rindex('}') + 1])
except Exception:
    pass

XB, Y, G, LB, BS, BC = [], [], [], [], [], []
for d in dirs:
    n = os.path.basename(d); f2 = BASE + '\\feats2\\' + n + '.npz'; f3 = BASE + '\\feats3\\' + n + '.npy'
    if not (os.path.exists(f2) and os.path.exists(f3)): continue
    z = np.load(f2, allow_pickle=True); l = np.load(f3).astype('float32')
    if len(z['Y']) != len(l) or len(l) == 0: continue
    XB.append(z['XB']); Y.append(z['Y']); G.append(z['G']); LB.append(l); BS.append(z['BS']); BC.append(z['BC'])
XB, Y, G, LB, BS, BC = [np.concatenate(a) for a in (XB, Y, G, LB, BS, BC)]
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y) if sp != 'background'}
use = sorted(sp for sp, n in rec_per.items() if n >= 2)
mask = np.isin(Y, use + ['background']); XB, Y, G, LB, BS, BC = XB[mask], Y[mask], G[mask], LB[mask], BS[mask], BC[mask]
sel = Y != 'background'
col = {c: i for i, c in enumerate(cls_all)}
P = None
for tr, te in GroupKFold(n_splits=5).split(XB, Y, G):
    m = LogisticRegression(C=2.0, max_iter=500, class_weight='balanced').fit(XB[tr], Y[tr])
    if P is None: classes = np.array(m.classes_); P = np.zeros((len(Y), len(classes)), dtype='float32')
    P[te] = m.predict_proba(XB[te])
in_bn = np.array([c in labels for c in classes]); cidx = np.array([col.get(c, -1) for c in classes])
raw = np.zeros((len(Y), len(classes)), dtype='float32'); ok = cidx >= 0; raw[:, ok] = LB[:, cidx[ok]]
raw[:, ~in_bn] = 0.0; raw[:, classes == 'background'] = 0
W = 0.15
z = np.log(np.clip(P, 1e-9, 1)) + W * raw
yi = np.array([list(classes).index(y) for y in Y])
def nll(T):
    q = z / T; q = q - q.max(1, keepdims=True); p = np.exp(q); p /= p.sum(1, keepdims=True); return -np.mean(np.log(p[np.arange(len(Y)), yi] + 1e-9))
Ts = np.linspace(.4, 3, 27); T = float(Ts[int(np.argmin([nll(t) for t in Ts]))])
q = z / T; q = q - q.max(1, keepdims=True); Pf = np.exp(q); Pf /= Pf.sum(1, keepdims=True)
top = classes[np.argmax(Pf, 1)]; mx = Pf.max(1)
bn = np.where(BC >= .2, BS, 'background')
win_acc = float((top[sel] == Y[sel]).mean()); bn_acc = float((bn[sel] == Y[sel]).mean())
ok_rec = n = 0; ok_bn_rec = 0
for g in set(G[sel]):
    mm = (G == g) & sel; ok_rec += classes[np.argmax(Pf[mm].mean(0))] == Y[mm][0]; n += 1
cal = []
for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    m = sel & (mx >= lo) & (mx < hi) & (top != 'background')
    if m.sum(): cal.append({'lo': lo, 'hi': min(hi, 1.0), 'windows': int(m.sum()), 'share': float(m.sum() / sel.sum()), 'right': float((top[m] == Y[m]).mean())})
cover = []
for t in (.3, .5, .7, .9):
    m = sel & (mx >= t) & (top != 'background'); cover.append({'t': t, 'covers': float(m.sum() / sel.sum()), 'precision': float((top[m] == Y[m]).mean())})
rows = []
for sp in use:
    m = Y == sp
    if m.sum() == 0: continue
    com = labels.get(sp) or xc.get(sp) or sp
    rows.append({'sci': sp, 'en': com, 'ru': ru.get(com, ''), 'recordings': rec_per[sp], 'windows': int(m.sum()), 'head': float((top[m] == sp).mean()), 'birdnet': float((bn[m] == sp).mean()), 'in_birdnet': sp in labels})
rows.sort(key=lambda r: -r['head'])
out = {'generated': datetime.date.today().isoformat(), 'species': len(use), 'recordings': int(len(set(G))), 'windows': int(sel.sum()), 'temperature': T, 'fusion_weight': W,
       'window_accuracy': {'birdnet': bn_acc, 'wetlandear': win_acc}, 'recording_accuracy': ok_rec / n, 'calibration': cal, 'coverage': cover, 'per_species': rows,
       'method': 'Cross-validation split by recording (5 folds): no recording is ever in both training and test. Test windows are 3-second fragments of Xeno-canto recordings.'}
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('species', len(use), 'recordings', len(set(G)), 'window: birdnet %.1f%% wetlandear %.1f%% | per recording %.1f%%' % (bn_acc * 100, win_acc * 100, ok_rec / n * 100))
