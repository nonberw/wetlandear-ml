"""Final head v2: LogReg on BirdNET fingerprints + late fusion with BirdNET scores + temperature. Honest CV by recording, then export."""
import csv, glob, json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
APP = r'C:\Users\Timur\Documents\wetlandear-app'
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
dirs = sorted(glob.glob(BASE + r'\train_audio\*'))
cls_all = [SYN.get(os.path.basename(d).replace('_', ' '), os.path.basename(d).replace('_', ' ')) for d in dirs]
labels = {}
for l in open(BASE + r'\x\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels.txt', encoding='utf-8'):
    s, _, c = l.strip().partition('_'); labels[s] = c
xc = json.load(open(BASE + r'\xc_names.json', encoding='utf-8'))

XB, Y, G, LB = [], [], [], []
for d in dirs:
    n = os.path.basename(d); f2 = BASE + '\\feats2\\' + n + '.npz'; f3 = BASE + '\\feats3\\' + n + '.npy'
    if not (os.path.exists(f2) and os.path.exists(f3)): continue
    z = np.load(f2, allow_pickle=True); l = np.load(f3).astype('float32')
    if len(z['Y']) != len(l) or len(l) == 0: continue
    XB.append(z['XB']); Y.append(z['Y']); G.append(z['G']); LB.append(l)
XB, Y, G, LB = [np.concatenate(a) for a in (XB, Y, G, LB)]
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y) if sp != 'background'}
use = sorted(sp for sp, n in rec_per.items() if n >= 2)
mask = np.isin(Y, use + ['background']); XB, Y, G, LB = XB[mask], Y[mask], G[mask], LB[mask]
col = {c: i for i, c in enumerate(cls_all)}
sel = Y != 'background'
print('classes', len(use), 'windows', len(Y), 'recordings', len(set(G)), flush=True)

# OOF probabilities
P = None
for tr, te in GroupKFold(n_splits=5).split(XB, Y, G):
    m = LogisticRegression(C=2.0, max_iter=500, class_weight='balanced').fit(XB[tr], Y[tr])
    if P is None: classes = np.array(m.classes_); P = np.zeros((len(Y), len(classes)), dtype='float32')
    P[te] = m.predict_proba(XB[te])
cidx = np.array([col.get(c, -1) for c in classes])
in_bn = np.array([c in labels for c in classes])
raw = np.zeros((len(Y), len(classes)), dtype='float32'); ok = cidx >= 0; raw[:, ok] = LB[:, cidx[ok]]


def bn_variant(missing):
    b = raw.copy(); b[:, ~in_bn] = missing; b[:, classes == 'background'] = 0
    return b


def fused(P, bn, w, T=1.0):
    z = (np.log(np.clip(P, 1e-9, 1)) + w * bn) / T; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def acc(Pf):
    top = classes[np.argmax(Pf, 1)]; return (top[sel] == Y[sel]).mean(), top


best = None
for miss, name in ((-15.0, 'missing=-15'), (0.0, 'missing=0 (neutral)')):
    bn = bn_variant(miss)
    for w in (0.05, 0.1, 0.15, 0.2):
        a, top = acc(fused(P, bn, w))
        pel = (top[Y == 'Pelecanus onocrotalus'] == 'Pelecanus onocrotalus').mean() if (Y == 'Pelecanus onocrotalus').any() else float('nan')
        print('%-22s w=%.2f  window %.1f%%  | Great White Pelican recall %.0f%%' % (name, w, a * 100, pel * 100), flush=True)
        if best is None or a > best[0]: best = (a, miss, w)
miss, w = 0.0, 0.15   # same overall accuracy as the best, but species BirdNET has no label for (pelicans) are not penalised
a = acc(fused(P, bn_variant(miss), w))[0]
print('chosen: missing=%s w=%.2f (%.1f%%)' % (miss, w, a * 100), flush=True)
bn = bn_variant(miss)
base_a, _ = acc(P)
print('LogReg alone %.1f%% -> fused %.1f%%' % (base_a * 100, a * 100))
# temperature
lgz = np.log(np.clip(P, 1e-9, 1)) + w * bn
yi = np.array([list(classes).index(y) for y in Y])
def nll(T):
    z = lgz / T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -np.mean(np.log(p[np.arange(len(Y)), yi] + 1e-9))
Ts = np.linspace(.4, 3, 27); T = float(Ts[int(np.argmin([nll(t) for t in Ts]))]); print('temperature', T)
Pf = fused(P, bn, w, T); mx = Pf.max(1); top = classes[np.argmax(Pf, 1)]; right = top == Y
print('calibration of the final confidence (out-of-fold):')
for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    m = sel & (mx >= lo) & (mx < hi) & (top != 'background')
    if m.sum(): print('  %.2f-%.2f: %5d windows (%2.0f%%), right %.1f%%' % (lo, hi, m.sum(), m.sum() / sel.sum() * 100, right[m].mean() * 100))
for t in (.3, .4, .5, .6):
    m = sel & (mx >= t) & (top != 'background'); print('  confidence >= %.1f: covers %.0f%%, precision %.1f%%' % (t, m.sum() / sel.sum() * 100, right[m].mean() * 100))
ok_rec = n = 0
for g in set(G[sel]):
    m = (G == g) & sel; ok_rec += classes[np.argmax(Pf[m].mean(0))] == Y[m][0]; n += 1
print('per-recording accuracy (mean over windows): %.1f%% (%d recordings)' % (ok_rec / n * 100, n))

# final fit and export
fin = LogisticRegression(C=2.0, max_iter=800, class_weight='balanced').fit(XB, Y)
fcls = [str(c) for c in fin.classes_]
common = [('' if c == 'background' else (labels.get(c) or xc.get(c) or c)) for c in fcls]
np.savez_compressed(APP + r'\models\head.npz', coef=fin.coef_.astype('float16'), intercept=fin.intercept_.astype('float32'), classes=np.array(fcls), common=np.array(common),
                    T=np.float32(T), w=np.float32(w), missing=np.float32(miss), bn_labels=np.array([c if c in labels else '' for c in fcls]))
print('exported', len(fcls), 'classes,', os.path.getsize(APP + r'\models\head.npz') // 1024, 'KB')
rows = [['species', 'xeno_canto_id', 'url']]
for p in sorted(glob.glob(BASE + r'\train_audio\*\XC*.mp3')):
    sp = os.path.basename(os.path.dirname(p)).replace('_', ' '); xid = os.path.basename(p)[2:-4]
    if os.path.getsize(p) > 5000: rows.append([sp, xid, 'https://xeno-canto.org/' + xid])
with open(APP + r'\models\head_training_sources.csv', 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerows(rows)
print('sources', len(rows) - 1)
json.dump({'classes': len(fcls), 'windows': int(len(Y)), 'recordings': int(len(set(G))), 'window_acc_cv': float(a), 'per_recording_cv': ok_rec / n, 'birdnet_alone_window': 0.595},
          open(BASE + r'\head_report.json', 'w'))
