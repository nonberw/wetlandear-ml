"""v4: clean + noisy training, honest cross-validation by recording, noise-robustness report, metrics.json, optional export.
python train_v4.py [--export]
Compares  A) head trained on clean recordings only  vs  B) head trained on clean + noisy copies,  both tested on clean AND noisy windows of held-out recordings."""
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


def fit_pred(tr, te, train_noisy):
    t = tr if train_noisy else tr[~isn[tr]]
    mdl = LogisticRegression(C=CREG, max_iter=500, class_weight='balanced').fit(XA[t], YA[t])
    P = np.zeros((len(te), len(classes)), dtype='float32'); P[:, np.searchsorted(classes, mdl.classes_)] = mdl.predict_proba(XA[te])
    return P


jobs = [(k, tn) for k in range(5) for tn in (False, True)]
res = Parallel(n_jobs=10, verbose=0)(delayed(fit_pred)(folds[k][0], folds[k][1], tn) for k, tn in jobs)
PA = np.zeros((len(YA), len(classes)), dtype='float32'); PB = PA.copy()
for (k, tn), P in zip(jobs, res): (PB if tn else PA)[folds[k][1]] = P
print('cross-validation done', flush=True)


def fuse(P, T=1.0):
    z = (np.log(np.clip(P, 1e-9, 1)) + W * raw) / T; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


Ts = np.linspace(.4, 3, 27)
def best_T(P):
    z0 = np.log(np.clip(P, 1e-9, 1)) + W * raw
    def nll(T):
        z = z0 / T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True); return -np.mean(np.log(p[np.arange(len(YA)), yi] + 1e-9))
    return float(Ts[int(np.argmin([nll(t) for t in Ts]))])


TA_, TB_ = best_T(PA), best_T(PB)
FA, FB = fuse(PA, TA_), fuse(PB, TB_)
topA, topB = classes[FA.argmax(1)], classes[FB.argmax(1)]; mxB = FB.max(1)
# BirdNET alone, restricted to our species list: highest BirdNET score among our species, "background" if its probability < .2
bnl = raw.copy(); bnl[:, ~in_bn] = -99; bnl[:, classes == 'background'] = -99
bn_top = np.where(1 / (1 + np.exp(-bnl.max(1))) >= .2, classes[bnl.argmax(1)], 'background')
# unrestricted BirdNET baseline for clean windows (what was published before)
bn_free = np.where(BC >= .2, BS, 'background')

def acc(top, mask): return float((top[mask] == YA[mask]).mean())
c, n_ = sel & ~isn, sel & isn
report = {'clean': {'birdnet_restricted': acc(bn_top, c), 'clean_only_head': acc(topA, c), 'clean_plus_noisy_head': acc(topB, c)},
          'noisy': {'birdnet_restricted': acc(bn_top, n_), 'clean_only_head': acc(topA, n_), 'clean_plus_noisy_head': acc(topB, n_)}}
print(json.dumps(report, indent=1), flush=True)
by_type = []
for k in ['reeds', 'wind', 'water', 'rain', 'bg', 'chorus', 'mix']:
    mk = n_ & np.r_[np.zeros(nc, bool), TN == k]
    by_type.append({'type': k, 'windows': int(mk.sum()), 'birdnet': acc(bn_top, mk), 'clean_only': acc(topA, mk), 'with_noise': acc(topB, mk)})
snr_all = np.r_[np.full(nc, np.nan), SN]
by_snr = []
for lo, hi in [(-3, 0), (0, 3), (3, 6), (6, 9), (9, 15.1)]:
    mk = n_ & (snr_all >= lo) & (snr_all < hi)
    by_snr.append({'lo': lo, 'hi': min(hi, 15), 'windows': int(mk.sum()), 'birdnet': acc(bn_top, mk), 'clean_only': acc(topA, mk), 'with_noise': acc(topB, mk)})
for r in by_type + by_snr: print(r, flush=True)

# --- public metrics (model B), clean windows = like before; noisy shown separately
mx = FB.max(1)
cal = []
for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    mk = c & (mx >= lo) & (mx < hi) & (topB != 'background')
    if mk.sum(): cal.append({'lo': lo, 'hi': min(hi, 1.0), 'windows': int(mk.sum()), 'share': float(mk.sum() / c.sum()), 'right': float((topB[mk] == YA[mk]).mean())})
cover, cover_noisy = [], []
for t in (.3, .5, .7, .9):
    for lst, base in ((cover, c), (cover_noisy, n_)):
        mk = base & (mx >= t) & (topB != 'background'); lst.append({'t': t, 'covers': float(mk.sum() / base.sum()), 'precision': float((topB[mk] == YA[mk]).mean())})
ok_rec = nrec = 0
for g in set(GA[c]):
    mk = (GA == g) & c; ok_rec += classes[FB[mk].mean(0).argmax()] == YA[mk][0]; nrec += 1
rows = []
for sp in use:
    mk = (YA == sp) & ~isn; mn = (YA == sp) & isn
    if mk.sum() == 0: continue
    com = labels.get(sp) or xc.get(sp) or sp
    rows.append({'sci': sp, 'en': com, 'ru': ru.get(com, ''), 'recordings': rec_per[sp], 'windows': int(mk.sum()), 'head': float((topB[mk] == sp).mean()), 'birdnet': float((bn_free[:nc][YA[:nc] == sp] == sp).mean()),
                 'head_noisy': float((topB[mn] == sp).mean()) if mn.sum() else None, 'birdnet_noisy': float((bn_top[mn] == sp).mean()) if mn.sum() else None, 'in_birdnet': sp in labels})
rows.sort(key=lambda r: -r['head'])
out = {'generated': datetime.date.today().isoformat(), 'version': 'v4', 'species': len(use), 'recordings': int(len(set(GA))), 'windows': int(c.sum()), 'noisy_windows': int(n_.sum()),
       'temperature': TB_, 'fusion_weight': W,
       'window_accuracy': {'birdnet': float((bn_free[sel[:nc]] == Y[sel[:nc]]).mean()), 'wetlandear': report['clean']['clean_plus_noisy_head']}, 'recording_accuracy': ok_rec / nrec,
       'calibration': cal, 'coverage': cover, 'coverage_noisy': cover_noisy,
       'robustness': {'summary': report, 'by_noise_type': by_type, 'by_snr': by_snr},
       'per_species': rows,
       'method': 'Cross-validation split by recording (5 folds): no recording is ever in both training and test. Test windows are 3-second fragments of Xeno-canto recordings; noisy test windows are the same kind of fragments mixed with reed rustle, wind, water, rain, field background or other birds at 12 to -3 dB signal-to-noise.'}
json.dump(out, open(BASE + r'\metrics_v4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('clean: BirdNET(free) %.1f%% -> WetlandEar %.1f%% | per recording %.1f%%' % (out['window_accuracy']['birdnet'] * 100, out['window_accuracy']['wetlandear'] * 100, ok_rec / nrec * 100), flush=True)

if EXPORT:
    fin = LogisticRegression(C=CREG, max_iter=800, class_weight='balanced').fit(XA, YA)
    fcls = [str(x) for x in fin.classes_]
    common = [('' if x == 'background' else (labels.get(x) or xc.get(x) or x)) for x in fcls]
    np.savez_compressed(APP + r'\models\head.npz', coef=fin.coef_.astype('float16'), intercept=fin.intercept_.astype('float32'), classes=np.array(fcls), common=np.array(common),
                        T=np.float32(TB_), w=np.float32(W), missing=np.float32(0.0), bn_labels=np.array([x if x in labels else '' for x in fcls]))
    rows_ = [['species', 'xeno_canto_id', 'url']]
    for p in sorted(glob.glob(BASE + r'\train_audio\*\XC*.mp3')):
        if os.path.getsize(p) > 5000:
            xid = os.path.basename(p)[2:-4]; rows_.append([os.path.basename(os.path.dirname(p)).replace('_', ' '), xid, 'https://xeno-canto.org/' + xid])
    with open(APP + r'\models\head_training_sources.csv', 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerows(rows_)
    print('exported', len(fcls), 'classes', os.path.getsize(APP + r'\models\head.npz') // 1024, 'KB; sources', len(rows_) - 1, flush=True)
