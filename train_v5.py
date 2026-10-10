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
res = Parallel(n_jobs=int(os.environ.get('NJOBS', '5')), verbose=0)(delayed(fit_pred)(folds[k][0], folds[k][1], tn) for k, tn in jobs)
PA = np.zeros((len(YA), len(classes)), dtype='float32'); PB = PA.copy()
for (k, tn), P in zip(jobs, res): (PB if tn else PA)[folds[k][1]] = P
print('cross-validation done', flush=True)


WV = np.full(len(classes), W, dtype='float32')


def fuse(P, T=1.0, wv=None):
    wv = WV if wv is None else wv
    z = (np.log(np.clip(P, 1e-9, 1)) + wv[None, :] * raw) / T; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


Ts = np.linspace(.4, 3, 27)
def best_T(P):
    z0 = np.log(np.clip(P, 1e-9, 1)) + WV[None, :] * raw
    def nll(T):
        z = z0 / T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True); return -np.mean(np.log(p[np.arange(len(YA)), yi] + 1e-9))
    return float(Ts[int(np.argmin([nll(t) for t in Ts]))])


TA_, TB_ = best_T(PA), best_T(PB)
CAND = (0.15, 0.4, 0.8, 1.5)


def tune_weights(P, T, rows_mask, passes=2):
    """Coordinate ascent: for each species with a BirdNET label pick the fusion weight that gives the best accuracy on rows_mask."""
    wv = np.full(len(classes), W, dtype='float32')
    lp = np.log(np.clip(P[rows_mask], 1e-9, 1)); rw = raw[rows_mask]; y = yi[rows_mask]; sl = sel[rows_mask]
    def acc_of(w): return float((np.argmax(lp + w[None, :] * rw, 1)[sl] == y[sl]).mean())
    best = acc_of(wv)
    for _ in range(passes):
        for c in np.where(in_bn)[0]:
            keep = wv[c]; bc = keep
            for cand in CAND:
                if cand == keep: continue
                wv[c] = cand; a_ = acc_of(wv)
                if a_ > best + 2e-5: best, bc = a_, cand
            wv[c] = bc
    return wv, best


# honest check: choose the weights on half of the recordings, test on the other half
rng = np.random.default_rng(7); ug = np.array(sorted(set(GA))); half = set(rng.choice(ug, len(ug) // 2, replace=False))
inA = np.array([g in half for g in GA]); inB = ~inA
base_B = float((np.argmax(np.log(np.clip(PB[inB], 1e-9, 1)) + WV[None, :] * raw[inB], 1)[sel[inB]] == yi[inB][sel[inB]]).mean())
wvA, accA = tune_weights(PB, TB_, inA)
tuned_B = float((np.argmax(np.log(np.clip(PB[inB], 1e-9, 1)) + wvA[None, :] * raw[inB], 1)[sel[inB]] == yi[inB][sel[inB]]).mean())
print('per-species BirdNET weights: held-out recordings %.2f%% -> %.2f%% (chosen on the other half: %.2f%%)' % (base_B * 100, tuned_B * 100, accA * 100), flush=True)
if tuned_B > base_B + 0.0005:
    WV, acc_all = tune_weights(PB, TB_, np.ones(len(YA), bool)); TB_ = best_T(PB)
    print('using per-species weights (all recordings): in-sample %.2f%%, temperature %.2f' % (acc_all * 100, TB_), flush=True)
else:
    print('per-species weights do not help on held-out recordings, keeping one weight', flush=True)
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
VER = os.environ.get('VERSION', 'v6')
out = {'generated': datetime.date.today().isoformat(), 'version': VER, 'species': len(use), 'recordings': int(len(set(GA))), 'windows': int(c.sum()), 'noisy_windows': int(n_.sum()),
       'temperature': TB_, 'fusion_weight': W, 'fusion_weight_per_species': bool(WV.std() > 0),
       'window_accuracy': {'birdnet': float((bn_free[sel[:nc]] == Y[sel[:nc]]).mean()), 'wetlandear': report['clean']['clean_plus_noisy_head']}, 'recording_accuracy': ok_rec / nrec,
       'calibration': cal, 'coverage': cover, 'coverage_noisy': cover_noisy,
       'robustness': {'summary': report, 'by_noise_type': by_type, 'by_snr': by_snr},
       'per_species': rows,
       'method': 'Cross-validation split by recording (5 folds): no recording is ever in both training and test. Test windows are 3-second fragments of Xeno-canto recordings; noisy test windows are the same kind of fragments mixed with reed rustle, wind, water, rain, field background or other birds at 12 to -3 dB signal-to-noise.'}
# Fair baselines (added for v6). "birdnet" above is BirdNET's own free top answer WITH the location filter of one fixed point (Korgalzhyn): a species the filter
# switches off can never be right, so it is not a fair baseline for recordings from other places. The restricted baseline below uses BirdNET's raw scores
# of our species WITHOUT any location filter.
for r in rows:
    mk_ = (YA == r['sci']) & ~isn
    r['birdnet_restricted'] = float((bn_top[mk_] == r['sci']).mean())
known_sp = [r['sci'] for r in rows if r['in_birdnet']]
kn = sel & ~isn & np.isin(YA, known_sp)
out['known_species'] = {'species': len(known_sp), 'windows': int(kn.sum()), 'birdnet_restricted': acc(bn_top, kn), 'wetlandear': acc(topB, kn)}
print('species BirdNET knows: %d | their windows: BirdNET(restricted, no location filter) %.1f%% vs WetlandEar %.1f%%' % (len(known_sp), out['known_species']['birdnet_restricted'] * 100, out['known_species']['wetlandear'] * 100), flush=True)
json.dump(out, open(BASE + r'\metrics_%s.json' % VER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('clean: BirdNET(free) %.1f%% -> WetlandEar %.1f%% | per recording %.1f%%' % (out['window_accuracy']['birdnet'] * 100, out['window_accuracy']['wetlandear'] * 100, ok_rec / nrec * 100), flush=True)

if EXPORT:
    fin = LogisticRegression(C=CREG, max_iter=800, class_weight='balanced').fit(XA, YA)
    fcls = [str(x) for x in fin.classes_]
    common = [('' if x == 'background' else (labels.get(x) or xc.get(x) or x)) for x in fcls]
    np.savez_compressed(APP + r'\models\head.npz', coef=fin.coef_.astype('float16'), intercept=fin.intercept_.astype('float32'), classes=np.array(fcls), common=np.array(common),
                        T=np.float32(TB_), w=WV.astype('float32'), missing=np.float32(0.0), bn_labels=np.array([x if x in labels else '' for x in fcls]))
    rows_ = [['species', 'xeno_canto_id', 'url']]
    for p in sorted(glob.glob(BASE + r'\train_audio\*\XC*.mp3')):
        if os.path.getsize(p) > 5000:
            xid = os.path.basename(p)[2:-4]; rows_.append([os.path.basename(os.path.dirname(p)).replace('_', ' '), xid, 'https://xeno-canto.org/' + xid])
    with open(APP + r'\models\head_training_sources.csv', 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerows(rows_)
    print('exported', len(fcls), 'classes', os.path.getsize(APP + r'\models\head.npz') // 1024, 'KB; sources', len(rows_) - 1, flush=True)
