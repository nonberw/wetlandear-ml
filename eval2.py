"""Does aggregating several windows per recording help? Uses every real recording collected so far."""
import glob, os, sys, warnings
import numpy as np, soundfile as sf
warnings.filterwarnings('ignore')
from scipy.signal import resample_poly
sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api')
import identify as I

SYN = {'Grus virgo': 'Anthropoides virgo'}   # taxonomy synonym used by Commons vs BirdNET
base = r'C:\Users\Timur\Documents\wetlandear-models'
items = []
for p in glob.glob(base + r'\train_audio\*\*') + glob.glob(base + r'\samples2\*'):
    sp = os.path.basename(os.path.dirname(p)).replace('_', ' ') if 'train_audio' in p else ' '.join(os.path.basename(p).split('_')[:2])
    items.append((SYN.get(sp, sp), p))
for p in glob.glob(base + r'\samples\*.mp3'):
    sp = ' '.join(os.path.basename(p).split('_')[:2]); items.append((SYN.get(sp, sp), p))

s = I._load()
labels = s['labels']; lab_sci = [l.split('_')[0] for l in labels]
plaus = I._run(s['meta'], np.array([[50.6, 69.0, 1.0]], dtype='float32')) > 0.03


def load(p):
    x, sr = sf.read(p, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    return resample_poly(x, 48000, sr).astype('float32')


def windows(x, k):
    n = 144000
    if len(x) < n: x = np.pad(x, (0, n - len(x)))
    ws = [x[i:i + n] for i in range(0, len(x) - n + 1, n // 2)]
    en = [float((w ** 2).mean()) for w in ws]
    return [ws[i] for i in np.argsort(en)[::-1][:k]]


def logits(w): return I._run(s['clf'], w.reshape(1, -1)).copy()


rows = []
for sp, p in items:
    if sp not in lab_sci: continue   # species BirdNET does not know (pelicans): cannot be scored here
    try: ws = windows(load(p), 5)
    except Exception: continue
    L = np.array([logits(w) for w in ws])
    P = 1 / (1 + np.exp(-np.clip(L, -15, 15))); P = np.where(plaus, P, 0)
    tgt = lab_sci.index(sp)
    single = np.mean([np.argmax(P[i]) == tgt for i in range(len(ws))])          # per-window accuracy
    top1_first = np.argmax(P[0]) == tgt                                          # strongest window only
    mean_p = P.mean(0); agg_mean = np.argmax(mean_p) == tgt                      # mean over windows
    max_p = P.max(0); agg_max = np.argmax(max_p) == tgt                          # max over windows
    rows.append((sp, single, top1_first, agg_mean, agg_max))

n = len(rows)
print(f'recordings scored: {n}')
print(f'per-window accuracy (avg of 5 strongest windows) : {np.mean([r[1] for r in rows])*100:5.1f}%')
print(f'strongest window only                            : {np.mean([r[2] for r in rows])*100:5.1f}%')
print(f'mean probability over 5 windows (recording-level): {np.mean([r[3] for r in rows])*100:5.1f}%')
print(f'max probability over 5 windows (recording-level) : {np.mean([r[4] for r in rows])*100:5.1f}%')
miss = [r[0] for r in rows if not r[3]]
print('missed by mean-aggregation:', sorted(set(miss)))
