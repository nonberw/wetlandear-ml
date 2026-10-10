"""Negatives for the in-app voice classifier: a few BirdNET fingerprints of every species plus some background windows (about 0.6 MB).
They teach the small classifier what "other birds and noise" look like, so a voice taught from a handful of recordings is not confused with them.
Output: wetlandear-app/models/negatives.bin (int8, unit vectors x 127) and negatives.json."""
import glob, json
import numpy as np
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
APP = r'C:\Users\Timur\Documents\wetlandear-app'
rng = np.random.default_rng(11)
rows, sci = [], []
for f in sorted(glob.glob(BASE + r'\feats2\*.npz')):
    z = np.load(f, allow_pickle=True); Y, G, X = z['Y'], z['G'], z['XB']
    m = Y != 'background'
    if m.sum() == 0: continue
    sp = str(Y[m][0]); gs = sorted(set(G[m])); rng.shuffle(gs)
    for g in gs[:2]:   # two different recordings, one fragment each
        idx = np.where((G == g) & m)[0]; rows.append(X[rng.choice(idx)]); sci.append(sp)
    bg = np.where(Y == 'background')[0]
    if len(bg):
        for i in rng.choice(bg, min(1, len(bg)), replace=False): rows.append(X[i]); sci.append('background')
X = np.array(rows, dtype=np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
q = np.clip(np.round(X * 127), -127, 127).astype(np.int8)
open(APP + r'\models\negatives.bin', 'wb').write(q.tobytes())
json.dump({'d': int(X.shape[1]), 'n': int(X.shape[0]), 'sci': sci, 'note': 'fingerprints of Xeno-canto recordings, one fragment each; used only as negatives'}, open(APP + r'\models\negatives.json', 'w'))
print(X.shape, q.nbytes // 1024, 'KB')
