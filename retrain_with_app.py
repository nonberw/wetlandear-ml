"""Retrain the WetlandEar head with the confirmed recordings exported from the app (Data > "For model training").
Put the exported *.json files into the folder `app_exports/` next to the other data and run:  python retrain_with_app.py
Steps: (1) cross-validated numbers on the Xeno-canto data as before, (2) a FIELD CHECK: how the XC-only model does on your confirmed recordings and how
often the suggestion the app showed you was right, (3) final fit on Xeno-canto + your confirmed samples (weight 3) and export head.npz."""
import base64, csv, glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
APP = r'C:\Users\Timur\Documents\wetlandear-app'
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
APP_WEIGHT = 3.0
dirs = sorted(glob.glob(BASE + r'\train_audio\*'))
cls_all = [SYN.get(os.path.basename(d).replace('_', ' '), os.path.basename(d).replace('_', ' ')) for d in dirs]
labels, by_common = {}, {}
for l in open(BASE + r'\x\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels.txt', encoding='utf-8'):
    s, _, c = l.strip().partition('_'); labels[s] = c; by_common[c.lower()] = s
xc = json.load(open(BASE + r'\xc_names.json', encoding='utf-8'))
for sci, en in xc.items():
    if en: by_common.setdefault(en.lower(), SYN.get(sci, sci))


def unpack(t):
    sc, b = t.split('|'); q = np.frombuffer(base64.b64decode(b), dtype=np.int8).astype('float32') * float(sc)
    return q / (np.linalg.norm(q) + 1e-9)


# ---------- Xeno-canto features ----------
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

# ---------- confirmed recordings from the app ----------
AX, AY, ASUG, ATS = [], [], [], []
for f in sorted(glob.glob(BASE + r'\app_exports\*.json')):
    j = json.load(open(f, encoding='utf-8'))
    if j.get('format') != 'wetlandear-training-v1': continue
    for it in j['items']:
        name = it.get('latin') or by_common.get((it.get('name') or '').lower())
        cls = SYN.get(name, name) if name else 'x:' + (it.get('name') or '?')
        try: AX.append(unpack(it['emb']))
        except Exception: continue
        AY.append(cls); ASUG.append(it.get('suggest')); ATS.append(it.get('ts'))
AX = np.array(AX, dtype='float32') if AX else np.zeros((0, 1024), dtype='float32'); AY = np.array(AY)
cnt = {c: int((AY == c).sum()) for c in set(AY)}
akeep = [c for c, n in cnt.items() if n >= 3 or c in use]
print('XC windows %d, species %d | app samples %d (usable classes: %s)' % (len(Y), len(use), len(AY), ', '.join('%s %d' % (c, cnt[c]) for c in akeep) or 'none'), flush=True)
amask = np.isin(AY, akeep) if len(AY) else np.zeros(0, bool)
AX, AY = AX[amask], AY[amask]; ASUG = [s for s, m in zip(ASUG, amask) if m]

# ---------- (1) honest CV on Xeno-canto (unchanged protocol) ----------
col = {c: i for i, c in enumerate(cls_all)}; in_bn = None
P = None
for tr, te in GroupKFold(n_splits=5).split(XB, Y, G):
    m = LogisticRegression(C=2.0, max_iter=500, class_weight='balanced').fit(XB[tr], Y[tr])
    if P is None: classes = np.array(m.classes_); P = np.zeros((len(Y), len(classes)), dtype='float32')
    P[te] = m.predict_proba(XB[te])
sel = Y != 'background'
print('Xeno-canto CV window accuracy (head alone): %.1f%%' % ((classes[np.argmax(P, 1)][sel] == Y[sel]).mean() * 100), flush=True)

# ---------- (2) field check on the app's confirmed samples ----------
if len(AY) >= 5:
    m0 = LogisticRegression(C=2.0, max_iter=500, class_weight='balanced').fit(XB, Y)
    pr = m0.predict(AX); pp = m0.predict_proba(AX).max(1)
    known = np.isin(AY, m0.classes_)
    print('\nFIELD CHECK on %d confirmed samples from the app (%d with a class the model knows):' % (len(AY), known.sum()))
    print('  Xeno-canto-only head is right on %.1f%% of the known ones; %.1f%% at confidence >= 0.5 (%d samples)' % (
        (pr[known] == AY[known]).mean() * 100, ((pr == AY) & (pp >= .5) & known).sum() / max(1, ((pp >= .5) & known).sum()) * 100, ((pp >= .5) & known).sum()))
    sug = [(s, a) for s, a in zip(ASUG, AY) if s and s.get('sci')]
    if sug:
        print('  the suggestion the app showed was right in %.1f%% of %d cases' % (np.mean([SYN.get(s['sci'], s['sci']) == a for s, a in sug]) * 100, len(sug)))
else:
    print('\n(not enough app samples for a field check yet: need 5, have %d)' % len(AY))

# ---------- (3) final fit with the app samples (weighted) ----------
Xf = np.vstack([XB, AX]) if len(AY) else XB; Yf = np.concatenate([Y, AY]) if len(AY) else Y
w = np.concatenate([np.ones(len(Y)), np.full(len(AY), APP_WEIGHT)]) if len(AY) else np.ones(len(Y))
fin = LogisticRegression(C=2.0, max_iter=800, class_weight='balanced').fit(Xf, Yf, sample_weight=w)
fcls = [str(c) for c in fin.classes_]
common = [('' if c == 'background' else (labels.get(c) or xc.get(c) or (c[2:] if c.startswith('x:') else c))) for c in fcls]
prev = np.load(APP + r'\models\head.npz', allow_pickle=False)
np.savez_compressed(APP + r'\models\head.npz', coef=fin.coef_.astype('float16'), intercept=fin.intercept_.astype('float32'), classes=np.array(fcls), common=np.array(common),
                    T=prev['T'], w=prev['w'], missing=prev['missing'], bn_labels=np.array([c if c in labels else '' for c in fcls]))
print('\nexported head.npz with %d classes (kept T=%.2f, fusion w=%.2f from the last calibrated version; re-run export_head2.py to recalibrate)' % (len(fcls), float(prev['T']), float(prev['w'])))
