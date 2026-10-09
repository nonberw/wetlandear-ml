"""Train a small classifier ("head") on BirdNET sound fingerprints for Korgalzhyn species and compare it with BirdNET alone.
Run:  python -I train_head.py [--fetch KEY] [--export]
  --fetch KEY  download more recordings from Xeno-canto (API v3 key from https://xeno-canto.org/account)
  --export     fit on everything and write C:\\...\\wetlandear-app\\models\\head.json (only do this when CV shows a gain)
Honest evaluation: recordings are split by file (never the same recording in train and test)."""
import glob, json, os, sys, time, urllib.parse, urllib.request, warnings
import numpy as np, soundfile as sf
warnings.filterwarnings('ignore')
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api')
import identify as I

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
SYN = {'Grus virgo': 'Anthropoides virgo'}
SPECIES = ['Pelecanus crispus', 'Pelecanus onocrotalus', 'Phoenicopterus roseus', 'Grus grus', 'Anthropoides virgo', 'Cygnus cygnus', 'Anser anser', 'Anser erythropus',
           'Tadorna ferruginea', 'Fulica atra', 'Podiceps cristatus', 'Larus ichthyaetus', 'Chlidonias hybrida', 'Vanellus vanellus', 'Acrocephalus arundinaceus',
           'Luscinia svecica', 'Alauda arvensis', 'Melanocorypha yeltoniensis', 'Panurus biarmicus', 'Botaurus stellaris', 'Recurvirostra avosetta', 'Himantopus himantopus']


def fetch_xc(key, per=15):
    UA = {'User-Agent': 'WetlandEar-training/1.0 (contact zzhaseka@gmail.com)'}
    for sp in SPECIES:
        d = os.path.join(BASE, 'train_audio', sp.replace(' ', '_')); os.makedirs(d, exist_ok=True)
        q = urllib.parse.quote('sp:"%s" q:">C" len:5-60' % sp)
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request(f'https://xeno-canto.org/api/3/recordings?query={q}&key={key}', headers=UA), timeout=60))
        except Exception as e:
            print('xc fail', sp, e); continue
        n = 0
        for r in j.get('recordings', []):
            if n >= per: break
            url = r.get('file'); lic = r.get('lic', '')
            if not url or 'nc' in lic.lower() and False: continue
            path = os.path.join(d, 'XC' + str(r['id']) + '.mp3')
            if not os.path.exists(path):
                try: open(path, 'wb').write(urllib.request.urlopen(urllib.request.Request(url if url.startswith('http') else 'https:' + url, headers=UA), timeout=90).read())
                except Exception: continue
                time.sleep(.4)
            n += 1
        print('fetched', sp, n)


if '--fetch' in sys.argv:
    fetch_xc(sys.argv[sys.argv.index('--fetch') + 1])

items = []
for p in glob.glob(BASE + r'\train_audio\*\*'):
    items.append((SYN.get(os.path.basename(os.path.dirname(p)).replace('_', ' '), os.path.basename(os.path.dirname(p)).replace('_', ' ')), p))
for p in glob.glob(BASE + r'\samples2\*'):
    sp = ' '.join(os.path.basename(p).split('_')[:2]); items.append((SYN.get(sp, sp), p))
for p in glob.glob(BASE + r'\samples\*.mp3'):
    sp = ' '.join(os.path.basename(p).split('_')[:2]); items.append((SYN.get(sp, sp), p))
seen, uniq = set(), []
for sp, p in items:   # same recording may appear in two folders
    k = (sp, os.path.getsize(p))
    if k in seen: continue
    seen.add(k); uniq.append((sp, p))
items = uniq

s = I._load(); labels = s['labels']; lab_sci = [l.split('_')[0] for l in labels]
plaus = I._run(s['meta'], np.array([[50.6, 69.0, 1.0]], dtype='float32')) > 0.03


def load(p):
    x, sr = sf.read(p, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    return resample_poly(x, 48000, sr).astype('float32')


X, Y, G, BN = [], [], [], []
for gi, (sp, p) in enumerate(items):
    try: x = load(p)
    except Exception: continue
    n = 144000
    if len(x) < n: x = np.pad(x, (0, n - len(x)))
    ws = [x[i:i + n] for i in range(0, len(x) - n + 1, n // 2)]
    en = np.array([float((w ** 2).mean()) for w in ws]); order = np.argsort(en)[::-1]
    strong = order[:min(10, len(ws))]; weak = order[-2:] if len(ws) > 6 else []
    for idx, lab in [(i, sp) for i in strong] + [(i, 'background') for i in weak]:
        w = ws[idx]; L = I._run(s['clf'], w.reshape(1, -1)).copy()
        e = I._run.__globals__['np'].asarray(s['clf'].get_tensor(s['clf'].get_output_details()[0]['index'] - 1)[0]).copy()
        P = 1 / (1 + np.exp(-np.clip(L, -15, 15))); P = np.where(plaus, P, 0)
        X.append(e / np.linalg.norm(e)); Y.append(lab); G.append(gi); BN.append(lab_sci[int(np.argmax(P))] if P.max() >= .2 else 'background')
X = np.array(X); Y = np.array(Y); G = np.array(G); BN = np.array(BN)
print(f'recordings {len(set(G))}, windows {len(X)}, species {len(set(Y)-{"background"})}')

# keep species with at least 2 recordings for cross-validation
rec_per = {sp: len(set(G[Y == sp])) for sp in set(Y)}
cv_species = [sp for sp, n in rec_per.items() if n >= 2 and sp != 'background']
mask = np.isin(Y, cv_species + ['background'])
print('species with >=2 recordings (used for honest CV):', len(cv_species))
Xc, Yc, Gc, BNc = X[mask], Y[mask], G[mask], BN[mask]
folds = GroupKFold(n_splits=min(5, len(set(Gc))))
head_ok = bird_ok = tot = 0; per = {}
for tr, te in folds.split(Xc, Yc, Gc):
    if len(set(Yc[tr])) < 3: continue
    clf = LogisticRegression(C=8.0, max_iter=2000, class_weight='balanced').fit(Xc[tr], Yc[tr])
    pred = clf.predict(Xc[te])
    for y, ph, pb in zip(Yc[te], pred, BNc[te]):
        if y == 'background': continue
        tot += 1; head_ok += (ph == y); bird_ok += (pb == y)
        d = per.setdefault(y, [0, 0, 0]); d[0] += 1; d[1] += (ph == y); d[2] += (pb == y)
print(f'\nCV per-window accuracy on unseen recordings ({tot} windows):')
print(f'  BirdNET alone : {bird_ok / tot * 100:5.1f}%')
print(f'  trained head  : {head_ok / tot * 100:5.1f}%')
for sp, (n, h, b) in sorted(per.items()): print(f'    {sp:28s} n={n:3d}  head {h/n*100:5.1f}%  birdnet {b/n*100:5.1f}%')
print('\nspecies with only 1 recording (cannot be cross-validated honestly):', sorted(sp for sp, n in rec_per.items() if n < 2 and sp != 'background'))

if '--export' in sys.argv:
    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight='balanced').fit(X, Y)
    out = {'classes': list(clf.classes_), 'coef': np.round(clf.coef_, 4).tolist(), 'intercept': np.round(clf.intercept_, 4).tolist(), 'trained_on': {'recordings': int(len(set(G))), 'windows': int(len(X))}}
    json.dump(out, open(r'C:\Users\Timur\Documents\wetlandear-app\models\head.json', 'w'))
    print('exported head.json')
