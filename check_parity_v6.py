"""Server head (api/identify.py head_top) against the browser files (head_web.json/bin) for real windows of 12 species, same formula as js/offline.js."""
import json, os, sys
import numpy as np
sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api'); sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\lib')
import identify as I
import soundfile as sf
from scipy.signal import resample_poly
APPM = r'C:\Users\Timur\Documents\wetlandear-app\models'
meta = json.load(open(APPM + r'\head_web.json', encoding='utf-8'))
coef = np.frombuffer(open(APPM + r'\head_web.bin', 'rb').read(), dtype='<f2').astype('float32').reshape(meta['n'], meta['d'])
b = np.array(meta['b'], dtype='float32'); w = np.array(meta['w'], dtype='float32'); T = meta['T']
s = I._load(); labels = s['labels']; sci_idx = {l.split('_')[0]: i for i, l in enumerate(labels)}
bn_idx = np.array([sci_idx.get(l, -1) if l else -1 for l in meta['bn_labels']])


def web_top(vec, logits):
    v = vec / (np.linalg.norm(vec) + 1e-9); z = coef @ v + b
    logp = z - (z.max() + np.log(np.exp(z - z.max()).sum())); bn = np.zeros(len(logp), dtype='float32'); ok = bn_idx >= 0; bn[ok] = np.clip(logits[bn_idx[ok]], -15, 15)
    f = (logp + w * bn) / T; p = np.exp(f - f.max()); p /= p.sum(); k = int(np.argmax(p)); return meta['classes'][k], float(p[k])


sp = ['Fulica_atra', 'Alauda_arvensis', 'Grus_grus', 'Cygnus_cygnus', 'Larus_ichthyaetus', 'Anser_anser', 'Falco_cherrug', 'Pelecanus_onocrotalus', 'Luscinia_megarhynchos', 'Coloeus_monedula', 'Aquila_clanga', 'Podiceps_cristatus']
same = n = 0; worst = 0.0
for name in sp:
    d = os.path.join(r'C:\Users\Timur\Documents\wetlandear-models\train_audio', name)
    if not os.path.isdir(d): continue
    f = [x for x in sorted(os.listdir(d)) if x.startswith(('XC', 'INAT'))][1]
    x, sr = sf.read(os.path.join(d, f), dtype='float32'); x = x.mean(1) if x.ndim > 1 else x; x = resample_poly(x, 48000, sr).astype('float32')
    en = [float((x[i:i + 144000] ** 2).mean()) for i in range(0, max(1, len(x) - 144000), 48000)]; i = int(np.argmax(en)) * 48000
    wv = np.pad(x[i:i + 144000], (0, max(0, 144000 - len(x[i:i + 144000]))))
    logits = I._run(s['clf'], wv.reshape(1, -1)); emb, vec = I.fingerprint(s['clf'])
    srv = I.head_top(vec, logits); sv = (srv[0]['sci'], srv[0]['conf']) if srv else (None, 0.0); wb = web_top(vec, logits)
    n += 1; same += (sv[0] == wb[0] or not srv); worst = max(worst, abs(sv[1] - wb[1]) if srv and sv[0] == wb[0] else 0)
    print('%-24s server %-26s %.3f | browser %-26s %.3f' % (name, sv[0], sv[1], wb[0], wb[1]))
print('same top species: %d of %d | largest confidence difference %.4f' % (same, n, worst))
