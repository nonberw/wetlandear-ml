"""Reference for the browser: three real windows (int16, base64) and what the SERVER code answers for them."""
import base64, json, os, sys
import numpy as np
sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api'); sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\lib')
import identify as I
import soundfile as sf
from scipy.signal import resample_poly
wins, srcs = [], []
for sp in ('Fulica_atra', 'Alauda_arvensis', 'Grus_grus'):
    d = os.path.join(r'C:\Users\Timur\Documents\wetlandear-models\train_audio', sp)
    f = os.path.join(d, sorted(os.listdir(d))[2])
    x, sr = sf.read(f, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    x = resample_poly(x, 48000, sr).astype('float32')
    en = [float((x[i:i + 144000] ** 2).mean()) for i in range(0, max(1, len(x) - 144000), 48000)]
    i = int(np.argmax(en)) * 48000
    w = x[i:i + 144000]; w = np.pad(w, (0, 144000 - len(w)))
    i16 = np.clip(np.round(w * 32767), -32768, 32767).astype('<i2'); wins.append(base64.b64encode(i16.tobytes()).decode()); srcs.append(sp)
pcm = [np.frombuffer(base64.b64decode(b), dtype='<i2').astype('float32') / 32768.0 for b in wins]
res = I.identify(pcm, 50.6, 69.0)
json.dump({'lat': 50.6, 'lon': 69.0, 'windows': wins, 'sources': srcs, 'server': res}, open(r'C:\Users\Timur\Documents\wetlandear-app\_parity.json', 'w'))
for s, r in zip(srcs, res): print(s, [(a['common'], a['conf']) for a in r['species'][:3]], '| head', [(a['common'], a['conf']) for a in r['head'][:2]], '| groups', {k: v for k, v in r['sound']['groups'].items() if v > .1})
