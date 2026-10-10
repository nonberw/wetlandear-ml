import os, sys
import numpy as np
sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api'); sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\lib')
import identify as I
import soundfile as sf
from scipy.signal import resample_poly
d = os.path.join(r'C:\Users\Timur\Documents\wetlandear-models\train_audio', 'Fulica_atra')
f = os.path.join(d, sorted(os.listdir(d))[0])
x, sr = sf.read(f, dtype='float32')
x = x.mean(1) if x.ndim > 1 else x
x = resample_poly(x, 48000, sr).astype('float32')
w = x[48000 * 2:48000 * 2 + 144000]
r = I.identify([w], 50.6, 69.0)[0]
print('species', [(s['common'], s['conf']) for s in r['species'][:3]], '| head', [(s['common'], s['conf']) for s in r['head'][:2]], '| emb chars', len(r['emb']), '| sound', r['sound']['groups'])
