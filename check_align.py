"""Why 264 species instead of 268? train_v5.py silently skips a folder when len(feats2) != len(feats3), and drops species with fewer than 2 recordings."""
import glob, os
import numpy as np
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
skipped, few = [], []
for d in sorted(glob.glob(BASE + r'\train_audio\*')):
    n = os.path.basename(d); f2 = BASE + '\\feats2\\' + n + '.npz'; f3 = BASE + '\\feats3\\' + n + '.npy'
    if not (os.path.exists(f2) and os.path.exists(f3)):
        skipped.append((n, 'missing cache', os.path.exists(f2), os.path.exists(f3))); continue
    z = np.load(f2, allow_pickle=True); l = np.load(f3)
    if len(z['Y']) != len(l) or len(l) == 0:
        skipped.append((n, 'length mismatch feats2=%d feats3=%d' % (len(z['Y']), len(l)))); continue
    sp = set(z['Y'][z['Y'] != 'background'])
    nrec = len(set(z['G'][z['Y'] != 'background']))
    if nrec < 2: few.append((n, nrec))
print('skipped folders:', len(skipped)); [print(' ', s) for s in skipped]
print('folders with fewer than 2 usable recordings:', len(few)); [print(' ', s) for s in few]
