"""Remove BirdNET-score caches (feats3) that no longer match the rebuilt feature caches (feats2), so that only changed species are recomputed."""
import glob, os
import numpy as np
B = r'C:\Users\Timur\Documents\wetlandear-models'
gone = 0
for f2 in glob.glob(B + r'\feats2\*.npz'):
    n = os.path.basename(f2)[:-4]; f3 = B + r'\feats3\%s.npy' % n
    if not os.path.exists(f3): continue
    try:
        if len(np.load(f2, allow_pickle=True)['Y']) != len(np.load(f3)):
            os.remove(f3); gone += 1
    except Exception:
        os.remove(f3); gone += 1
print('removed', gone, 'stale logit caches')
