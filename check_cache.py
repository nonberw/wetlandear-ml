"""Are the per-species feature caches in feats2/ complete (same number of files as the folder has now)?"""
import glob, os
import numpy as np
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
bad = []
for d in sorted(glob.glob(BASE + r'\train_audio\*')):
    n = os.path.basename(d)
    files = [f for f in glob.glob(d + r'\*') if os.path.getsize(f) > 5000]
    c = BASE + r'\feats2' + '\\' + n + '.npz'
    if not os.path.exists(c):
        bad.append((n, 'missing')); continue
    try:
        z = np.load(c, allow_pickle=True)
        if int(z['nfiles']) != len(files): bad.append((n, 'cache %d vs folder %d' % (int(z['nfiles']), len(files))))
    except Exception as e:
        bad.append((n, 'unreadable'))
print('folders with a stale or missing cache:', len(bad))
for b in bad[:20]: print(' ', b)
