import glob, os, json
import numpy as np
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
labels = {}
for l in open(BASE + r'\x\V2.4\BirdNET_GLOBAL_6K_V2.4_Labels.txt', encoding='utf-8'):
    s, _, c = l.strip().partition('_'); labels[s] = c
xc = json.load(open(BASE + r'\xc_names.json', encoding='utf-8'))
out = []
for d in sorted(glob.glob(BASE + r'\train_audio\*')):
    n = os.path.basename(d).replace('_', ' '); sci = SYN.get(n, n)
    f = BASE + '\\feats2\\' + os.path.basename(d) + '.npz'
    if not os.path.exists(f): continue
    z = np.load(f, allow_pickle=True)
    nrec = len(set(z['G'][z['Y'] != 'background'])) if len(z['Y']) else 0
    if nrec < 2: continue
    out.append((sci, labels.get(sci) or xc.get(n) or sci, nrec))
print(len(out))
print('; '.join('%s=%s' % (a, b) for a, b, c in out))
json.dump(out, open(BASE + r'\classes_en.json', 'w', encoding='utf-8'), ensure_ascii=False)
