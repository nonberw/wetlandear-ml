"""Browser version of the WetlandEar head (models/head.npz): head_web.json (names, temperature, fusion weights, BirdNET labels) and head_web.bin (float16 weights).
Run after every head export."""
import json
import numpy as np
APP = r'C:\Users\Timur\Documents\wetlandear-app\models'
z = np.load(APP + r'\head.npz', allow_pickle=False)
coef = z['coef'].astype('<f2')
w = np.asarray(z['w'], dtype='float32')
if w.ndim == 0: w = np.full(coef.shape[0], float(w), dtype='float32')
meta = {'d': int(coef.shape[1]), 'n': int(coef.shape[0]), 'classes': [str(c) for c in z['classes']], 'common': [str(c) for c in z['common']], 'bn_labels': [str(c) for c in z['bn_labels']],
        'T': float(z['T']), 'w': [round(float(x), 5) for x in w], 'b': [round(float(x), 6) for x in z['intercept']]}
json.dump(meta, open(APP + r'\head_web.json', 'w', encoding='utf-8'), ensure_ascii=False)
open(APP + r'\head_web.bin', 'wb').write(coef.tobytes())
print(coef.shape, 'classes;', len(coef.tobytes()) // 1024, 'KB')
