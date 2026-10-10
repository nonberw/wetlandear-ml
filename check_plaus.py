"""Which species does the location filter (BirdNET meta model at Korgalzhyn, 50.6 N 69.0 E) switch off? Compares with the species whose 'free BirdNET' accuracy in metrics_v5 is exactly 0."""
import json, sys
import numpy as np
sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api'); sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\lib')
import identify as I
s = I._load()
plaus = I._run(s['meta'], np.array([[50.6, 69.0, 1.0]], dtype='float32'))
lab = [l.split('_')[0] for l in s['labels']]
M = json.load(open(r'C:\Users\Timur\Documents\wetlandear-models\backup_v5\metrics_v5.json', encoding='utf-8'))
zero = [r for r in M['per_species'] if r['in_birdnet'] and r['birdnet'] == 0]
masked_all = [r for r in M['per_species'] if r['in_birdnet'] and plaus[lab.index(r['sci'])] <= 0.03]
print('species in BirdNET whose free accuracy is exactly 0:', len(zero))
print('species in BirdNET that the location filter switches off (meta score <= 0.03):', len(masked_all))
zs = {r['sci'] for r in zero}; ms = {r['sci'] for r in masked_all}
print('zero AND masked:', len(zs & ms), '| zero but NOT masked:', sorted(zs - ms), '| masked but not zero:', sorted(ms - zs))
print('mean free accuracy of masked species: %.3f ; of the other in-BirdNET species: %.3f' % (np.mean([r['birdnet'] for r in masked_all]), np.mean([r['birdnet'] for r in M['per_species'] if r['in_birdnet'] and r['sci'] not in ms])))
print('windows share of masked species: %.3f' % (sum(r['windows'] for r in masked_all) / sum(r['windows'] for r in M['per_species'])))

json.dump(sorted(ms), open('C:/Users/Timur/Documents/wetlandear-ml/paper/figs/masked.json', 'w'))
print("saved", len(ms))
