"""Puts the v6 head into the app: head.npz, browser files, name table, training-size table and the list of training sources (Xeno-canto and iNaturalist ids with links)."""
import csv, glob, json, os, re, shutil, subprocess, sys
BASE = r'C:\Users\Timur\Documents\wetlandear-models'; APP = r'C:\Users\Timur\Documents\wetlandear-app'
shutil.copy(BASE + r'\backup_v6\head.npz', APP + r'\models\head.npz')
shutil.copy(BASE + r'\backup_v6\species-sci.js', APP + r'\js\species-sci.js')
print('head.npz and species-sci.js copied')
subprocess.check_call([sys.executable, BASE + r'\make_head_web.py'])
M = json.load(open(BASE + r'\metrics_v6.json', encoding='utf-8'))
json.dump({r['sci']: r['recordings'] for r in M['per_species']}, open(APP + r'\models\data_size.json', 'w', encoding='utf-8'), separators=(',', ':'), ensure_ascii=False)
# The Xeno-canto list (head_training_sources.csv) is read by the reference-voice player and did not change, so it stays as it is.
# The iNaturalist recordings go to their own file.
rows = [['species', 'inaturalist_sound_id', 'url']]
for d in sorted(glob.glob(BASE + r'\train_audio\*')):
    sp = os.path.basename(d).replace('_', ' ')
    for f in sorted(os.listdir(d)):
        n = re.match(r'^INAT(\d+)\.', f)
        if n and os.path.getsize(os.path.join(d, f)) > 5000: rows.append([sp, n.group(1), 'https://www.inaturalist.org/sounds/' + n.group(1)])
with open(APP + r'\models\head_training_sources_inat.csv', 'w', newline='', encoding='utf-8') as fh:
    csv.writer(fh).writerows(rows)
print('data_size.json: %d species; iNaturalist sources: %d' % (len(M['per_species']), len(rows) - 1))
