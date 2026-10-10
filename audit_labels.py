"""Audit of the training folders: files that are not named XC<number> or INAT<number> came from the first, manual stage. If the file name carries a different species
than the folder, the label is probably wrong. Writes the list to audit_labels.json; with --move the suspicious files go to train_quarantine/ (nothing is deleted)."""
import json, os, re, shutil, sys
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
TA = os.path.join(BASE, 'train_audio'); QUAR = os.path.join(BASE, 'train_quarantine')
SYN = {'Grus_virgo': 'Anthropoides_virgo', 'Ichthyaetus_ichthyaetus': 'Larus_ichthyaetus'}
bin_re = re.compile(r'([A-Z][a-z]+)[_ ]([a-z]{3,})')
rows = []
for d in sorted(os.listdir(TA)):
    folder = d
    for f in sorted(os.listdir(os.path.join(TA, d))):
        if re.match(r'^(XC|INAT)\d+\.', f):
            continue
        names = ['%s_%s' % m for m in bin_re.findall(f)]
        ok = any(n == folder or n == SYN.get(folder) for n in names)
        rows.append({'folder': folder, 'file': f, 'binomials_in_name': names, 'matches_folder': ok or not names, 'unknown_name': not names})
bad = [r for r in rows if not r['matches_folder']]
json.dump({'checked': rows, 'suspicious': bad}, open(os.path.join(BASE, 'audit_labels.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('manual-stage files: %d | name says another species: %d | no species in name: %d' % (len(rows), len(bad), sum(r['unknown_name'] for r in rows)))
for r in bad: print('  %-26s %s' % (r['folder'], r['file']))
if '--move' in sys.argv:
    for r in bad:
        dst = os.path.join(QUAR, r['folder']); os.makedirs(dst, exist_ok=True)
        shutil.move(os.path.join(TA, r['folder'], r['file']), os.path.join(dst, r['file']))
    print('moved', len(bad), 'files to train_quarantine')
