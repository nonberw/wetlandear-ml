import json, os, shutil, re
APP = r'C:\Users\Timur\Documents\wetlandear-app'
SITE = r'C:\Users\Timur\Documents\wetlandear'
ML = r'C:\Users\Timur\Documents\wetlandear-ml'
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
NEW = {'Green Sandpiper': 'Черныш', "Temminck's Stint": 'Белохвостый песочник', 'Bar-tailed Godwit': 'Малый веретенник', 'Red-necked Phalarope': 'Круглоносый плавунчик',
       'Black-winged Pratincole': 'Луговая тиркушка', 'Lesser Sand-Plover': 'Монгольский зуёк', 'Terek Sandpiper': 'Мородунка', 'Curlew Sandpiper': 'Краснозобик', 'Greater Sand-Plover': 'Толстоклювый зуёк'}
p = APP + r'\js\species-ru.js'
js = open(p, encoding='utf-8').read(); cur = json.loads(js[js.index('{'):js.rindex('}') + 1]); cur.update(NEW)
open(p, 'w', encoding='utf-8').write('/* Russian names for species the app can suggest. Key: English common name (as used by BirdNET). */\nwindow.WE_RU = ' + json.dumps(cur, ensure_ascii=False, indent=0) + ';\n')
m = json.load(open(BASE + r'\metrics_v4.json', encoding='utf-8'))
for r in m['per_species']:
    if not r['ru'] and r['en'] in cur: r['ru'] = cur[r['en']]
json.dump(m, open(BASE + r'\metrics_v4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
shutil.copy(BASE + r'\metrics_v4.json', SITE + r'\data\metrics.json')
# ml repo
for src, dst in [('chain_train.log', 'v4_train_and_robustness.log'), ('metrics_v4.json', 'v4_metrics.json')]:
    shutil.copy(BASE + '\\' + src, ML + r'\reports\\' + dst)
for f in ['fetch_xc6.py', 'build_noisy.py', 'train_v4.py', 'make_species_js.py', 'final_chain.sh', 'ship_v4.py']:
    shutil.copy(BASE + '\\' + f, ML + '\\' + f)
print(len(cur), 'ru names;', sum(1 for r in m['per_species'] if not r['ru']), 'still missing')
