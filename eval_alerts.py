"""How reliable are the gunshot and engine alerts (YAMNet groups)? Positives: open-licence gunshot and engine/chainsaw recordings from Wikimedia Commons.
Negatives: thunder, rain, wind, a door slam, and 1,200 3-second windows of the Xeno-canto bird recordings. Rules of the app: gun >= 0.35; engine >= 0.5 and animal < 0.5."""
import json, os, re, sys, urllib.request
import numpy as np
sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api'); sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\lib')
import identify as I
import soundfile as sf
from scipy.signal import resample_poly
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
D = BASE + r'\commons_test'; os.makedirs(D, exist_ok=True)
UA = {'User-Agent': 'WetlandEar-research/1.0 (zzhaseka@gmail.com)'}
res = json.load(open(BASE + r'\commons_audio.json'))
def api(params):
    import urllib.parse
    return json.load(urllib.request.urlopen(urllib.request.Request('https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(dict(params, format='json')), headers=UA), timeout=30))
pick = {}
by_title = {x['title']: x for q in res.values() for x in q}
GUN = ['9 mm gunshot-mike-koenig-123.wav', 'Dragout - War sounds (cc0) (freesound).ogg', 'Gunshots 8.ogg', 'Puška Puca.ogg', 'Wurstelprater Wien 2024 Luftgewehr Pressluft schießt zweimal.ogg']
ENG = ['Chainsaw %d.ogg' % i for i in range(1, 13)] + ['WWS Chainsaw.ogg', 'WWS MSThti.ogg', '2006 Porsche Cayman S engine sound.wav', 'Combustion Engine Drives Sawmill (1988).mp3']
DOOR = ['Slam closet door (Gravity Sound).wav']
for g, names in (('gun', GUN), ('engine', ENG), ('door', DOOR)):
    for n in names:
        if 'File:' + n in by_title: pick[('File:' + n)] = g
# thunder, rain and wind: a handful from the category and from the search
j = api({'action': 'query', 'list': 'categorymembers', 'cmtitle': 'Category:Audio files of thunder', 'cmtype': 'file', 'cmlimit': 40})
th = [m['title'] for m in j['query']['categorymembers'] if m['title'].lower().rsplit('.', 1)[-1] in ('ogg', 'wav', 'flac', 'mp3')][:20]
extra = {}
for t in th:
    jj = api({'action': 'query', 'titles': t, 'prop': 'imageinfo', 'iiprop': 'url|size|extmetadata'}); p = list(jj['query']['pages'].values())[0]['imageinfo'][0]
    lic = (p['extmetadata'].get('LicenseShortName') or {}).get('value', '')
    if p['size'] < 6_000_000 and lic.upper().startswith(('CC0', 'CC BY', 'PUBLIC', 'PD')): extra[t] = {'url': p['url'], 'title': t}; pick[t] = 'thunder'
for q, g in (('rain', 'rain'), ('wind noise', 'wind')):
    for x in res[q][:12]:
        if not re.match(r'^File:(De|En|Fr)[-_ ]', x['title']) and 'XC' not in x['title']: pick[x['title']] = g; extra[x['title']] = x
def fetch(title):
    x = by_title.get(title) or extra.get(title); fn = os.path.join(D, re.sub(r'[^A-Za-z0-9._-]+', '_', title[5:]))
    if not os.path.exists(fn):
        try: open(fn, 'wb').write(urllib.request.urlopen(urllib.request.Request(x['url'], headers=UA), timeout=60).read())
        except Exception: return None
    return fn
def windows(fn, maxw=14):
    try: x, sr = sf.read(fn, dtype='float32')
    except Exception: return []
    x = x.mean(1) if x.ndim > 1 else x
    if len(x) < sr: return []
    x = resample_poly(x, 48000, sr).astype('float32'); n = 144000
    if len(x) < n: x = np.pad(x, (0, n - len(x)))
    ws = [x[i:i + n] for i in range(0, len(x) - n + 1, n // 2)]
    en = np.array([(w ** 2).mean() for w in ws]); o = np.argsort(en)[::-1][:maxw]
    return [ws[i] for i in o]
def flags(w):
    s = I.sound_type(w)['groups']
    return s, s['gun'] >= .35, (s['engine'] >= .5 and s['animal'] < .5)
rows = {}; SC = {}
for t, g in pick.items():
    fn = fetch(t)
    if not fn: continue
    for w in windows(fn):
        s, gun, eng = flags(w); r = rows.setdefault(g, [0, 0, 0, []]); r[0] += 1; r[1] += gun; r[2] += eng; SC.setdefault(g, []).append((s['gun'], s['engine'], s['animal'], t))
# birds: random windows from our recordings
rng = np.random.default_rng(2); import glob
files = glob.glob(BASE + r'\train_audio\*\*.mp3'); rng.shuffle(files); b = [0, 0, 0]
for f in files[:600]:
    for w in windows(f, 2)[:2]:
        s, gun, eng = flags(w); b[0] += 1; b[1] += gun; b[2] += eng; SC.setdefault('birds', []).append((s['gun'], s['engine'], s['animal'], f))
    if b[0] >= 1200: break
rows['birds (Xeno-canto)'] = [b[0], b[1], b[2], []]
print('%-20s %6s  %-14s %-14s' % ('sound', 'windows', 'gun alert', 'engine alert'))
for g, r in rows.items(): print('%-20s %6d  %4d (%4.1f%%)  %4d (%4.1f%%)' % (g, r[0], r[1], r[1] / max(1, r[0]) * 100, r[2], r[2] / max(1, r[0]) * 100))

import pickle; pickle.dump(SC, open(BASE + '/alert_scores.pkl', 'wb'))
print()
for th in (.1, .15, .2, .25, .3, .35, .5):
    print('gun>=%.2f: ' % th + '  '.join('%s %.1f%%' % (g, 100 * np.mean([x[0] >= th for x in v])) for g, v in SC.items()))
for th in (.2, .3, .4, .5):
    print('engine>=%.2f & animal<.5: ' % th + '  '.join('%s %.1f%%' % (g, 100 * np.mean([x[1] >= th and x[2] < .5 for x in v])) for g, v in SC.items()))
print('best gun window per clip:', {g: sorted(set((round(max(x[0] for x in v if x[3] == t), 2)) for t in set(y[3] for y in v))) for g, v in SC.items() if g in ('gun',)})
