"""Add open-licence (CC0, CC BY, CC BY-SA) sound observations from iNaturalist for species that have few recordings. Files go to train_audio/<Genus_species>/INAT<id>.<ext>."""
import json, os, sys, time, urllib.parse, urllib.request
import soundfile as sf
BASE = r'C:\Users\Timur\Documents\wetlandear-models'; TA = BASE + r'\train_audio'
UA = {'User-Agent': 'WetlandEar-research/1.0 (zzhaseka@gmail.com)'}
OK = ('cc0', 'cc-by', 'cc-by-sa')
def get(u):
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40))
        except Exception: time.sleep(2)
    return {}
tot = {}
for d in sorted(os.listdir(TA)):
    n = len([f for f in os.listdir(os.path.join(TA, d)) if os.path.getsize(os.path.join(TA, d, f)) > 5000])
    if n >= 40: continue
    sp = d.replace('_', ' ')
    t = get('https://api.inaturalist.org/v1/taxa?' + urllib.parse.urlencode({'q': sp, 'rank': 'species', 'per_page': 3})).get('results', [])
    t = [x for x in t if x['name'].lower() == sp.lower()] or t[:1]
    if not t: continue
    o = get('https://api.inaturalist.org/v1/observations?' + urllib.parse.urlencode({'taxon_id': t[0]['id'], 'sounds': 'true', 'per_page': 100, 'quality_grade': 'research,needs_id'})).get('results', [])
    got = 0
    for r in o:
        for so in r.get('sounds', []):
            if (so.get('license_code') or '') not in OK or not so.get('file_url'): continue
            ext = so['file_url'].split('?')[0].rsplit('.', 1)[-1].lower()
            fn = os.path.join(TA, d, 'INAT%d.%s' % (so['id'], ext))
            if os.path.exists(fn): continue
            try:
                open(fn, 'wb').write(urllib.request.urlopen(urllib.request.Request(so['file_url'], headers=UA), timeout=60).read())
                x, sr = sf.read(fn, dtype='float32')   # must be readable by the feature builder
                if len(x) < sr: raise ValueError('short')
                got += 1
            except Exception:
                try: os.remove(fn)
                except Exception: pass
    if got: tot[d] = got; print('%-34s had %2d, +%d' % (d, n, got), flush=True)
print('added', sum(tot.values()), 'files for', len(tot), 'species')
