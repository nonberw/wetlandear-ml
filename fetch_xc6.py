"""Round 4 download: top every species folder up to TARGET recordings (weak species get more), skip ids already present,
exclude No-Derivatives licences, spread over countries. Key from XC_KEY (never stored)."""
import json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ.get('XC_KEY', '')
if not KEY: sys.exit('set XC_KEY')
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
TA = BASE + r'\train_audio'
UA = {'User-Agent': 'WetlandEar-training/1.0 (research prototype; contact zzhaseka@gmail.com)'}
m = json.load(open(BASE + r'\metrics.json', encoding='utf-8'))
WEAK = {r['sci'] for r in m['per_species'] if r['head'] < 0.5 or r['head'] < r['birdnet'] - 0.02}
SYN = {'Anthropoides virgo': 'Grus virgo', 'Larus ichthyaetus': 'Ichthyaetus ichthyaetus'}
TARGET, TARGET_WEAK = 32, 60
EXTRA_NAMES = ['Aquila clanga', 'Clanga clanga', 'Coloeus monedula', 'Pelecanus crispus']


def api(sp, quality, page=1):
    q = 'sp:"%s" q:">%s" len:3-60' % (sp, quality)
    url = 'https://xeno-canto.org/api/3/recordings?' + urllib.parse.urlencode({'query': q, 'key': KEY, 'page': page})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception: time.sleep(2)
    return {}


def pick(folder):
    sp = folder.replace('_', ' ')
    have = set(os.listdir(TA + '\\' + folder)) if os.path.isdir(TA + '\\' + folder) else set()
    target = TARGET_WEAK if sp in WEAK else TARGET
    need = target - len(have)
    if need <= 0: return folder, []
    recs = {}
    for quality in ('B', 'C', 'D'):
        for page in (1, 2):
            for r in api(SYN.get(sp, sp), quality, page).get('recordings', []):
                if '-nd' in (r.get('lic') or '').lower(): continue
                if 'XC%s.mp3' % r['id'] in have: continue
                recs[r['id']] = r
        if len(recs) >= need * 2: break
    order = sorted(recs.values(), key=lambda r: (r.get('q') or 'Z'))
    seen, out = {}, []
    for r in order:
        c = r.get('cnt')
        if seen.get(c, 0) >= 6: continue
        seen[c] = seen.get(c, 0) + 1; out.append(r)
        if len(out) >= need: break
    d = TA + '\\' + folder; os.makedirs(d, exist_ok=True)
    return folder, [(r, d + '\\XC%s.mp3' % r['id']) for r in out]


def dl(job):
    r, path = job
    if os.path.exists(path) and os.path.getsize(path) > 5000: return 1
    try:
        data = urllib.request.urlopen(urllib.request.Request(r['file'], headers=UA), timeout=60).read()
        if len(data) > 6_000_000 or len(data) < 5000: return 0
        open(path, 'wb').write(data); return 1
    except Exception:
        return 0


if __name__ == '__main__':
    folders = sorted(os.listdir(TA))
    with ThreadPoolExecutor(max_workers=12) as ex:
        jobs = []
        for folder, lst in ex.map(pick, folders):
            jobs += lst
        print('picked', len(jobs), 'new recordings', flush=True)
        res = list(ex.map(dl, jobs))
    print('DOWNLOADED', sum(res), 'of', len(jobs), flush=True)
