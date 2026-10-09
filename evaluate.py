"""Honest check: real CC recordings from Wikimedia Commons, run through the same BirdNET + location filter as the app."""
import json, os, sys, urllib.parse, urllib.request
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api')
import identify as I

UA = {'User-Agent': 'WetlandEar-eval/1.0 (research prototype; contact zzhaseka@gmail.com)'}
OUT = r'C:\Users\Timur\Documents\wetlandear-models\samples2'
os.makedirs(OUT, exist_ok=True)

SPECIES = [  # (latin, english)
    ('Pelecanus onocrotalus', 'Great White Pelican'), ('Pelecanus crispus', 'Dalmatian Pelican'), ('Phoenicopterus roseus', 'Greater Flamingo'),
    ('Grus grus', 'Common Crane'), ('Grus virgo', 'Demoiselle Crane'), ('Anser anser', 'Graylag Goose'), ('Cygnus cygnus', 'Whooper Swan'),
    ('Fulica atra', 'Eurasian Coot'), ('Podiceps cristatus', 'Great Crested Grebe'), ('Tadorna ferruginea', 'Ruddy Shelduck'),
    ('Acrocephalus arundinaceus', 'Great Reed Warbler'), ('Chlidonias hybrida', 'Whiskered Tern'), ('Vanellus vanellus', 'Northern Lapwing'),
    ('Alauda arvensis', 'Eurasian Skylark'), ('Larus ichthyaetus', 'Pallas\'s Gull'), ('Luscinia svecica', 'Bluethroat'),
]


def api(params):
    url = 'https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(params)
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


def find_files(latin, n=2):
    r = api({'action': 'query', 'list': 'search', 'srsearch': 'filetype:audio ' + latin, 'srnamespace': 6, 'srlimit': 8, 'format': 'json'})
    titles = [x['title'] for x in r['query']['search'] if latin.split()[0].lower() in x['title'].lower() or latin.split()[1].lower() in x['title'].lower()]
    out = []
    for t in titles:
        info = api({'action': 'query', 'titles': t, 'prop': 'imageinfo', 'iiprop': 'url|size|extmetadata', 'format': 'json'})
        for p in info['query']['pages'].values():
            ii = (p.get('imageinfo') or [{}])[0]
            if ii.get('url') and ii.get('size', 0) < 8_000_000:
                out.append((t, ii['url']))
        if len(out) >= n:
            break
    return out


def windows(x48, k=3):
    n = 144000
    if len(x48) < n:
        x48 = np.pad(x48, (0, n - len(x48)))
    ws = [x48[i:i + n] for i in range(0, len(x48) - n + 1, n // 2)]
    en = [float((w ** 2).mean()) for w in ws]
    idx = np.argsort(en)[::-1][:k]
    return [ws[i] for i in idx]


results = []
for latin, eng in SPECIES:
    try:
        files = find_files(latin)
    except Exception as e:
        files = []
    if not files:
        results.append((latin, eng, 0, None, None, 'no recording found'))
        continue
    for title, url in files:
        path = os.path.join(OUT, latin.replace(' ', '_') + '_' + str(abs(hash(url)) % 10000) + os.path.splitext(url)[1])
        try:
            if not os.path.exists(path):
                open(path, 'wb').write(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
            x, sr = sf.read(path, dtype='float32')
            x = x.mean(1) if x.ndim > 1 else x
            x48 = resample_poly(x, 48000, sr).astype('float32')
            wins = windows(x48)
            res = I.identify(wins, 50.6, 69.0)
            tops = [(r['species'][0]['sci'], r['species'][0]['common'], r['species'][0]['conf']) if r['species'] else (None, None, 0) for r in res]
            hit = sum(1 for t in tops if t[0] == latin)
            results.append((latin, eng, len(wins), hit, tops, title))
        except Exception as e:
            results.append((latin, eng, 0, None, None, 'error ' + str(e)[:60]))

print('\n=== RESULT (top-1 of the strongest windows, location filter ON) ===')
tot = ok = 0
for latin, eng, n, hit, tops, title in results:
    if hit is None:
        print(f'{eng:22s} {latin:28s} -  {title}')
        continue
    tot += n; ok += hit
    print(f'{eng:22s} {latin:28s} {hit}/{n}  top1: ' + '; '.join(f'{c} {p:.2f}' if c else 'nothing' for _, c, p in tops))
print(f'\nTOTAL windows right: {ok}/{tot}')
