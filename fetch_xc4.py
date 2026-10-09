"""Large Xeno-canto download for ~150 species of Kazakhstan wetlands and steppe. Key from XC_KEY (never stored). Skips what exists."""
import json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ.get('XC_KEY', '')
if not KEY: sys.exit('set XC_KEY')
BASE = r'C:\Users\Timur\Documents\wetlandear-models\train_audio'
UA = {'User-Agent': 'WetlandEar-training/1.0 (research prototype; contact zzhaseka@gmail.com)'}
SPECIES = """Cygnus cygnus|Cygnus olor|Cygnus columbianus|Anser anser|Anser albifrons|Anser erythropus|Anser fabalis|Branta ruficollis|Tadorna ferruginea|Tadorna tadorna|Anas platyrhynchos|Anas crecca|Anas acuta|Spatula clypeata|Mareca penelope|Mareca strepera|Spatula querquedula|Aythya ferina|Aythya fuligula|Aythya nyroca|Netta rufina|Bucephala clangula|Mergus merganser|Oxyura leucocephala|Fulica atra|Gallinula chloropus|Rallus aquaticus|Porzana porzana|Podiceps cristatus|Podiceps nigricollis|Podiceps grisegena|Tachybaptus ruficollis|
Pelecanus crispus|Pelecanus onocrotalus|Phoenicopterus roseus|Ardea cinerea|Ardea purpurea|Ardea alba|Egretta garzetta|Nycticorax nycticorax|Botaurus stellaris|Ixobrychus minutus|Plegadis falcinellus|Platalea leucorodia|Grus grus|Grus virgo|Ciconia ciconia|Ciconia nigra|
Ichthyaetus ichthyaetus|Larus cachinnans|Larus argentatus|Chroicocephalus ridibundus|Chroicocephalus genei|Hydrocoloeus minutus|Sterna hirundo|Sternula albifrons|Chlidonias hybrida|Chlidonias leucopterus|Chlidonias niger|Gelochelidon nilotica|Hydroprogne caspia|
Vanellus vanellus|Vanellus leucurus|Vanellus gregarius|Himantopus himantopus|Recurvirostra avosetta|Tringa totanus|Tringa nebularia|Tringa glareola|Tringa erythropus|Tringa stagnatilis|Actitis hypoleucos|Calidris alpina|Calidris minuta|Calidris pugnax|Limosa limosa|Numenius arquata|Gallinago gallinago|Charadrius dubius|Charadrius alexandrinus|Glareola pratincola|Burhinus oedicnemus|
Circus aeruginosus|Circus cyaneus|Haliaeetus albicilla|Buteo rufinus|Aquila nipalensis|Falco tinnunculus|Falco naumanni|Otis tarda|Tetrax tetrax|Chlamydotis macqueenii|
Alauda arvensis|Melanocorypha yeltoniensis|Melanocorypha calandra|Calandrella brachydactyla|Eremophila alpestris|Galerida cristata|Anthus campestris|Motacilla flava|Motacilla alba|Luscinia svecica|Acrocephalus arundinaceus|Acrocephalus scirpaceus|Acrocephalus schoenobaenus|Acrocephalus agricola|Locustella luscinioides|Locustella fluviatilis|Panurus biarmicus|Emberiza schoeniclus|Emberiza melanocephala|Oenanthe isabellina|Oenanthe oenanthe|Lanius minor|Sturnus vulgaris|Pastor roseus|Corvus frugilegus|Corvus cornix|Pica pica|Passer montanus|Hirundo rustica|Riparia riparia|Cuculus canorus|Upupa epops|Merops apiaster|Coracias garrulus""".replace('\n', '').split('|')
PER = 16
names = {}


def api(sp, quality):
    q = 'sp:"%s" q:">%s" len:3-35' % (sp, quality)
    url = 'https://xeno-canto.org/api/3/recordings?' + urllib.parse.urlencode({'query': q, 'key': KEY})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception: time.sleep(2)
    return {}


def pick(sp):
    recs = []
    for quality in ('C', 'D', 'E'):
        recs = [r for r in api(sp, quality).get('recordings', []) if '-nd' not in (r.get('lic') or '').lower()]
        if len(recs) >= 8: break
    recs.sort(key=lambda r: (r.get('q') or 'Z'))
    seen, out = {}, []
    for r in recs:
        c = r.get('cnt')
        if seen.get(c, 0) >= 3: continue
        seen[c] = seen.get(c, 0) + 1; out.append(r)
        if len(out) >= PER: break
    d = os.path.join(BASE, sp.replace(' ', '_')); os.makedirs(d, exist_ok=True)
    if out: names[sp] = out[0].get('en')
    return sp, [(r, os.path.join(d, 'XC%s.mp3' % r['id'])) for r in out]


def dl(job):
    r, path = job
    if os.path.exists(path) and os.path.getsize(path) > 5000: return 1
    try:
        data = urllib.request.urlopen(urllib.request.Request(r['file'], headers=UA), timeout=60).read()
        if len(data) > 4_000_000 or len(data) < 5000: return 0
        open(path, 'wb').write(data); return 1
    except Exception:
        return 0


with ThreadPoolExecutor(max_workers=12) as ex:
    jobs, empty = [], []
    for sp, lst in ex.map(pick, SPECIES):
        jobs += lst
        if not lst: empty.append(sp)
    print('picked', len(jobs), 'recordings for', len(SPECIES) - len(empty), 'species; no data for:', ', '.join(empty), flush=True)
    json.dump(names, open(r'C:\Users\Timur\Documents\wetlandear-models\xc_names.json', 'w', encoding='utf-8'), ensure_ascii=False)
    res = list(ex.map(dl, jobs))
print('DOWNLOADED', sum(res), 'of', len(jobs), flush=True)
