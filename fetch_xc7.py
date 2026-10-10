"""Round 5 download: (a) weak and rare species: take EVERYTHING usable on Xeno-canto (any quality, up to 300 s, no-derivatives excluded);
(b) about 100 more species of Kazakhstan, 24 recordings each. Skips ids already present. Key from XC_KEY (never stored)."""
import json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ.get('XC_KEY', '')
if not KEY: sys.exit('set XC_KEY')
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
TA = BASE + r'\train_audio'
UA = {'User-Agent': 'WetlandEar-training/1.0 (research prototype; contact zzhaseka@gmail.com)'}
RARE = ['Aythya marila', 'Falco cherrug', 'Aquila heliaca', 'Neophron percnopterus', 'Aquila nipalensis', 'Mergellus albellus', 'Pelecanus onocrotalus', 'Pelecanus crispus', 'Charadrius leschenaultii',
        'Otis tarda', 'Aythya nyroca', 'Calidris pugnax', 'Circus macrourus', 'Circus aeruginosus', 'Melanocorypha bimaculata', 'Clanga clanga', 'Coloeus monedula', 'Gyps fulvus', 'Tetrax tetrax',
        'Emberiza bruniceps', 'Emberiza melanocephala', 'Calandrella acutirostris', 'Plegadis falcinellus', 'Ciconia nigra', 'Podiceps grisegena', 'Ixobrychus minutus', 'Oenanthe deserti',
        'Tadorna tadorna', 'Mergus merganser', 'Tringa erythropus', 'Upupa epops', 'Anthus pratensis', 'Corvus cornix', 'Acrocephalus agricola', 'Spatula querquedula']
NEW = """Haliaeetus albicilla|Haliaeetus leucoryphus|Hieraaetus pennatus|Circaetus gallicus|Buteo rufinus|Buteo lagopus|Falco tinnunculus|Falco columbarius|Falco biarmicus|Falco naumanni|Coracias garrulus|Merops apiaster|Merops persicus|Cuculus canorus|Caprimulgus europaeus|Alcedo atthis|Delichon urbicum|Sturnus vulgaris|Pastor roseus|Corvus frugilegus|Panurus biarmicus|Acrocephalus schoenobaenus|Acrocephalus scirpaceus|Acrocephalus dumetorum|Locustella luscinioides|Locustella fluviatilis|Locustella naevia|Sylvia atricapilla|Phylloscopus sibilatrix|Regulus regulus|Muscicapa striata|Ficedula hypoleuca|Luscinia megarhynchos|Phoenicurus ochruros|Oenanthe oenanthe|Oenanthe isabellina|Monticola saxatilis|Turdus atrogularis|Prunella modularis|Motacilla alba|Motacilla flava|Anthus campestris|Emberiza schoeniclus|Emberiza hortulana|Fringilla montifringilla|Spinus spinus|Coccothraustes coccothraustes|Loxia curvirostra|Carpodacus erythrinus|Eremophila alpestris|Galerida cristata|Calandrella brachydactyla|Melanocorypha calandra|Melanocorypha leucoptera|Lullula arborea|Charadrius dubius|Charadrius hiaticula|Anarhynchus alexandrinus|Himantopus himantopus|Recurvirostra avosetta|Tringa totanus|Tringa nebularia|Tringa glareola|Tringa stagnatilis|Actitis hypoleucos|Gallinago gallinago|Scolopax rusticola|Limosa limosa|Numenius arquata|Calidris alpina|Calidris minuta|Chroicocephalus ridibundus|Larus genei|Sterna hirundo|Sternula albifrons|Hydroprogne caspia|Chlidonias niger|Chlidonias leucopterus|Podiceps auritus|Tachybaptus ruficollis|Botaurus stellaris|Nycticorax nycticorax|Ardea cinerea|Ardea alba|Platalea leucorodia|Ciconia ciconia|Anas platyrhynchos|Anas acuta|Spatula clypeata|Mareca strepera|Mareca penelope|Netta rufina|Aythya fuligula|Aythya ferina|Bucephala clangula|Cygnus olor|Cygnus columbianus|Branta ruficollis|Rallus aquaticus|Porzana porzana|Gallinula chloropus|Strix aluco|Asio flammeus|Tetrao urogallus|Lyrurus tetrix|Tetrastes bonasia|Lagopus lagopus|Perdix dauurica|Tetraogallus altaicus|Grus leucogeranus|Anthropoides virgo|Phoenicopterus roseus|Anser fabalis|Pica pica|Corvus corone|Passer montanus|Passer hispaniolensis|Carduelis carduelis""".replace('\n', '').split('|')
RARE_MAX, NEW_MAX = 80, 24


def api(sp, query_extra, page=1):
    q = 'sp:"%s" %s' % (sp, query_extra)
    url = 'https://xeno-canto.org/api/3/recordings?' + urllib.parse.urlencode({'query': q.strip(), 'key': KEY, 'page': page})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception: time.sleep(2)
    return {}


def pick(arg):
    sp, maxn, rare = arg
    folder = sp.replace(' ', '_'); d = TA + '\\' + folder
    have = set(os.listdir(d)) if os.path.isdir(d) else set()
    need = maxn - len(have)
    if need <= 0: return sp, []
    recs = {}
    qs = ['len:3-300'] if rare else ['q:">C" len:3-60', 'q:">D" len:3-90', 'len:3-120']
    for qx in qs:
        for page in (1, 2, 3):
            j = api(sp, qx, page)
            for r in j.get('recordings', []):
                if '-nd' in (r.get('lic') or '').lower() or 'XC%s.mp3' % r['id'] in have: continue
                recs[r['id']] = r
            if page >= int(j.get('numPages', 1) or 1): break
        if len(recs) >= need: break
    order = sorted(recs.values(), key=lambda r: (r.get('q') or 'Z'))
    seen, out = {}, []
    for r in order:
        c = r.get('cnt')
        if not rare and seen.get(c, 0) >= 5: continue
        seen[c] = seen.get(c, 0) + 1; out.append(r)
        if len(out) >= need: break
    if out: os.makedirs(d, exist_ok=True)
    return sp, [(r, d + '\\XC%s.mp3' % r['id']) for r in out]


def dl(job):
    r, path = job
    if os.path.exists(path) and os.path.getsize(path) > 5000: return 1
    try:
        data = urllib.request.urlopen(urllib.request.Request(r['file'], headers=UA), timeout=90).read()
        if len(data) > 12_000_000 or len(data) < 5000: return 0
        open(path, 'wb').write(data); return 1
    except Exception:
        return 0


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=12) as ex:
        jobs, empty = [], []
        for sp, lst in ex.map(pick, [(s, RARE_MAX, True) for s in RARE] + [(s, NEW_MAX, False) for s in NEW]):
            jobs += lst
            if not lst and not os.path.isdir(TA + '\\' + sp.replace(' ', '_')): empty.append(sp)
        print('picked', len(jobs), 'new recordings; species with nothing found:', ', '.join(empty), flush=True)
        res = list(ex.map(dl, jobs))
    print('DOWNLOADED', sum(res), 'of', len(jobs), flush=True)
