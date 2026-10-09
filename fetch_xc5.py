"""Round 3 download: ~100 more species of Kazakhstan + extra recordings for the weakest species. Key from XC_KEY (never stored)."""
import json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ.get('XC_KEY', '')
if not KEY: sys.exit('set XC_KEY')
BASE = r'C:\Users\Timur\Documents\wetlandear-models\train_audio'
UA = {'User-Agent': 'WetlandEar-training/1.0 (research prototype; contact zzhaseka@gmail.com)'}
NEW = """Glareola nordmanni|Charadrius leschenaultii|Charadrius mongolus|Calidris ferruginea|Calidris temminckii|Tringa ochropus|Xenus cinereus|Phalaropus lobatus|Limosa lapponica|Numenius phaeopus|Pluvialis squatarola|Pluvialis apricaria|Aythya marila|Mergellus albellus|Mergus serrator|Anser indicus|Ardeola ralloides|Bubulcus ibis|Larus fuscus|Larus canus|Aquila chrysaetos|Aquila heliaca|Aquila clanga|Buteo buteo|Accipiter nisus|Accipiter gentilis|Milvus migrans|Pandion haliaetus|Falco peregrinus|Falco subbuteo|Falco cherrug|Falco vespertinus|Circus macrourus|Circus pygargus|Gyps fulvus|Neophron percnopterus|Athene noctua|Asio otus|Bubo bubo|Otus scops|Emberiza citrinella|Emberiza bruniceps|Emberiza calandra|Carduelis carduelis|Chloris chloris|Linaria cannabina|Fringilla coelebs|Passer domesticus|Curruca communis|Curruca nisoria|Curruca curruca|Phylloscopus collybita|Phylloscopus trochilus|Saxicola rubicola|Saxicola maurus|Oenanthe pleschanka|Oenanthe deserti|Phoenicurus phoenicurus|Luscinia luscinia|Erithacus rubecula|Turdus merula|Turdus pilaris|Turdus philomelos|Parus major|Cyanistes caeruleus|Remiz pendulinus|Lanius collurio|Lanius excubitor|Corvus corax|Corvus monedula|Garrulus glandarius|Oriolus oriolus|Alaudala cheleensis|Calandrella acutirostris|Ammomanes deserti|Streptopelia turtur|Streptopelia decaocto|Columba livia|Columba palumbus|Syrrhaptes paradoxus|Pterocles alchata|Perdix perdix|Coturnix coturnix|Phasianus colchicus|Alectoris chukar|Crex crex|Apus apus|Jynx torquilla|Picus viridis|Dendrocopos major|Motacilla citreola|Motacilla cinerea|Anthus pratensis|Anthus trivialis|Acrocephalus palustris|Hippolais icterina|Iduna caligata|Cettia cetti|Bubo scandiacus""".replace('\n', '').split('|')
WEAK = ['Aquila nipalensis', 'Pelecanus onocrotalus', 'Oxyura leucocephala', 'Ardea purpurea', 'Calidris pugnax', 'Lanius minor', 'Egretta garzetta', 'Larus cachinnans',
        'Tetrax tetrax', 'Aythya nyroca', 'Corvus cornix', 'Hydrocoloeus minutus', 'Melanocorypha yeltoniensis', 'Anser albifrons', 'Fulica atra', 'Podiceps cristatus']
PER_NEW, PER_WEAK = 16, 30
names = {}


def api(sp, quality, maxlen):
    q = 'sp:"%s" q:">%s" len:3-%d' % (sp, quality, maxlen)
    url = 'https://xeno-canto.org/api/3/recordings?' + urllib.parse.urlencode({'query': q, 'key': KEY})
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception: time.sleep(2)
    return {}


def pick(arg):
    sp, per, maxlen = arg
    recs = []
    for quality in ('C', 'D', 'E'):
        recs = [r for r in api(sp, quality, maxlen).get('recordings', []) if '-nd' not in (r.get('lic') or '').lower()]
        if len(recs) >= per // 2: break
    recs.sort(key=lambda r: (r.get('q') or 'Z'))
    d = os.path.join(BASE, sp.replace(' ', '_')); os.makedirs(d, exist_ok=True)
    have = {f for f in os.listdir(d)}
    seen, out = {}, []
    for r in recs:
        c = r.get('cnt')
        if seen.get(c, 0) >= 4: continue
        seen[c] = seen.get(c, 0) + 1; out.append(r)
        if len(out) >= per: break
    if out: names[sp] = out[0].get('en')
    return sp, [(r, os.path.join(d, 'XC%s.mp3' % r['id'])) for r in out]


def dl(job):
    r, path = job
    if os.path.exists(path) and os.path.getsize(path) > 5000: return 1
    try:
        data = urllib.request.urlopen(urllib.request.Request(r['file'], headers=UA), timeout=60).read()
        if len(data) > 5_000_000 or len(data) < 5000: return 0
        open(path, 'wb').write(data); return 1
    except Exception:
        return 0


with ThreadPoolExecutor(max_workers=14) as ex:
    args = [(s, PER_NEW, 35) for s in NEW] + [(s, PER_WEAK, 90) for s in WEAK]
    jobs, empty = [], []
    for sp, lst in ex.map(pick, args):
        jobs += lst
        if not lst: empty.append(sp)
    print('picked', len(jobs), 'recordings; no data for:', ', '.join(empty), flush=True)
    old = {}
    p = r'C:\Users\Timur\Documents\wetlandear-models\xc_names.json'
    if os.path.exists(p): old = json.load(open(p, encoding='utf-8'))
    old.update(names); json.dump(old, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    res = list(ex.map(dl, jobs))
print('DOWNLOADED', sum(res), 'of', len(jobs), flush=True)
