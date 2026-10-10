"""Downloads the chosen openly licensed photos (2400 px wide thumbnails from Wikimedia Commons) into photos/ and writes photos/credits.json with author and licence."""
import json, os, re, time, urllib.parse, urllib.request
UA = {'User-Agent': 'WetlandEar-video/1.0 (non-commercial research prototype)'}
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'photos'); os.makedirs(OUT, exist_ok=True)
CHOSEN = {
    'mountains': 'File:Tian Shan in Kyrgyzstan 02.jpg',
    'cranes': 'File:Ländchen Glien02.03.2014 16-59-21.JPG',
    'pelican': 'File:Dalmatian Pelican. Pelecanus crispus (42882199270).jpg',
    'flamingo': 'File:NPBS-15-8206 Greater flamingo Phoenicopterus roseus Pulicat lake.jpg',
    'demoiselle': 'File:Demoiselle crane closeup.jpg',
    'kazakhstan': 'File:Lakes Balkhash and Alakol, southern uplands of Kazakhstan by NASA',
}


def api(**p):
    p['format'] = 'json'
    for wait in (0, 8, 20, 45):
        time.sleep(wait + 2)
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request('https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(p), headers=UA), timeout=40))
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
    raise RuntimeError('rate limited')


def find_title(t):
    """exact title if it exists, otherwise the closest search hit (file names with accents sometimes differ)"""
    j = api(action='query', titles=t, prop='imageinfo', iiprop='url|size|extmetadata|mime', iiurlwidth=2400)
    p = list(j['query']['pages'].values())[0]
    if 'imageinfo' in p: return p
    j = api(action='query', list='search', srsearch=re.sub(r'^File:|\.\w+$', '', t) + ' filetype:bitmap', srnamespace=6, srlimit=3)
    for r in j['query']['search']:
        j2 = api(action='query', titles=r['title'], prop='imageinfo', iiprop='url|size|extmetadata|mime', iiurlwidth=2400)
        p = list(j2['query']['pages'].values())[0]
        if 'imageinfo' in p: return p
    return None


credits = {}
for key, title in CHOSEN.items():
    p = find_title(title)
    if not p: print('NOT FOUND', key, title); continue
    ii = p['imageinfo'][0]; md = ii['extmetadata']; lic = (md.get('LicenseShortName') or {}).get('value', '')
    art = re.sub(r'<[^>]+>', '', (md.get('Artist') or {}).get('value', '')).strip()[:70]
    url = ii.get('thumburl') or ii['url']; fn = os.path.join(OUT, key + '.jpg')
    if not os.path.exists(fn):
        for wait in (0, 10, 30):
            time.sleep(wait + 2)
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read(); open(fn, 'wb').write(data); break
            except Exception as e:
                print('retry', key, e)
    credits[key] = {'title': p['title'][5:], 'author': art, 'license': lic, 'page': ii.get('descriptionurl')}
    print('%-11s %-60s %-10s %s %d KB' % (key, p['title'][5:60], lic, art[:25], os.path.getsize(fn) // 1024 if os.path.exists(fn) else -1))
json.dump(credits, open(os.path.join(OUT, 'credits.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
