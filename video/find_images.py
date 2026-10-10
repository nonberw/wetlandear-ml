"""Looks for photos on Wikimedia Commons with licences that allow use in a video (CC0, public domain, CC BY). Share-alike and non-commercial licences are skipped.
Prints candidates; fetch_images.py downloads the chosen ones and writes the credits."""
import json, re, sys, urllib.parse, urllib.request
UA = {'User-Agent': 'WetlandEar-video/1.0 (non-commercial research prototype)'}


def api(**p):
    import time
    p['format'] = 'json'
    for wait in (0, 6, 15, 30, 60):   # Commons answers 429 when asked too fast: back off and retry
        time.sleep(wait + 1.5)
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request('https://commons.wikimedia.org/w/api.php?' + urllib.parse.urlencode(p), headers=UA), timeout=40))
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
    raise RuntimeError('rate limited')


OK = re.compile(r'^(CC0|Public domain|PD|CC BY (1|2|3|4)\.\d)$|^CC BY \d', re.I)
BAD = re.compile(r'SA|NC|ND', re.I)


def search(q, n=24):
    j = api(action='query', list='search', srsearch=q + ' filetype:bitmap', srnamespace=6, srlimit=n)
    titles = [r['title'] for r in j['query']['search']]
    out = []
    for i in range(0, len(titles), 20):
        j = api(action='query', titles='|'.join(titles[i:i + 20]), prop='imageinfo', iiprop='url|size|extmetadata|mime', iiurlwidth=2400)
        for p in j['query']['pages'].values():
            ii = (p.get('imageinfo') or [{}])[0]; md = ii.get('extmetadata', {}); lic = (md.get('LicenseShortName') or {}).get('value', '')
            if not ii.get('url') or ii.get('mime') not in ('image/jpeg', 'image/png'): continue
            if not OK.search(lic.strip()) or BAD.search(lic): continue
            if (ii.get('width') or 0) < 2400 or (ii.get('height') or 0) < 1300: continue
            art = re.sub(r'<[^>]+>', '', (md.get('Artist') or {}).get('value', ''))[:80]
            out.append({'title': p['title'], 'thumb': ii.get('thumburl'), 'w': ii['width'], 'h': ii['height'], 'lic': lic, 'artist': art, 'page': ii.get('descriptionurl')})
    return out


if __name__ == '__main__':
    qs = sys.argv[1:] or ['Dalmatian pelican Pelecanus crispus', 'Greater flamingo flock', 'Demoiselle crane', 'Common crane flock sunrise', 'Tian Shan mountains', 'steppe lake Kazakhstan', 'Korgalzhyn', 'reed bed sunrise lake birds']
    res = {}
    for q in qs:
        try: r = search(q)
        except Exception as e: r = []; print('ERR', q, e)
        res[q] = r
        print('\n##', q, len(r))
        for x in r[:8]: print('  %-70s %4dx%-4d %-10s %s' % (x['title'][5:75], x['w'], x['h'], x['lic'], x['artist'][:30]))
    json.dump(res, open('image_candidates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
