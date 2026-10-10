import json, shutil
APP = r'C:\Users\Timur\Documents\wetlandear-app'
SITE = r'C:\Users\Timur\Documents\wetlandear'
ML = r'C:\Users\Timur\Documents\wetlandear-ml'
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
RU = {'Wood Warbler': 'Пеночка-трещотка', 'Common Grasshopper-Warbler': 'Пёстрый сверчок', 'Eurasian Nightjar': 'Обыкновенный козодой', 'Goldcrest': 'Желтоголовый королёк', 'Coloeus monedula': 'Галка',
      'Common House-Martin': 'Воронок', 'Tawny Owl': 'Серая неясыть', 'Blue-cheeked Bee-eater': 'Персидская щурка', 'Eurasian Siskin': 'Чиж', 'Wood Lark': 'Лесной жаворонок', 'Red Crossbill': 'Клёст-еловик',
      'Melanocorypha bimaculata': 'Двупятнистый жаворонок', 'Haliaeetus leucoryphus': 'Орлан-долгохвост', 'Common Ringed Plover': 'Галстучник', 'European Pied Flycatcher': 'Мухоловка-пеструшка',
      'Eurasian Woodcock': 'Вальдшнеп', 'Common Nightingale': 'Обыкновенный соловей', 'Common Kingfisher': 'Обыкновенный зимородок', 'Dunnock': 'Лесная завирушка', 'Booted Eagle': 'Орёл-карлик',
      'Brambling': 'Юрок', 'Black Redstart': 'Горихвостка-чернушка', 'Western Capercaillie': 'Обыкновенный глухарь', 'Eurasian Blackcap': 'Черноголовая славка', 'Merlin': 'Дербник', 'Spotted Flycatcher': 'Серая мухоловка',
      'Black Grouse': 'Тетерев', 'Horned Grebe': 'Красношейная поганка', 'Hawfinch': 'Обыкновенный дубонос', 'Common Rosefinch': 'Обыкновенная чечевица', 'Willow Ptarmigan': 'Белая куропатка', 'Spanish Sparrow': 'Испанский воробей',
      'Tetraogallus altaicus': 'Алтайский улар', 'Ortolan Bunting': 'Садовая овсянка', 'Black-throated Thrush': 'Чернозобый дрозд', 'Rufous-tailed Rock-Thrush': 'Каменный дрозд', 'Rough-legged Hawk': 'Зимняк',
      "Blyth's Reed Warbler": 'Камышевка Блита', 'Short-toed Snake-Eagle': 'Змееяд', 'Hazel Grouse': 'Рябчик', 'Short-eared Owl': 'Болотная сова', 'Greater Spotted Eagle': 'Большой подорлик', 'Carrion Crow': 'Чёрная ворона',
      'Falco biarmicus': 'Ланнер', 'Pelecanus crispus': 'Кудрявый пеликан', 'Perdix dauurica': 'Бородатая куропатка'}
EN_FIX = {'Coloeus monedula': 'Western Jackdaw', 'Melanocorypha bimaculata': 'Bimaculated Lark', 'Haliaeetus leucoryphus': "Pallas's Fish Eagle", 'Falco biarmicus': 'Lanner Falcon', 'Pelecanus crispus': 'Dalmatian Pelican',
          'Perdix dauurica': 'Daurian Partridge', 'Tetraogallus altaicus': 'Altai Snowcock'}
p = APP + r'\js\species-ru.js'
js = open(p, encoding='utf-8').read(); cur = json.loads(js[js.index('{'):js.rindex('}') + 1]); cur.update(RU)
open(p, 'w', encoding='utf-8').write('/* Russian names for species the app can suggest. Key: English common name (as used by BirdNET). */\nwindow.WE_RU = ' + json.dumps(cur, ensure_ascii=False, indent=0) + ';\n')
m = json.load(open(BASE + r'\metrics_v5.json', encoding='utf-8'))
for r in m['per_species']:
    if r['sci'] == 'Larus ichthyaetus': r['en'] = "Pallas's Gull"; r['ru'] = 'Черноголовый хохотун'
    if r['en'] in EN_FIX: r['en'] = EN_FIX[r['en']]
    if not r['ru']: r['ru'] = RU.get(r['sci']) or RU.get(r['en']) or cur.get(r['en'], '')
m['fusion_weight_per_species'] = False
json.dump(m, open(BASE + r'\metrics_v5.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
shutil.copy(BASE + r'\metrics_v5.json', SITE + r'\data\metrics.json')
for src, dst in [('chain6_train.log', 'v5_train_and_robustness.log'), ('metrics_v5.json', 'v5_metrics.json')]:
    shutil.copy(BASE + '\\' + src, ML + r'\reports\\' + dst)
for f in ['train_v5.py', 'make_v5.py', 'fetch_xc7.py', 'fix_feats3.py', 'final_chain6.sh', 'make_negatives.py', 'sim_voices.py', 'sim_voices2.py', 'ship_v5.py', 'test_worker.js']:
    shutil.copy(BASE + '\\' + f, ML + '\\' + f)
print(len(cur), 'ru names;', sum(1 for r in m['per_species'] if not r['ru']), 'missing in metrics')
