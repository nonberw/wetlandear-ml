"""Write js/species-sci.js (English common name -> scientific name) for the app from the exported head (models/head.npz).
Used for the species card and for Darwin Core / eBird exports. Run after every head export."""
import json
import numpy as np
APP = r'C:\Users\Timur\Documents\wetlandear-app'
z = np.load(APP + r'\models\head.npz', allow_pickle=True)
m = {}
for sci, com in zip(z['classes'], z['common']):
    if str(sci) != 'background' and str(com):
        m[str(com)] = str(sci)
# the reference list of the app uses its own English names
m.update({'Greater Flamingo': 'Phoenicopterus roseus', 'Dalmatian Pelican': 'Pelecanus crispus', 'Great White Pelican': 'Pelecanus onocrotalus', 'Demoiselle Crane': 'Anthropoides virgo',
          'Common Crane': 'Grus grus', 'Whooper Swan': 'Cygnus cygnus', 'Greylag Goose': 'Anser anser', 'Lesser White-fronted Goose': 'Anser erythropus', 'Ruddy Shelduck': 'Tadorna ferruginea',
          "Pallas's Gull": 'Larus ichthyaetus', 'Whiskered Tern': 'Chlidonias hybrida', 'Great Crested Grebe': 'Podiceps cristatus', 'Eurasian Coot': 'Fulica atra', 'Northern Lapwing': 'Vanellus vanellus',
          'Great Reed Warbler': 'Acrocephalus arundinaceus', 'Bluethroat': 'Luscinia svecica', 'Eurasian Skylark': 'Alauda arvensis', 'Black Lark': 'Melanocorypha yeltoniensis'})
open(APP + r'\js\species-sci.js', 'w', encoding='utf-8').write('/* English common name -> scientific name, for the species card and data exports. Generated from models/head.npz. */\nwindow.WE_SCI = ' + json.dumps(dict(sorted(m.items())), ensure_ascii=False, indent=0) + ';\n')
print(len(m), 'names')
