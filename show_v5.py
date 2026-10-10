import json
m = json.load(open(r'C:\Users\Timur\Documents\wetlandear-models\metrics_v5.json', encoding='utf-8'))
rows = m['per_species']
print('species', m['species'], 'recordings', m['recordings'], 'windows', m['windows'], 'noisy', m['noisy_windows'], 'per-species fusion', m.get('fusion_weight_per_species'))
print('window', m['window_accuracy'], 'rec', round(m['recording_accuracy'], 3))
print('coverage clean', [(c['t'], round(c['covers'], 2), round(c['precision'], 3)) for c in m['coverage']])
print('coverage noisy', [(c['t'], round(c['covers'], 2), round(c['precision'], 3)) for c in m['coverage_noisy']])
wins = sum(1 for r in rows if r['head'] > r['birdnet'] + .005); ties = sum(1 for r in rows if abs(r['head'] - r['birdnet']) <= .005); loses = sum(1 for r in rows if r['head'] < r['birdnet'] - .005)
print('better/equal/worse: %d / %d / %d' % (wins, ties, loses))
print('WEAKEST:', '; '.join('%s %.0f%% (%d rec)' % (r['en'], r['head'] * 100, r['recordings']) for r in sorted(rows, key=lambda r: r['head'])[:14]))
print('WORSE THAN BIRDNET:', '; '.join('%s %+.0f' % (r['en'], (r['head'] - r['birdnet']) * 100) for r in sorted(rows, key=lambda r: r['head'] - r['birdnet'])[:10] if r['head'] < r['birdnet']))
print('no RU:', [(r['en'], r['sci']) for r in rows if not r['ru']])
