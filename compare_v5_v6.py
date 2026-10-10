import json, statistics as st
a = json.load(open(r'C:\Users\Timur\Documents\wetlandear-models\backup_v5\metrics_v5.json', encoding='utf-8'))
b = json.load(open(r'C:\Users\Timur\Documents\wetlandear-models\metrics_v6.json', encoding='utf-8'))
pc = lambda x: '%.1f%%' % (x * 100)
print('species %d -> %d | recordings %d -> %d | clean windows %d -> %d' % (a['species'], b['species'], a['recordings'], b['recordings'], a['windows'], b['windows']))
print('window acc  ours %s -> %s | per recording %s -> %s' % (pc(a['window_accuracy']['wetlandear']), pc(b['window_accuracy']['wetlandear']), pc(a['recording_accuracy']), pc(b['recording_accuracy'])))
ra, rb = a['robustness']['summary'], b['robustness']['summary']
print('fair BirdNET (restricted) clean %s -> %s ; ours clean %s -> %s ; lead %.1f -> %.1f pp' % (pc(ra['clean']['birdnet_restricted']), pc(rb['clean']['birdnet_restricted']), pc(ra['clean']['clean_plus_noisy_head']), pc(rb['clean']['clean_plus_noisy_head']),
      (ra['clean']['clean_plus_noisy_head'] - ra['clean']['birdnet_restricted']) * 100, (rb['clean']['clean_plus_noisy_head'] - rb['clean']['birdnet_restricted']) * 100))
print('noisy: BirdNET %s -> %s ; ours %s -> %s ; lead %.1f -> %.1f pp' % (pc(ra['noisy']['birdnet_restricted']), pc(rb['noisy']['birdnet_restricted']), pc(ra['noisy']['clean_plus_noisy_head']), pc(rb['noisy']['clean_plus_noisy_head']),
      (ra['noisy']['clean_plus_noisy_head'] - ra['noisy']['birdnet_restricted']) * 100, (rb['noisy']['clean_plus_noisy_head'] - rb['noisy']['birdnet_restricted']) * 100))
print('calibration v6 (conf>=.5):', [(c['t'], pc(c['covers']), pc(c['precision'])) for c in b['coverage']])
print('known-to-BirdNET species:', b.get('known_species'))
A = {r['sci']: r for r in a['per_species']}; B = {r['sci']: r for r in b['per_species']}
both = [s for s in A if s in B]
d = {s: B[s]['head'] - A[s]['head'] for s in both}
print('species in both: %d | mean per-species accuracy %s -> %s' % (len(both), pc(st.mean(A[s]['head'] for s in both)), pc(st.mean(B[s]['head'] for s in both))))
print('better %d, equal(+-1pp) %d, worse %d' % (sum(v > .01 for v in d.values()), sum(abs(v) <= .01 for v in d.values()), sum(v < -.01 for v in d.values())))
new = [s for s in B if s not in A]; print('new species in v6:', [(B[s]['en'], B[s]['recordings'], pc(B[s]['head'])) for s in new])
gained = [s for s in both if B[s]['recordings'] - A[s]['recordings'] >= 5]
print('species with >=5 more recordings: %d | their accuracy %s -> %s' % (len(gained), pc(st.mean(A[s]['head'] for s in gained)), pc(st.mean(B[s]['head'] for s in gained))))
same = [s for s in both if B[s]['recordings'] == A[s]['recordings']]
print('species with unchanged recordings: %d | accuracy %s -> %s' % (len(same), pc(st.mean(A[s]['head'] for s in same)), pc(st.mean(B[s]['head'] for s in same))))
print('weak species (v5 recordings < 15):')
for s in sorted(both, key=lambda s: A[s]['recordings']):
    if A[s]['recordings'] < 15: print('  %-24s rec %2d->%2d  acc %s -> %s' % (A[s]['en'][:24], A[s]['recordings'], B[s]['recordings'], pc(A[s]['head']), pc(B[s]['head'])))
print('largest drops:', [(B[s]['en'][:20], pc(d[s])) for s in sorted(both, key=lambda s: d[s])[:6]])
print('largest gains:', [(B[s]['en'][:20], pc(d[s])) for s in sorted(both, key=lambda s: -d[s])[:6]])
