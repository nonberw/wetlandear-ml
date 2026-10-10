"""Create train_v5.py from train_v4.py: per-species weight of BirdNET's own score in the fusion (chosen on out-of-fold probabilities,
checked on recordings that were not used for choosing), and a vector weight in the exported head."""
s = open(r'C:\Users\Timur\Documents\wetlandear-models\train_v4.py', encoding='utf-8').read()

# 1. fuse with a weight vector
s = s.replace("""def fuse(P, T=1.0):
    z = (np.log(np.clip(P, 1e-9, 1)) + W * raw) / T; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)""",
"""WV = np.full(len(classes), W, dtype='float32')


def fuse(P, T=1.0, wv=None):
    wv = WV if wv is None else wv
    z = (np.log(np.clip(P, 1e-9, 1)) + wv[None, :] * raw) / T; z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)""", 1)
s = s.replace("    z0 = np.log(np.clip(P, 1e-9, 1)) + W * raw\n", "    z0 = np.log(np.clip(P, 1e-9, 1)) + WV[None, :] * raw\n", 1)

# 2. tune per-class weights after the first temperature fit
a = "TA_, TB_ = best_T(PA), best_T(PB)\n"
assert a in s
tune = '''TA_, TB_ = best_T(PA), best_T(PB)
CAND = (0.15, 0.4, 0.8, 1.5)


def tune_weights(P, T, rows_mask, passes=2):
    """Coordinate ascent: for each species with a BirdNET label pick the fusion weight that gives the best accuracy on rows_mask."""
    wv = np.full(len(classes), W, dtype='float32')
    lp = np.log(np.clip(P[rows_mask], 1e-9, 1)); rw = raw[rows_mask]; y = yi[rows_mask]; sl = sel[rows_mask]
    def acc_of(w): return float((np.argmax(lp + w[None, :] * rw, 1)[sl] == y[sl]).mean())
    best = acc_of(wv)
    for _ in range(passes):
        for c in np.where(in_bn)[0]:
            keep = wv[c]; bc = keep
            for cand in CAND:
                if cand == keep: continue
                wv[c] = cand; a_ = acc_of(wv)
                if a_ > best + 2e-5: best, bc = a_, cand
            wv[c] = bc
    return wv, best


# honest check: choose the weights on half of the recordings, test on the other half
rng = np.random.default_rng(7); ug = np.array(sorted(set(GA))); half = set(rng.choice(ug, len(ug) // 2, replace=False))
inA = np.array([g in half for g in GA]); inB = ~inA
base_B = float((np.argmax(np.log(np.clip(PB[inB], 1e-9, 1)) + WV[None, :] * raw[inB], 1)[sel[inB]] == yi[inB][sel[inB]]).mean())
wvA, accA = tune_weights(PB, TB_, inA)
tuned_B = float((np.argmax(np.log(np.clip(PB[inB], 1e-9, 1)) + wvA[None, :] * raw[inB], 1)[sel[inB]] == yi[inB][sel[inB]]).mean())
print('per-species BirdNET weights: held-out recordings %.2f%% -> %.2f%% (chosen on the other half: %.2f%%)' % (base_B * 100, tuned_B * 100, accA * 100), flush=True)
if tuned_B > base_B + 0.0005:
    WV, acc_all = tune_weights(PB, TB_, np.ones(len(YA), bool)); TB_ = best_T(PB)
    print('using per-species weights (all recordings): in-sample %.2f%%, temperature %.2f' % (acc_all * 100, TB_), flush=True)
else:
    print('per-species weights do not help on held-out recordings, keeping one weight', flush=True)
'''
s = s.replace(a, tune, 1)

# 3. export the vector
s = s.replace("T=np.float32(TB_), w=np.float32(W), missing=np.float32(0.0)", "T=np.float32(TB_), w=WV.astype('float32'), missing=np.float32(0.0)", 1)
s = s.replace("'temperature': TB_, 'fusion_weight': W,", "'temperature': TB_, 'fusion_weight': W, 'fusion_weight_per_species': bool(WV.std() > 0),", 1)
s = s.replace("'version': 'v4'", "'version': 'v5'").replace("metrics_v4.json", "metrics_v5.json")
open(r'C:\Users\Timur\Documents\wetlandear-models\train_v5.py', 'w', encoding='utf-8').write(s)
print('train_v5.py written')
