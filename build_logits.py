"""Add BirdNET's own scores for OUR species (restricted logits) to every cached window, so we can stack/fuse. Same window selection as build_feats2."""
import glob, os, sys, warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings('ignore')
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
_ctx = {}


def classes_list():
    out = []
    for d in sorted(glob.glob(BASE + r'\train_audio\*')):
        n = os.path.basename(d).replace('_', ' '); out.append(SYN.get(n, n))
    return out


def init():
    sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api')
    import identify as I
    s = I._load(); lab = [l.split('_')[0] for l in s['labels']]
    cls = classes_list(); idx = [lab.index(c) if c in lab else -1 for c in cls]
    _ctx.update(I=I, s=s, idx=np.array(idx))


def do_folder(d):
    import soundfile as sf
    from scipy.signal import resample_poly
    name = os.path.basename(d); cache = BASE + '\\feats2\\' + name + '.npz'
    out_f = BASE + '\\feats3\\' + name + '.npy'
    if os.path.exists(out_f): return name, 'cached'
    if not _ctx: init()
    I, s, idx = _ctx['I'], _ctx['s'], _ctx['idx']
    files = sorted(f for f in glob.glob(d + r'\*') if os.path.getsize(f) > 5000)
    rows = []
    for f in files:
        try:
            x, sr = sf.read(f, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
            x = resample_poly(x, 48000, sr).astype('float32')
        except Exception:
            continue
        n = 144000
        if len(x) < n: x = np.pad(x, (0, n - len(x)))
        ws = [x[i:i + n] for i in range(0, len(x) - n + 1, n // 2)]
        en = np.array([float((w ** 2).mean()) for w in ws]); order = np.argsort(en)[::-1]
        strong = list(order[:min(14, len(ws))]); weak = list(order[-3:]) if len(ws) > 8 else []
        for i in strong + weak:
            L = I._run(s['clf'], ws[i].reshape(1, -1)).copy()
            v = np.full(len(idx), -15.0, dtype='float32'); ok = idx >= 0; v[ok] = np.clip(L[idx[ok]], -15, 15)
            rows.append(v)
    os.makedirs(BASE + r'\feats3', exist_ok=True)
    np.save(out_f, np.array(rows, dtype='float16'))
    return name, 'built %d' % len(rows)


if __name__ == '__main__':
    os.makedirs(BASE + r'\feats3', exist_ok=True)
    with Pool(int(os.environ.get('NPROC', '3'))) as pool:
        for name, st in pool.imap_unordered(do_folder, sorted(glob.glob(BASE + r'\train_audio\*'))):
            print(name, st, flush=True)
    print('ALLDONE', flush=True)
