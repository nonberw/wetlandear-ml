"""Noise-robustness data: for every recording take its 6 strongest 3-s windows and make 2 noisy copies each (reed rustle, wind, water, rain,
real quiet field background from other recordings, other birds singing at the same time, or a mix) at a random signal-to-noise ratio.
Stores BirdNET fingerprint + BirdNET scores for our species, so the head can be trained and tested on dirty sound.
Usage: python build_noisy.py bank   (once)   |   python build_noisy.py 14   (processes)"""
import glob, hashlib, os, sys, warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings('ignore')
BASE = r'C:\Users\Timur\Documents\wetlandear-models'
OUT = BASE + r'\feats_noisy'
BANK = BASE + r'\noise_bank.npz'
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
N = 144000
SR = 48000
_ctx = {}


def load48(f):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(f, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    x = resample_poly(x, SR, sr).astype('float32')
    if len(x) < N: x = np.pad(x, (0, N - len(x)))
    return x


def windows(x):
    ws = [x[i:i + N] for i in range(0, len(x) - N + 1, N // 2)]
    en = np.array([float((w ** 2).mean()) for w in ws])
    return ws, en


def make_bank(seed=1):
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(BASE + r'\train_audio\*\*.mp3')); rng.shuffle(files)
    quiet, loud = [], []
    for f in files:
        if len(quiet) >= 350 and len(loud) >= 250: break
        try: ws, en = windows(load48(f))
        except Exception: continue
        o = np.argsort(en)
        if len(quiet) < 350 and len(ws) > 6: quiet.append(ws[o[0]] / (np.sqrt((ws[o[0]] ** 2).mean()) + 1e-9))
        if len(loud) < 250: loud.append(ws[o[-1]] / (np.sqrt((ws[o[-1]] ** 2).mean()) + 1e-9))
    np.savez_compressed(BANK, quiet=np.array(quiet, dtype='float16'), loud=np.array(loud, dtype='float16'))
    print('bank', len(quiet), len(loud))


def rms1(x): return x / (np.sqrt((x ** 2).mean()) + 1e-9)


def gust(rng, hz, depth=0.8):
    from scipy.signal import butter, sosfilt
    e = sosfilt(butter(2, hz, fs=SR, output='sos'), rng.standard_normal(N + 20000))[20000:]
    e = (e - e.min()) / (e.max() - e.min() + 1e-9)
    return (1 - depth) + depth * e


def synth(kind, rng):
    from scipy.signal import butter, sosfilt
    w = rng.standard_normal(N)
    if kind == 'reeds':        # dry leaves/reed rustle: band-limited noise with crackle and slow gusts
        y = sosfilt(butter(4, [1800, 11000], 'band', fs=SR, output='sos'), w) * gust(rng, 1.5)
        imp = (rng.random(N) < 0.0015) * rng.standard_normal(N) * 6
        y = y + sosfilt(butter(2, [2500, 9000], 'band', fs=SR, output='sos'), imp)
    elif kind == 'wind':       # low rumble with gusts, a little hiss
        y = sosfilt(butter(4, 450, 'low', fs=SR, output='sos'), w) * gust(rng, 0.6, 0.9) + 0.05 * sosfilt(butter(2, 3000, 'low', fs=SR, output='sos'), rng.standard_normal(N))
    elif kind == 'water':      # pink-ish broadband, ripples
        f = np.fft.rfft(w); fr = np.arange(len(f)) + 1.0; y = np.fft.irfft(f / np.sqrt(fr), N) * gust(rng, 20, 0.4)
    else:                      # rain: bright noise with drops
        y = sosfilt(butter(3, [900, 14000], 'band', fs=SR, output='sos'), w) * (0.6 + 0.4 * gust(rng, 8, 0.7))
        y = y + sosfilt(butter(2, [3000, 12000], 'band', fs=SR, output='sos'), (rng.random(N) < 0.004) * rng.standard_normal(N) * 8)
    return rms1(y.astype('float32'))


def noise_of(kind, rng):
    bank = _ctx['bank']
    if kind == 'bg':
        q = bank['quiet']; n = rms1(q[rng.integers(len(q))].astype('float32'))
        return n if rng.random() < .5 else np.roll(n, int(rng.integers(N)))
    if kind == 'chorus':
        l = bank['loud']; return rms1(l[rng.integers(len(l))].astype('float32'))
    if kind == 'mix':
        a, b = rng.choice(['reeds', 'wind', 'water', 'rain', 'bg'], 2, replace=False)
        return rms1(noise_of(a, rng) + noise_of(b, rng) * float(rng.uniform(.4, 1)))
    return synth(kind, rng)


def init():
    sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api')
    import identify as I
    s = I._load(); lab = [l.split('_')[0] for l in s['labels']]
    cls = []
    for d in sorted(glob.glob(BASE + r'\train_audio\*')):
        n = os.path.basename(d).replace('_', ' '); cls.append(SYN.get(n, n))
    idx = np.array([lab.index(c) if c in lab else -1 for c in cls])
    z = np.load(BANK)
    _ctx.update(I=I, s=s, idx=idx, bank={'quiet': z['quiet'], 'loud': z['loud']})


KINDS = ['reeds', 'wind', 'water', 'rain', 'bg', 'chorus', 'mix']


def do_folder(d):
    name = os.path.basename(d)
    files = sorted(f for f in glob.glob(d + r'\*') if os.path.getsize(f) > 5000)
    cache = OUT + '\\' + name + '.npz'
    if os.path.exists(cache) and int(np.load(cache, allow_pickle=True)['nfiles']) == len(files):
        return name, 'cached'
    if not _ctx: init()
    I, s, idx = _ctx['I'], _ctx['s'], _ctx['idx']
    sp = SYN.get(name.replace('_', ' '), name.replace('_', ' '))
    clf = s['clf']; oidx = clf.get_output_details()[0]['index']
    XB, LB, Y, G, T, S = [], [], [], [], [], []
    for f in files:
        try: ws, en = windows(load48(f))
        except Exception: continue
        gid = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(gid)
        for i in np.argsort(en)[::-1][:6]:
            w = ws[i]; ps = float((w ** 2).mean())
            for kind in rng.choice(KINDS, 2, replace=False):
                snr = float(rng.uniform(3, 15) if kind == 'chorus' else rng.uniform(-3, 12))
                nz = noise_of(str(kind), rng) * np.sqrt(ps / (10 ** (snr / 10)))
                mix = (w + nz).astype('float32'); pk = np.abs(mix).max()
                if pk > 1: mix = mix / pk
                L = I._run(clf, mix.reshape(1, -1)).copy(); eb = clf.get_tensor(oidx - 1)[0].copy()
                v = np.full(len(idx), -15.0, dtype='float32'); ok = idx >= 0; v[ok] = np.clip(L[idx[ok]], -15, 15)
                XB.append(eb / np.linalg.norm(eb)); LB.append(v); Y.append(sp); G.append(gid); T.append(str(kind)); S.append(snr)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(cache, XB=np.array(XB, dtype='float32'), LB=np.array(LB, dtype='float16'), Y=np.array(Y), G=np.array(G), T=np.array(T), S=np.array(S, dtype='float32'), nfiles=len(files))
    return name, 'built %d' % len(Y)


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        make_bank(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    with Pool(int(sys.argv[1])) as pool:
        for name, st in pool.imap_unordered(do_folder, sorted(glob.glob(BASE + r'\train_audio\*'))):
            print(name, st, flush=True)
    print('ALLDONE', flush=True)
