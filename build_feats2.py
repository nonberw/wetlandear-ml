"""Feature builder v2: per window BirdNET fingerprint (1024) + YAMNet fingerprint (1024, mean over 3 sub-windows) + BirdNET top species.
Parallel over species folders; per-folder cache in feats2/ so it can be re-run while downloads continue."""
import glob, hashlib, os, sys, warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings('ignore')

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
OUT = BASE + r'\feats2'
os.makedirs(OUT, exist_ok=True)
SYN = {'Grus virgo': 'Anthropoides virgo', 'Ichthyaetus ichthyaetus': 'Larus ichthyaetus'}
_ctx = {}


def init():
    sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\api')
    sys.path.insert(0, r'C:\Users\Timur\Documents\wetlandear-app\lib')
    import identify as I
    from ai_edge_litert.interpreter import Interpreter
    s = I._load()
    y = Interpreter(model_path=BASE + r'\yamnet\yamnet.tflite', experimental_preserve_all_tensors=True); y.allocate_tensors()
    _ctx.update(I=I, s=s, y=y, lab=np.array([l.split('_')[0] for l in s['labels']]),
                plaus=I._run(s['meta'], np.array([[50.6, 69.0, 1.0]], dtype='float32')) > 0.03)


def yam(y, seg):
    i = y.get_input_details()[0]; o = y.get_output_details()[0]
    y.set_tensor(i['index'], seg.astype('float32')); y.invoke()
    return y.get_tensor(o['index'] - 5).reshape(-1).copy()


def one_file(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    I, s, y, lab, plaus = _ctx['I'], _ctx['s'], _ctx['y'], _ctx['lab'], _ctx['plaus']
    x, sr = sf.read(path, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    x = resample_poly(x, 48000, sr).astype('float32')
    n = 144000
    if len(x) < n: x = np.pad(x, (0, n - len(x)))
    ws = [x[i:i + n] for i in range(0, len(x) - n + 1, n // 2)]
    en = np.array([float((w ** 2).mean()) for w in ws]); order = np.argsort(en)[::-1]
    strong = list(order[:min(14, len(ws))]); weak = list(order[-3:]) if len(ws) > 8 else []
    clf = s['clf']; oidx = clf.get_output_details()[0]['index']
    out = []
    for idx, bg in [(i, 0) for i in strong] + [(i, 1) for i in weak]:
        w = ws[idx]
        L = I._run(clf, w.reshape(1, -1)).copy(); eb = clf.get_tensor(oidx - 1)[0].copy()
        P = 1 / (1 + np.exp(-np.clip(L, -15, 15))); P = np.where(plaus, P, 0); k = int(np.argmax(P))
        x16 = w.reshape(-1, 3).mean(axis=1)
        ey = np.mean([yam(y, x16[c - 7800:c + 7800]) for c in (9600, 24000, 38400)], axis=0)
        out.append((eb / np.linalg.norm(eb), ey / (np.linalg.norm(ey) + 1e-9), lab[k], float(P[k]), bg))
    return out


def do_folder(d):
    name = os.path.basename(d)
    files = sorted(f for f in glob.glob(d + r'\*') if os.path.getsize(f) > 5000)
    cache = OUT + '\\' + name + '.npz'
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if int(z['nfiles']) == len(files):
            return name, len(files), 'cached'
    if not _ctx: init()
    sp = SYN.get(name.replace('_', ' '), name.replace('_', ' '))
    XB, XY, Y, G, BS, BC = [], [], [], [], [], []
    for f in files:
        try: wins = one_file(f)
        except Exception: continue
        gid = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
        for eb, ey, bs, bc, bg in wins:
            XB.append(eb); XY.append(ey); Y.append('background' if bg else sp); G.append(gid); BS.append(bs); BC.append(bc)
    np.savez_compressed(cache, XB=np.array(XB, dtype='float32'), XY=np.array(XY, dtype='float32'), Y=np.array(Y), G=np.array(G), BS=np.array(BS), BC=np.array(BC, dtype='float32'), nfiles=len(files))
    return name, len(files), 'built %d windows' % len(Y)


if __name__ == '__main__':
    dirs = sorted(glob.glob(BASE + r'\train_audio\*'))
    extra = [BASE + r'\samples2', BASE + r'\samples']   # old loose files are folded in via train_audio only; skip
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for name, nf, status in pool.imap_unordered(do_folder, dirs):
            print('%-30s %3d files  %s' % (name, nf, status), flush=True)
    print('ALLDONE', flush=True)
