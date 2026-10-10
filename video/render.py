"""Intro video for WetlandEar, drawn with code: a dawn landscape with mountains and flying birds, a listening scene with a spectrogram, the real site and app
(screenshots taken by capture.py), and a closing scene for reserves. Writes intro_silent.mp4 (the sound comes from audio.py and is added by make_video.py).
PREVIEW=1 renders a small, fast version; FRAMES=2,5,8 writes single PNG frames for review."""
import math, os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__)); SHOTS = os.path.join(HERE, 'shots')
PREVIEW = os.environ.get('PREVIEW') == '1'
W, H = (1280, 720) if PREVIEW else (1920, 1080)
FPS = 30; U = H / 1080.0; DUR = 44.0
SCENES = [0, 9, 17, 27, 37, 44]; TR = 0.8
FB, FR_, FS_ = r'C:\Windows\Fonts\segoeuib.ttf', r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\seguisb.ttf'
_fc = {}


def font(path, size):
    k = (path, int(size))
    if k not in _fc: _fc[k] = ImageFont.truetype(path, int(size))
    return _fc[k]


def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
def lerp(a, b, t): return a + (b - a) * t
def mix(c1, c2, t): return tuple(lerp(a, b, t) for a, b in zip(c1, c2))
def seg(t, a, b): return ease((t - a) / (b - a))          # 0..1 between a and b
def pulse(t, a, b, c, d): return min(seg(t, a, b), 1 - seg(t, c, d))   # fade in a..b, hold, fade out c..d


# ---------------------------------------------------------------- sky, sun, mountains, water
KEYS = [(0.0, [(8, 18, 60), (28, 50, 120), (70, 100, 180)]), (0.5, [(38, 72, 160), (130, 125, 190), (250, 170, 120)]), (1.0, [(80, 140, 235), (150, 190, 250), (255, 228, 190)])]


def palette(p):
    p = clamp(p)
    for (p0, c0), (p1, c1) in zip(KEYS[:-1], KEYS[1:]):
        if p <= p1:
            k = (p - p0) / (p1 - p0); return [mix(a, b, k) for a, b in zip(c0, c1)]
    return KEYS[-1][1]


def sky_array(p):
    top, mid, bot = palette(p)
    y = np.linspace(0, 1, H)[:, None]
    col = np.where(y < .6, np.array(top) * (1 - y / .6) + np.array(mid) * (y / .6), np.array(mid) * (1 - (y - .6) / .4) + np.array(bot) * ((y - .6) / .4))
    return np.repeat(col[:, None, :], W, axis=1).astype(np.uint8)


_rng = np.random.default_rng(11)


def ridge(n, base, amp, rough, seed):
    r = np.random.default_rng(seed); x = np.zeros(n)
    for o, a in ((1, 1.0), (2, .55), (4, .3), (8, .16), (16, .08)):
        k = max(2, int(n / (60 * o / rough))); pts = r.standard_normal(k + 3)
        xs = np.linspace(0, k, n); i = xs.astype(int); f = xs - i; f = f * f * (3 - 2 * f)
        x += a * (pts[i] * (1 - f) + pts[i + 1] * f)
    x = (x - x.mean()) / (np.abs(x).max() + 1e-9)
    return base - amp * x


LAYERS = [dict(base=.60, amp=120, rough=1.0, speed=6, k=.18, seed=1), dict(base=.66, amp=150, rough=1.3, speed=12, k=.40, seed=2),
          dict(base=.74, amp=190, rough=1.6, speed=22, k=.62, seed=3), dict(base=.84, amp=230, rough=2.0, speed=40, k=.82, seed=4)]
RW = int(W * 2.2)
for L in LAYERS: L['prof'] = ridge(RW, L['base'] * H, L['amp'] * U, L['rough'], L['seed'])


def draw_mountains(arr, t, p, lift=0.0):
    """mountain layers drawn at half resolution (soft mist, soft edges) and scaled up: about four times faster than full size"""
    top, mid, bot = palette(p); horizon = np.array(bot, dtype=np.float32)
    deep = np.array(mix((8, 14, 42), (24, 52, 120), clamp(p)), dtype=np.float32)
    h2, w2 = H // 2, W // 2
    base_img = Image.fromarray(arr).convert('RGBA')
    yy = np.arange(h2, dtype=np.float32)[:, None]; span = max(2, int(150 * U))
    for L in LAYERS:
        off = int(L['speed'] * U * t) % (RW - W)
        ry = (L['prof'][off:off + W:2][:w2][None, :] + lift * H) / 2
        d = yy - ry; m = d >= 0
        idx = np.clip(d, 0, span).astype(np.int32)
        g = 1 - idx.astype(np.float32) / span
        base = horizon * (1 - L['k']) + deep * L['k']
        lut = np.stack([base[c] * (1 - g * .55) + horizon[c] * (g * .55) for c in range(3)], -1)
        lay = np.zeros((h2, w2, 4), np.uint8); lay[..., :3] = np.clip(lut, 0, 255).astype(np.uint8); lay[..., 3] = (m * 255).astype(np.uint8)
        big = Image.fromarray(lay, 'RGBA').resize((W, H), Image.BILINEAR)
        base_img.alpha_composite(big)
    return np.asarray(base_img.convert('RGB')).copy()


_GLOW = {}


def draw_sun(img, t, p, y0, y1):
    """glowing sun rising from y0 to y1 (fractions of H); the glow sprite is made once"""
    k = ease(t)
    cx, cy = int(W * .80), int(H * lerp(y0, y1, k)); R = int(70 * U)
    if 'g' not in _GLOW:
        glow = Image.new('RGBA', (int(900 * U), int(900 * U)), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
        for r in range(int(430 * U), 0, -6):
            a = int(70 * (1 - r / (430 * U)) ** 2.2); gd.ellipse((glow.width / 2 - r, glow.height / 2 - r, glow.width / 2 + r, glow.height / 2 + r), fill=(255, 205, 140, a))
        _GLOW['g'] = glow
    glow = _GLOW['g']
    img.alpha_composite(glow, (cx - glow.width // 2, cy - glow.height // 2)) if (cx - glow.width // 2) >= 0 and (cy - glow.height // 2) >= 0 else img.paste(glow, (cx - glow.width // 2, cy - glow.height // 2), glow)
    d = ImageDraw.Draw(img); d.ellipse((cx - R, cy - R, cx + R, cy + R), fill=(255, 238, 205, 255))


def draw_water(img, t, p):
    top, mid, bot = palette(p); d = ImageDraw.Draw(img, 'RGBA'); y0 = int(H * .915)
    for i, y in enumerate(range(y0, H, max(2, int(5 * U)))):
        k = (y - y0) / (H - y0 + 1e-9)
        c = mix(bot, (14, 34, 90), .35 + .6 * k); d.rectangle((0, y, W, y + int(5 * U)), fill=tuple(int(v) for v in c) + (255,))
    rng = np.random.default_rng(3)
    for i in range(70):   # shimmer lines of reflected light
        x = (rng.uniform(0, W) + 20 * math.sin(t * .8 + i)) % W; y = y0 + rng.uniform(0, H - y0)
        w = rng.uniform(20, 120) * U * (0.6 + 0.4 * math.sin(t * 1.3 + i * 2)); a = int(60 + 50 * math.sin(t * 1.7 + i))
        d.line((x, y, x + w, y), fill=(255, 235, 200, max(10, a)), width=max(1, int(2 * U)))


def draw_reeds(img, t, dark=(8, 22, 60)):
    d = ImageDraw.Draw(img); rng = np.random.default_rng(9)
    for side in (0, 1):
        for i in range(46):
            x0 = (rng.uniform(0, W * .26) if side == 0 else W - rng.uniform(0, W * .26)); hgt = rng.uniform(150, 360) * U; sway = math.sin(t * 1.2 + i * .7) * 14 * U
            pts = [(x0 + sway * (k / 6) ** 2, H - k / 6 * hgt) for k in range(7)]
            d.line(pts, fill=dark, width=max(2, int(4 * U)))
            if i % 3 == 0:
                ex, ey = pts[-1]; d.ellipse((ex - 5 * U, ey - 24 * U, ex + 5 * U, ey + 4 * U), fill=dark)


# ---------------------------------------------------------------- birds
_spr = {}


def sprite(phase_i, light):
    k = (phase_i, light)
    if k in _spr: return _spr[k]
    S = 4; cw, ch = 256 * S, 128 * S; im = Image.new('RGBA', (cw, ch), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    col = (235, 244, 255, 235) if light else (10, 24, 64, 235)
    ph = phase_i / 16 * 2 * math.pi; w = math.sin(ph); cx, cy, Lw = cw / 2, ch * .5, 112 * S
    for s in (-1, 1):
        pts = [(cx, cy), (cx + s * .45 * Lw, cy - w * .45 * Lw - .10 * Lw), (cx + s * Lw, cy - w * .85 * Lw + .16 * Lw)]
        d.line(pts, fill=col, width=int(9 * S), joint='curve')
        d.ellipse((pts[-1][0] - 4 * S, pts[-1][1] - 4 * S, pts[-1][0] + 4 * S, pts[-1][1] + 4 * S), fill=col)
    d.ellipse((cx - 11 * S, cy - 6 * S, cx + 11 * S, cy + 9 * S), fill=col)
    _spr[k] = im.resize((256, 128), Image.LANCZOS); return _spr[k]


class Flock:
    def __init__(self, n, seed, ymin=.10, ymax=.50, vmin=90, vmax=240, smin=26, smax=74):
        r = np.random.default_rng(seed)
        self.b = [dict(x0=r.uniform(0, W + 500), y=r.uniform(ymin, ymax) * H, v=r.uniform(vmin, vmax) * U, s=r.uniform(smin, smax) * U, f=r.uniform(2.0, 3.4), ph=r.uniform(0, 6.28), bob=r.uniform(4, 16) * U, bf=r.uniform(.3, .9)) for _ in range(n)]
        self.b.sort(key=lambda q: q['s'])

    def draw(self, img, t, light=False, alpha=1.0):
        for q in self.b:
            x = (q['x0'] + q['v'] * t) % (W + 500) - 250; y = q['y'] + q['bob'] * math.sin(t * q['bf'] * 2 + q['ph'])
            pi = int(((t * q['f'] + q['ph'] / 6.28) % 1) * 16); sz = int(q['s'] // 4 * 4) or 8
            sp = sprite(pi, light).resize((sz * 2, sz), Image.BILINEAR)
            if alpha < 1: sp = sp.copy(); sp.putalpha(sp.getchannel('A').point(lambda v: int(v * alpha)))
            img.alpha_composite(sp, (int(x - sz), int(y - sz / 2)))


# ---------------------------------------------------------------- text helpers
def put(img, layer, x, y):
    """alpha-composite a small layer at (x, y), clipped to the frame"""
    x, y = int(x), int(y); lw, lh = layer.size
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(img.width, x + lw), min(img.height, y + lh)
    if x1 <= x0 or y1 <= y0: return
    img.alpha_composite(layer.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def text(img, xy, s, f, fill=(11, 27, 58), alpha=1.0, anchor='la', shadow=False):
    if alpha <= 0.01: return
    pad = int(10 * U)
    probe = ImageDraw.Draw(Image.new('L', (4, 4)))
    l, t_, r, b = probe.textbbox((0, 0), s, font=f, anchor=anchor)
    lw, lh = int(r - l) + 2 * pad + 6, int(b - t_) + 2 * pad + 6
    lay = Image.new('RGBA', (lw, lh), (0, 0, 0, 0)); d = ImageDraw.Draw(lay); ox, oy = pad - l, pad - t_
    if shadow: d.text((ox + 2 * U, oy + 3 * U), s, font=f, fill=(0, 0, 0, int(110 * alpha)), anchor=anchor)
    d.text((ox, oy), s, font=f, fill=tuple(fill) + (int(255 * alpha),), anchor=anchor)
    put(img, lay, xy[0] - ox, xy[1] - oy)


def pill(img, x, y, w, h, fill, alpha=1.0, outline=None, r=None):
    pad = 3; lay = Image.new('RGBA', (int(w) + 2 * pad, int(h) + 2 * pad), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.rounded_rectangle((pad, pad, pad + w, pad + h), radius=r if r is not None else h / 2, fill=tuple(fill) + (int(255 * alpha),), outline=(tuple(outline) + (int(255 * alpha),)) if outline else None, width=max(1, int(2 * U)))
    put(img, lay, x - pad, y - pad)


def logo(img, cx, cy, s, alpha=1.0):
    pad = 4; n = int(s) + 2 * pad; lay = Image.new('RGBA', (n, n), (0, 0, 0, 0)); d = ImageDraw.Draw(lay); a = int(255 * alpha); c = n / 2
    d.rounded_rectangle((c - s / 2, c - s / 2, c + s / 2, c + s / 2), radius=s * .28, fill=(31, 91, 255, a))
    for dx, hh in ((-.31, .19), (-.10, .44), (.11, .70), (.32, .31)):
        d.rounded_rectangle((c + dx * s - s * .055, c - hh * s / 2, c + dx * s + s * .055, c + hh * s / 2), radius=s * .05, fill=(255, 255, 255, a))
    put(img, lay, cx - c, cy - c)




_SH = {}


def soft_shadow(img, x, y, w, h, radius, alpha, blur):
    """blurred drop shadow made on a small layer and cached by size"""
    key = (int(w), int(h), int(radius))
    if key not in _SH:
        m = int(blur * 3); lay = Image.new('RGBA', (int(w) + 2 * m, int(h) + 2 * m), (0, 0, 0, 0)); ImageDraw.Draw(lay).rounded_rectangle((m, m, m + w, m + h), radius=radius, fill=(11, 27, 58, 90))
        _SH[key] = (lay.filter(ImageFilter.GaussianBlur(blur)), m)
    lay, m = _SH[key]
    if alpha < 1: lay = lay.copy(); lay.putalpha(lay.getchannel('A').point(lambda v: int(v * alpha)))
    put(img, lay, x - m + 8 * U, y - m + 24 * U)


def shot(name):
    p = os.path.join(SHOTS, name + '.png')
    return Image.open(p).convert('RGB') if os.path.exists(p) else None


def fit_w(im, w): return im.resize((int(w), int(im.height * w / im.width)), Image.LANCZOS)


# ---------------------------------------------------------------- scenes
FLOCK1 = Flock(26, 4); FLOCK5 = Flock(22, 8, ymin=.08, ymax=.42)


def scene_dawn(t):
    """0-9 s: dawn over the mountains, birds fly by, the title appears"""
    p = lerp(.05, 1.0, ease(t / 8.5))
    arr = sky_array(p); img = Image.fromarray(arr).convert('RGBA')
    draw_sun(img, ease(t / 8.5), p, .80, .40)
    a = np.array(img.convert('RGB')); a = draw_mountains(a, t, p); img = Image.fromarray(a).convert('RGBA')
    draw_water(img, t, p); FLOCK1.draw(img, t, light=p < .45); draw_reeds(img, t)
    text(img, (W / 2, H * .30), 'WetlandEar', font(FB, 150 * U), (255, 255, 255), pulse(t, 1.6, 3.0, 8.2, 8.9), 'mm', shadow=True)
    text(img, (W / 2, H * .30 + 110 * U), 'Слушаем болото. Узнаём птиц.', font(FS_, 54 * U), (255, 255, 255), pulse(t, 2.6, 4.0, 8.2, 8.9), 'mm', shadow=True)
    logo(img, W / 2, H * .30 - 170 * U, 96 * U, pulse(t, 1.0, 2.2, 8.2, 8.9))
    return img


def spectro_strip(w, h):
    """a long synthetic spectrogram image: noise floor, a few species-like calls (harmonic curves)"""
    r = np.random.default_rng(21); a = r.random((h, w)).astype(np.float32) * .22
    a += (np.linspace(1, .2, h)[:, None] * .12)
    yy = np.arange(h)[:, None].astype(np.float32); xx = np.arange(w)[None, :].astype(np.float32)
    calls = [(500, 0.62, 26, .75, 'sweep'), (1100, 0.45, 22, .9, 'trill'), (1700, 0.70, 34, .8, 'sweep'), (2300, 0.30, 28, .85, 'trill'), (2900, 0.55, 30, .8, 'sweep')]
    for x0, f0, ln, amp, kind in calls:
        for rep in range(5):
            xs = x0 + rep * (ln * 4.2) + r.uniform(-ln * .9, ln * 1.4)
            for hmn in (1, 2, 3):
                t_ = np.linspace(0, 1, int(ln * 3))
                fy = (f0 * hmn * r.uniform(.85, 1.2) * (1 + r.uniform(.1, .45) * (t_ if kind == 'sweep' else np.sin(t_ * r.uniform(9, 20)) * .35))) * h * .55
                for u, yv in zip(t_, fy):
                    xi = int(xs + u * ln * 3); yi = int(h - 8 - yv)
                    if 0 <= xi < w and 3 <= yi < h - 3: a[yi - 2:yi + 3, xi:xi + 2] += amp / hmn
    im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6))
    arr = np.asarray(im).astype(np.float32) / 255
    stops = np.array([[10, 20, 60], [31, 91, 255], [150, 200, 255], [255, 255, 255]], dtype=np.float32); pos = np.array([0, .45, .8, 1])
    out = np.stack([np.interp(arr, pos, stops[:, c]) for c in range(3)], -1).astype(np.uint8)
    return Image.fromarray(out)


_STRIP = {}


def scene_listen(t):
    """9-17 s: long recordings, the model listens and suggests, a person confirms"""
    arr = sky_array(0.0); img = Image.fromarray(arr).convert('RGBA')
    a = np.array(img.convert('RGB')); a = draw_mountains(a, t + 30, .02, lift=.18); img = Image.fromarray(a).convert('RGBA')
    pw, ph_ = int(1250 * U), int(470 * U); px, py = int(110 * U), int(H * .40)
    pill(img, px - 20 * U, py - 20 * U, pw + 40 * U, ph_ + 40 * U, (255, 255, 255), .08, (255, 255, 255), r=34 * U)
    if 'strip' not in _STRIP: _STRIP['strip'] = spectro_strip(4200, int(ph_ / 1)) if not PREVIEW else spectro_strip(4200, ph_)
    strip = _STRIP['strip']; off = int(t * 190 * U) % (strip.width - pw)
    img.paste(strip.crop((off, 0, off + pw, ph_)), (px, py))
    hx = px + int(pw * .62); d = ImageDraw.Draw(img); d.line((hx, py - 8 * U, hx, py + ph_ + 8 * U), fill=(255, 190, 90), width=max(2, int(4 * U)))
    text(img, (px, py - 70 * U), 'Сотни часов записей.', font(FB, 64 * U), (255, 255, 255), pulse(t, .3, 1.1, 2.0, 2.7), 'la', shadow=True)
    text(img, (px, py - 70 * U), 'Слушать вручную невозможно.', font(FB, 64 * U), (255, 255, 255), pulse(t, 2.8, 3.6, 6.8, 7.6), 'la', shadow=True)
    chips = [('Серый журавль', 3.3), ('Лысуха', 4.4), ('Большая поганка', 5.5), ('Кудрявый пеликан', 6.6)]
    for i, (name, t0) in enumerate(chips):
        k = seg(t, t0, t0 + .5); y = py + 12 * U + i * 108 * U
        if k <= 0: continue
        cx = px + pw + 70 * U + (1 - k) * 80 * U
        pill(img, cx, y, 390 * U, 84 * U, (255, 255, 255), k)
        text(img, (cx + 28 * U, y + 24 * U), name, font(FS_, 34 * U), (11, 27, 58), k, 'la')
        ok = seg(t, t0 + .9, t0 + 1.3)
        if ok > 0:
            bx_, by_ = cx + 390 * U - 66 * U, y + 18 * U
            pill(img, bx_, by_, 48 * U, 48 * U, (20, 128, 74), ok)
            lay = Image.new('RGBA', img.size, (0, 0, 0, 0)); ImageDraw.Draw(lay).line([(bx_ + 12 * U, by_ + 25 * U), (bx_ + 21 * U, by_ + 34 * U), (bx_ + 37 * U, by_ + 15 * U)], fill=(255, 255, 255, int(255 * ok)), width=max(3, int(6 * U)), joint='curve')
            img.alpha_composite(lay)
    text(img, (px, py + ph_ + 60 * U), 'Модель предлагает. Человек подтверждает.', font(FS_, 48 * U), (200, 225, 255), pulse(t, 5.4, 6.2, 7.6, 8.4), 'la')
    text(img, (px, py + ph_ + 118 * U), 'Названия видов в этой сцене — пример.', font(FR_, 28 * U), (150, 180, 230), pulse(t, 5.4, 6.2, 7.6, 8.4), 'la')
    return img


def browser_frame(img, x, y, w, h, content, scroll=0.0, alpha=1.0):
    soft_shadow(img, x, y, w, h, 28 * U, alpha, 26 * U)
    lay = Image.new('RGBA', (int(w) + 4, int(h) + 4), (0, 0, 0, 0)); d = ImageDraw.Draw(lay); a = int(255 * alpha); o = 2
    d.rounded_rectangle((o, o, o + w, o + h), radius=22 * U, fill=(255, 255, 255, a), outline=(205, 215, 235, a), width=2)
    d.rounded_rectangle((o, o, o + w, o + 54 * U), radius=22 * U, fill=(238, 242, 250, a)); d.rectangle((o, o + 30 * U, o + w, o + 54 * U), fill=(238, 242, 250, a))
    for i, c in enumerate(((255, 96, 92), (255, 189, 68), (0, 202, 78))): d.ellipse((o + 22 * U + i * 28 * U, o + 18 * U, o + 38 * U + i * 28 * U, o + 34 * U), fill=c + (a,))
    d.rounded_rectangle((o + 150 * U, o + 12 * U, o + w - 150 * U, o + 42 * U), radius=15 * U, fill=(255, 255, 255, a))
    put(img, lay, x - o, y - o)
    text(img, (x + w / 2, y + 27 * U), 'wetlandear.tech', font(FR_, 20 * U), (85, 101, 127), alpha, 'mm')
    if content is not None:
        cw, chh = int(w - 4), int(h - 56 * U - 4); c = content.resize((cw, int(content.height * cw / content.width)), Image.BILINEAR)
        view = c.crop((0, 0, cw, min(chh, c.height))).convert('RGBA')
        if alpha < 1: view.putalpha(int(255 * alpha))
        put(img, view, x + 2, y + 56 * U)


def scene_site(t):
    """17-27 s: the project site"""
    img = Image.new('RGBA', (W, H), (247, 249, 253, 255)); d = ImageDraw.Draw(img)
    for i, (cx, cy, r, c) in enumerate(((.12, .18, 420, (226, 236, 255)), (.9, .85, 520, (230, 238, 255)), (.55, .05, 300, (238, 243, 255)))):
        d.ellipse((W * cx - r * U, H * cy - r * U, W * cx + r * U, H * cy + r * U), fill=c + (255,))
    Flock(10, 31, ymin=.05, ymax=.9, vmin=30, vmax=70, smin=18, smax=34).draw(img, t, light=False, alpha=.18)
    names = ['site_0', 'site_1', 'site_6']; ims = [shot(n) for n in names]
    bw, bh = int(1380 * U), int(840 * U); bx, by = int((W - bw) / 2), int(H * .22)
    k = clamp(t / 10.0) * 2.999; i = int(k); f = ease(k - i) if (k - i) > .72 else 0.0
    slide = ease(seg(t, 0, .8)); by2 = int(by + (1 - slide) * 60 * U)
    cur = ims[i]; nxt = ims[min(i + 1, 2)]
    browser_frame(img, bx, by2, bw, bh, cur, alpha=slide)
    if f > 0 and nxt is not None: browser_frame(img, bx, by2, bw, bh, nxt, alpha=f)
    text(img, (W / 2, H * .105), 'Сайт проекта', font(FB, 70 * U), (11, 27, 58), pulse(t, .1, .8, 9.2, 9.9), 'mm')
    text(img, (W / 2, H * .165), 'Как это работает, метрики, платформа для заповедников', font(FR_, 36 * U), (85, 101, 127), pulse(t, .3, 1.0, 9.2, 9.9), 'mm')
    return img


def phone(img, x, y, scr, alpha=1.0, w=400):
    w = w * U; h = w * 2.17
    soft_shadow(img, x, y, w, h, 64 * U, alpha, 28 * U)
    a = int(255 * alpha); lay = Image.new('RGBA', (int(w) + 4, int(h) + 4), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.rounded_rectangle((2, 2, 2 + w, 2 + h), radius=64 * U, fill=(14, 26, 60, a)); put(img, lay, x - 2, y - 2)
    if scr is not None:
        sw = int(w - 24 * U); sh_ = int(h - 24 * U); s = scr.resize((sw, int(scr.height * sw / scr.width)), Image.BILINEAR).crop((0, 0, sw, sh_)).convert('RGBA')
        m = Image.new('L', (sw, sh_), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, sw, sh_), radius=int(48 * U), fill=int(255 * alpha)); s.putalpha(m)
        put(img, s, x + 12 * U, y + 12 * U)
    n = Image.new('RGBA', (int(116 * U) + 4, int(22 * U) + 4), (0, 0, 0, 0)); ImageDraw.Draw(n).rounded_rectangle((2, 2, 2 + 116 * U, 2 + 22 * U), radius=11 * U, fill=(14, 26, 60, a))
    put(img, n, x + w / 2 - 58 * U - 2, y + 22 * U - 2)


APP_STEPS = [('app_pulse', 'Пульс заповедника', ['Сколько событий, видов и сколько подтверждено человеком.', 'Живое состояние на одном экране.']),
             ('app_quick', 'Быстрые действия', ['Записать звук, загрузить файлы, проверить события.', 'Одна кнопка «+» и поиск Ctrl+K.']),
             ('app_science', 'Наука и учёт', ['Первые прилёты по годам, индикаторные виды.', 'Данные для отчётов и грантов.']),
             ('app_map', 'Карта станций', ['Где и что слышали, где тревоги.', 'Работает и без интернета.'])]


def scene_app(t):
    """27-37 s: the app on a phone"""
    img = Image.new('RGBA', (W, H), (247, 249, 253, 255)); d = ImageDraw.Draw(img)
    for cx, cy, r, c in ((.85, .2, 480, (226, 236, 255)), (.08, .9, 520, (232, 239, 255))): d.ellipse((W * cx - r * U, H * cy - r * U, W * cx + r * U, H * cy + r * U), fill=c + (255,))
    Flock(9, 33, ymin=.05, ymax=.9, vmin=30, vmax=70, smin=18, smax=34).draw(img, t, light=False, alpha=.16)
    n = len(APP_STEPS); step = 10.0 / n; i = min(n - 1, int(t / step)); lt = t - i * step
    px, py = int(W * .12), int(H * .09)
    slide = ease(seg(t, 0, .8)); pyy = int(py + (1 - slide) * 140 * U)
    cur = shot(APP_STEPS[i][0]); prev = shot(APP_STEPS[i - 1][0]) if i > 0 else None
    k = seg(lt, 0, .55)
    phone(img, px, pyy, cur if (i == 0 or k >= 1) else None, alpha=slide)
    if i > 0 and k < 1 and prev is not None and cur is not None:
        pw_ = 400 * U; sw = int(pw_ - 24 * U); sh_ = int(pw_ * 2.17 - 24 * U)
        a_ = prev.resize((sw, int(prev.height * sw / prev.width))).crop((0, 0, sw, sh_)); b_ = cur.resize((sw, int(cur.height * sw / cur.width))).crop((0, 0, sw, sh_))
        mixed = Image.blend(a_, b_, k); phone(img, px, pyy, mixed, alpha=slide)
    tx = int(W * .12 + 400 * U + 170 * U)
    for j, (nm, title, lines) in enumerate(APP_STEPS):
        a = pulse(t, j * step + .15, j * step + .65, (j + 1) * step - .2, (j + 1) * step + .1) if j < n - 1 else pulse(t, j * step + .15, j * step + .65, 9.8, 10.2)
        if a > 0:
            text(img, (tx, H * .30), title, font(FB, 78 * U), (11, 27, 58), a, 'la')
            for q, ln in enumerate(lines): text(img, (tx, H * .30 + 130 * U + q * 62 * U), ln, font(FR_, 40 * U), (85, 101, 127), a, 'la')
    for j, (lab) in enumerate(('1  Записал', '2  Модель предложила', '3  Человек подтвердил')):
        a = seg(t, 1.2 + j * .5, 1.8 + j * .5)
        pill(img, tx + j * 372 * U, H * .76, 356 * U, 62 * U, (232, 239, 255), a); text(img, (tx + j * 372 * U + 178 * U, H * .76 + 31 * U), lab, font(FS_, 27 * U), (31, 91, 255), a, 'mm')
    text(img, (px, H - 46 * U), 'На экранах приложения — демонстрационные данные.', font(FR_, 26 * U), (120, 135, 160), pulse(t, .6, 1.4, 9.2, 9.9), 'la')
    return img


PINS = [(.34, .66, 'Кургальджинские озёра'), (.60, .70, 'степи и озёра'), (.80, .60, 'горы')]


def scene_reserves(t):
    """37-44 s: back to the landscape, for the reserves"""
    p = .9
    arr = sky_array(p); img = Image.fromarray(arr).convert('RGBA')
    draw_sun(img, 1.0, p, .40, .40)
    a = np.array(img.convert('RGB')); a = draw_mountains(a, t + 60, p); img = Image.fromarray(a).convert('RGBA')
    draw_water(img, t + 20, p); FLOCK5.draw(img, t + 10, light=False); draw_reeds(img, t)
    text(img, (W / 2, H * .22), 'Для заповедников Казахстана', font(FB, 92 * U), (255, 255, 255), pulse(t, .4, 1.4, 5.4, 6.0), 'mm', shadow=True)
    chips = ['268 видов', 'Telegram-тревоги', 'PDF-отчёты', 'Без интернета', 'Публичные страницы']
    wd = [int(len(c) * 19 * U + 60 * U) for c in chips]; total = sum(wd) + 24 * U * (len(chips) - 1); x = (W - total) / 2
    for c, w_ in zip(chips, wd):
        k = seg(t, 1.2 + chips.index(c) * .25, 1.7 + chips.index(c) * .25) * (1 - seg(t, 4.8, 5.6))
        pill(img, x, H * .30, w_, 68 * U, (255, 255, 255), .92 * k); text(img, (x + w_ / 2, H * .30 + 34 * U), c, font(FS_, 32 * U), (11, 27, 58), k, 'mm'); x += w_ + 24 * U
    fin = seg(t, 5.2, 6.2)
    if fin > 0:
        veil = Image.new('RGBA', (W, H), (8, 20, 60, int(150 * fin))); img.alpha_composite(veil)
        logo(img, W / 2, H * .36, 130 * U, fin)
        text(img, (W / 2, H * .50), 'WetlandEar', font(FB, 130 * U), (255, 255, 255), fin, 'mm')
        text(img, (W / 2, H * .50 + 100 * U), 'Слушаем болото. Узнаём птиц.', font(FS_, 52 * U), (210, 228, 255), fin, 'mm')
        text(img, (W / 2, H * .72), 'wetlandear.tech   ·   app.wetlandear.tech', font(FS_, 44 * U), (255, 255, 255), fin, 'mm')
        text(img, (W / 2, H * .80), 'Модель предлагает, человек подтверждает. Данные: Xeno-canto, iNaturalist. Основа: BirdNET.', font(FR_, 26 * U), (170, 195, 235), fin, 'mm')
    return img


SCENE_FN = [scene_dawn, scene_listen, scene_site, scene_app, scene_reserves]


def frame(i):
    t = i / FPS; s = max(k for k in range(len(SCENES) - 1) if t >= SCENES[k] - 1e-9); s = min(s, len(SCENE_FN) - 1)
    img = SCENE_FN[s](t - SCENES[s]); end = SCENES[s + 1]
    if s + 1 < len(SCENE_FN) and t > end - TR:
        nxt = SCENE_FN[s + 1](0.0); img = Image.blend(img.convert('RGB'), nxt.convert('RGB'), ease((t - (end - TR)) / TR))
    # first scene fades in from dark, last fades out
    if t < .8: img = Image.blend(Image.new('RGB', (W, H), (6, 14, 40)), img.convert('RGB'), ease(t / .8))
    if t > DUR - 1.0: img = Image.blend(img.convert('RGB'), Image.new('RGB', (W, H), (6, 14, 40)), ease((t - (DUR - 1.0)) / 1.0))
    return img.convert('RGB')


def main():
    if os.environ.get('FRAMES'):
        for s in os.environ['FRAMES'].split(','):
            f = float(s); frame(int(f * FPS)).save(os.path.join(HERE, '_f_%s.png' % s.replace('.', '_'))); print('frame', s, flush=True)
        return
    out = os.path.join(HERE, 'intro_silent.mp4'); n = int(DUR * FPS)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(n):
        pr.stdin.write(frame(i).tobytes())
        if i % 60 == 0: print('frame %d/%d' % (i, n), flush=True)
    pr.stdin.close(); pr.wait(); print('written', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
