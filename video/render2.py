"""Intro video v2: English (or Russian with VLANG=ru), real open-licence photos with slow camera moves, five different transitions, light rays, floating particles,
letter-by-letter titles, a vignette, and the music from audio2.py. Reuses the drawing helpers of render.py.
PREVIEW=1 small and fast; FRAMES=5,13,21 writes single PNG frames; otherwise writes intro2_silent.mp4."""
import json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio_ffmpeg
import render as R
from render import W, H, U, FPS, ease, clamp, lerp, mix, seg, pulse, font, FB, FR_, FS_, text, pill, logo, put

LANGV = os.environ.get('VLANG', 'en')
R.SHOTS = os.path.join(R.HERE, 'shots' if LANGV == 'ru' else 'shots_' + LANGV)
PHOTOS = os.path.join(R.HERE, 'photos')
B = [0, 10, 17.5, 27.5, 35, 42.5, 50]; TR = 1.0; DUR = 50.0
CREDITS = json.load(open(os.path.join(PHOTOS, 'credits.json'), encoding='utf-8')) if os.path.exists(os.path.join(PHOTOS, 'credits.json')) else {}
WHO = {'mountains': 'Bgag', 'cranes': 'Dirk Ingo Franke', 'pelican': 'gailhampshire', 'flamingo': 'VasuVR', 'demoiselle': 'Keenanhye'}
PH_CREDIT = ', '.join('%s (%s)' % (WHO[k], CREDITS.get(k, {}).get('license', '?')) for k in WHO)

TXT = {
    'en': dict(tag='We listen to the wetland. We name the birds.', s2a='Hundreds of hours of recordings.', s2b='Nobody can listen to all of it.', s2c='The model suggests. A person confirms.', s2n='Species in this scene are an example.',
               chips=['Common Crane', 'Eurasian Coot', 'Great Crested Grebe', 'Dalmatian Pelican'], p1='Reserves are vast.', p2='The people in the field are few.', p3='Birds keep calling. Somebody has to listen.',
               cap=['Dalmatian pelican', 'Greater flamingo', 'Demoiselle crane'], pnote='Illustrative photos', site='The project site', site2='How it works, open metrics, a platform for reserves',
               steps=[('Reserve pulse', ['Events, species and what a person confirmed.', 'The state of the reserve on one screen.']), ('Quick actions', ['Record, upload files, check events.', 'One "+" button and Ctrl+K search.']),
                      ('Science and records', ['First arrivals by year, indicator species.', 'Data for reports and grants.'])], flow=['1  Record', '2  Model suggests', '3  Person confirms'], demo='The app screens show demo data.',
               fin1='Built for the reserves of Kazakhstan', fchips=['270 species', 'Telegram alerts', 'PDF reports', 'Works offline', 'Public pages'],
               fin2='The model suggests, a person confirms.  Data: Xeno-canto, iNaturalist.  Core: BirdNET.', photos='Photos via Wikimedia Commons: '),
    'ru': dict(tag='Слушаем болото. Узнаём птиц.', s2a='Сотни часов записей.', s2b='Слушать вручную невозможно.', s2c='Модель предлагает. Человек подтверждает.', s2n='Названия видов в этой сцене — пример.',
               chips=['Серый журавль', 'Лысуха', 'Большая поганка', 'Кудрявый пеликан'], p1='Заповедники огромны.', p2='Людей в поле мало.', p3='Птицы поют. Кто-то должен слушать.',
               cap=['Кудрявый пеликан', 'Розовый фламинго', 'Журавль-красавка'], pnote='Иллюстративные фото', site='Сайт проекта', site2='Как это работает, открытые метрики, платформа для заповедников',
               steps=[('Пульс заповедника', ['Сколько событий, видов и подтверждено человеком.', 'Состояние на одном экране.']), ('Быстрые действия', ['Записать звук, загрузить файлы, проверить события.', 'Кнопка «+» и поиск Ctrl+K.']),
                      ('Наука и учёт', ['Первые прилёты по годам, индикаторные виды.', 'Данные для отчётов и грантов.'])], flow=['1  Записал', '2  Модель предложила', '3  Человек подтвердил'], demo='На экранах приложения — демонстрационные данные.',
               fin1='Для заповедников Казахстана', fchips=['270 видов', 'Telegram-тревоги', 'PDF-отчёты', 'Без интернета', 'Публичные страницы'],
               fin2='Модель предлагает, человек подтверждает.  Данные: Xeno-canto, iNaturalist.  Основа: BirdNET.', photos='Фото с Wikimedia Commons: '),
}
T = TXT[LANGV]

# ---------------------------------------------------------------- generic effects
_VIG = {}


def vignette(img):
    if 'v' not in _VIG:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32); d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        _VIG['v'] = (1 - np.clip(d - .55, 0, 1) ** 1.6 * .42)[..., None]
    a = np.asarray(img.convert('RGB')).astype(np.float32) * _VIG['v']
    return Image.fromarray(a.astype(np.uint8))


_MOTE = {}


def motes(img, t, n=46, seed=5, tint=(255, 240, 215), amp=1.0, speed=1.0):
    if 's' not in _MOTE:
        s = Image.new('RGBA', (28, 28), (0, 0, 0, 0)); d = ImageDraw.Draw(s)
        for r in range(13, 0, -1): d.ellipse((14 - r, 14 - r, 14 + r, 14 + r), fill=(255, 255, 255, int(120 * (1 - r / 13) ** 1.8)))
        _MOTE['s'] = s
    r = np.random.default_rng(seed); base = _MOTE['s']
    for i in range(n):
        x = (r.uniform(0, W) + r.uniform(-30, 30) * speed * t * .3 + 22 * U * math.sin(t * .6 + i)) % W; y = (r.uniform(0, H) - r.uniform(8, 26) * U * speed * t) % H
        sz = int(r.uniform(10, 30) * U); a = (.35 + .35 * math.sin(t * 1.4 + i * 1.9)) * amp
        sp = base.resize((sz, sz), Image.BILINEAR); tint_l = Image.new('RGBA', sp.size, tint + (0,)); tint_l.putalpha(sp.getchannel('A').point(lambda v: int(v * a)))
        put(img, tint_l, x, y)


def rays(img, cx, cy, strength):
    if strength <= 0.01: return
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i in range(9):
        ang = math.radians(100 + i * 9 + 3 * math.sin(i)); w = math.radians(2.6 + (i % 3)); L = H * 1.7
        pts = [(cx, cy), (cx + L * math.cos(ang - w), cy + L * math.sin(ang - w)), (cx + L * math.cos(ang + w), cy + L * math.sin(ang + w))]
        d.polygon(pts, fill=(255, 224, 170, int(24 * strength)))
    img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(6 * U)))


def kinetic(img, center, s, f, fill, t, t0, per=.045, rise=30, shadow=True, hold=1.0):
    """letter-by-letter title: each letter rises and fades in"""
    ws = [f.getlength(c) for c in s]; x = center[0] - sum(ws) / 2
    for i, c in enumerate(s):
        a = ease((t - t0 - i * per) / .4) * hold
        if a > .01: text(img, (x, center[1] + (1 - a) * rise * U), c, f, fill, a, 'lm', shadow)
        x += ws[i]


def shine(img, cx, cy, wid, hgt, t, t0, dur=1.2):
    k = (t - t0) / dur
    if not 0 < k < 1: return
    x = cx - wid / 2 + k * (wid + 160 * U) - 80 * U
    lay = Image.new('RGBA', (int(120 * U), int(hgt)), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i in range(int(120 * U)): d.line((i, 0, i, hgt), fill=(255, 255, 255, int(70 * math.sin(math.pi * i / (120 * U)) ** 2)))
    lay = lay.rotate(-14, expand=True, resample=Image.BICUBIC); put(img, lay, x, cy - hgt / 2 - 20 * U)


# ---------------------------------------------------------------- photos with a slow camera
_PH = {}


def photo(name):
    if name not in _PH:
        im = Image.open(os.path.join(PHOTOS, name + '.jpg')).convert('RGB'); k = (H * 1.25) / min(im.size) if min(im.size) < H * 1.25 else 1.0
        _PH[name] = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS) if k > 1 else im
    return _PH[name]


def kb(name, t, t0, t1, z0, z1, p0, p1):
    """Ken Burns: slow zoom and pan over a photo; p are (x, y) in 0..1 for where the crop sits"""
    im = photo(name); iw, ih = im.size; k = ease((t - t0) / (t1 - t0)); z = lerp(z0, z1, k)
    ch = ih / z; cw = ch * W / H
    if cw > iw: cw = iw / z; ch = cw * H / W
    px = lerp(p0[0], p1[0], k); py = lerp(p0[1], p1[1], k); x0 = (iw - cw) * px; y0 = (ih - ch) * py
    return im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BILINEAR).convert('RGBA')


_GR = {}


def grade(img, bottom=.65, tint=(11, 27, 58), tint_a=.16):
    if 'g' not in _GR:
        g = np.zeros((H, W, 4), np.uint8); y = np.linspace(0, 1, H)[:, None]; a = np.clip((y - .35) / .65, 0, 1) ** 1.5 * 190
        g[..., 3] = np.repeat(a, W, 1).astype(np.uint8); g[..., :3] = (6, 14, 40); _GR['g'] = Image.fromarray(g, 'RGBA')
    img.alpha_composite(Image.new('RGBA', (W, H), tint + (int(255 * tint_a),))); img.alpha_composite(_GR['g'])
    return img


_CARD = {}


def card(name, caption, angle, w=430, h=540):
    key = (name, angle)
    if key in _CARD: return _CARD[key]
    w, h = int(w * U), int(h * U); b = int(12 * U); cap = int(76 * U)
    im = photo(name); iw, ih = im.size; tw, th = w, h; r = tw / th
    cw = ih * r
    if cw > iw: cw = iw; ch_ = iw / r
    else: ch_ = ih
    crop = im.crop((int((iw - cw) / 2), int((ih - ch_) / 2), int((iw + cw) / 2), int((ih + ch_) / 2))).resize((tw, th), Image.LANCZOS)
    total_w, total_h = w + 2 * b, h + 2 * b + cap
    c = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0)); d = ImageDraw.Draw(c); d.rounded_rectangle((0, 0, total_w - 1, total_h - 1), radius=int(22 * U), fill=(255, 255, 255, 255))
    m = Image.new('L', (tw, th), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, tw - 1, th - 1), radius=int(14 * U), fill=255); c.paste(crop, (b, b), m)
    d.text((total_w / 2, b + th + cap / 2 + 4 * U), caption, font=font(FS_, 31 * U), fill=(11, 27, 58, 255), anchor='mm')
    pad = int(60 * U); sh = Image.new('RGBA', (total_w + 2 * pad, total_h + 2 * pad), (0, 0, 0, 0)); ImageDraw.Draw(sh).rounded_rectangle((pad, pad + 14 * U, pad + total_w, pad + total_h + 14 * U), radius=int(22 * U), fill=(0, 0, 0, 120))
    sh = sh.filter(ImageFilter.GaussianBlur(18 * U)); sh.alpha_composite(c, (pad, pad))
    out = sh.rotate(angle, expand=True, resample=Image.BICUBIC); _CARD[key] = out; return out


# ---------------------------------------------------------------- scenes
FL1 = R.Flock(28, 41); FL6 = R.Flock(24, 47, ymin=.08, ymax=.42)


def scene_dawn(t):
    p = lerp(.05, 1.0, ease(t / 9.0)); arr = R.sky_array(p); img = Image.fromarray(arr).convert('RGBA')
    sun_k = ease(t / 9.0); R.draw_sun(img, sun_k, p, .80, .40)
    sx, sy = int(W * .80), int(H * lerp(.80, .40, sun_k)); rays(img, sx, sy, seg(t, 2, 6))
    a = np.array(img.convert('RGB')); a = R.draw_mountains(a, t + 2, p); img = Image.fromarray(a).convert('RGBA')
    R.draw_water(img, t, p); FL1.draw(img, t, light=p < .45); R.draw_reeds(img, t); motes(img, t, 40, 5, (255, 236, 205), .8)
    logo(img, W / 2, H * .30 - 175 * U, 100 * U, pulse(t, .9, 1.8, 9.0, 9.9))
    kinetic(img, (W / 2, H * .30), 'WetlandEar', font(FB, 156 * U), (255, 255, 255), t, 1.3, hold=1 - seg(t, 8.9, 9.9))
    shine(img, W / 2, H * .30, 780 * U, 200 * U, t, 3.4)
    text(img, (W / 2, H * .30 + 112 * U), T['tag'], font(FS_, 52 * U), (255, 255, 255), pulse(t, 2.9, 4.0, 8.9, 9.9), 'mm', shadow=True)
    return img


def scene_listen(t):
    arr = R.sky_array(0.0); img = Image.fromarray(arr).convert('RGBA'); a = np.array(img.convert('RGB')); a = R.draw_mountains(a, t + 30, .02, lift=.18); img = Image.fromarray(a).convert('RGBA')
    motes(img, t + 3, 36, 9, (150, 190, 255), .7)
    pw, ph_ = int(1250 * U), int(470 * U); px, py = int(110 * U), int(H * .40)
    pill(img, px - 20 * U, py - 20 * U, pw + 40 * U, ph_ + 40 * U, (255, 255, 255), .08, (255, 255, 255), r=34 * U)
    if 'strip' not in R._STRIP: R._STRIP['strip'] = R.spectro_strip(4200, ph_)
    strip = R._STRIP['strip']; off = int(t * 190 * U) % (strip.width - pw); img.paste(strip.crop((off, 0, off + pw, ph_)), (px, py))
    hx = px + int(pw * .62); d = ImageDraw.Draw(img); d.line((hx, py - 8 * U, hx, py + ph_ + 8 * U), fill=(255, 190, 90), width=max(2, int(4 * U)))
    text(img, (px, py - 70 * U), T['s2a'], font(FB, 64 * U), (255, 255, 255), pulse(t, 1.2, 2.0, 3.0, 3.7), 'la', shadow=True)
    text(img, (px, py - 70 * U), T['s2b'], font(FB, 64 * U), (255, 255, 255), pulse(t, 3.9, 4.7, 7.8, 8.6), 'la', shadow=True)
    for i, name in enumerate(T['chips']):
        t0 = 4.3 + i * 1.0; k = seg(t, t0, t0 + .5); y = py + 12 * U + i * 108 * U
        if k <= 0: continue
        cx = px + pw + 70 * U + (1 - k) * 90 * U; pill(img, cx, y, 390 * U, 84 * U, (255, 255, 255), k)
        text(img, (cx + 28 * U, y + 24 * U), name, font(FS_, 33 * U), (11, 27, 58), k, 'la'); ok = seg(t, t0 + .9, t0 + 1.3)
        if ok > 0:
            bx_, by_ = cx + 390 * U - 66 * U, y + 18 * U; pill(img, bx_, by_, 48 * U, 48 * U, (20, 128, 74), ok)
            lay = Image.new('RGBA', img.size, (0, 0, 0, 0)); ImageDraw.Draw(lay).line([(bx_ + 12 * U, by_ + 25 * U), (bx_ + 21 * U, by_ + 34 * U), (bx_ + 37 * U, by_ + 15 * U)], fill=(255, 255, 255, int(255 * ok)), width=max(3, int(6 * U)), joint='curve'); img.alpha_composite(lay)
    text(img, (px, py + ph_ + 60 * U), T['s2c'], font(FS_, 48 * U), (200, 225, 255), pulse(t, 6.3, 7.1, 8.6, 9.4), 'la')
    text(img, (px, py + ph_ + 118 * U), T['s2n'], font(FR_, 28 * U), (150, 180, 230), pulse(t, 6.3, 7.1, 8.6, 9.4), 'la')
    return img


def scene_photos(t):
    """mountains and cranes with a slow camera, then three photo cards"""
    img = kb('mountains', t, 0, 6.2, 1.0, 1.22, (.1, .55), (.62, .45))
    c2 = seg(t, 3.4, 4.4)
    if c2 > 0:
        cr = kb('cranes', t, 3.4, 10.6, 1.0, 1.18, (0.0, .5), (1.0, .5)); cr.putalpha(int(255 * c2)); img.alpha_composite(cr)
    c3 = seg(t, 6.8, 7.8)
    if c3 > 0:
        bg = kb('mountains', t, 6.8, 11.5, 1.3, 1.4, (.5, .5), (.6, .5)).filter(ImageFilter.GaussianBlur(26 * U)); bg.putalpha(int(255 * c3)); img.alpha_composite(Image.new('RGBA', (W, H), (8, 20, 60, int(120 * c3)))) if False else None
        dark = Image.new('RGBA', (W, H), (8, 20, 60, int(150 * c3))); img.alpha_composite(bg); img.alpha_composite(dark)
    grade(img, tint_a=.12 * (1 - c3))
    text(img, (int(110 * U), H * .80), T['p1'], font(FB, 100 * U), (255, 255, 255), pulse(t, .8, 1.8, 3.2, 4.0), 'lm', shadow=True)
    text(img, (int(110 * U), H * .80), T['p2'], font(FB, 100 * U), (255, 255, 255), pulse(t, 4.2, 5.2, 6.2, 7.0), 'lm', shadow=True)
    angs = [-5, 2.5, -2.5]; xs = [.17, .50, .83]
    for i in range(3):
        t0 = 7.2 + i * .45; k = seg(t, t0, t0 + .8)
        if k <= 0: continue
        cd = card(['pelican', 'flamingo', 'demoiselle'][i], T['cap'][i], angs[i]); cd2 = cd.copy(); cd2.putalpha(cd.getchannel('A').point(lambda v: int(v * k)))
        fl = 10 * U * math.sin(t * 1.1 + i * 2); put(img, cd2, W * xs[i] - cd.width / 2, H * .50 - cd.height / 2 + (1 - k) * 160 * U + fl)
    text(img, (W / 2, H * .90), T['p3'], font(FB, 56 * U), (255, 255, 255), pulse(t, 8.3, 9.1, 11.0, 11.6), 'mm', shadow=True)
    text(img, (W / 2, H - 38 * U), T['pnote'] + '. ' + T['photos'] + PH_CREDIT, font(FR_, 21 * U), (210, 222, 245), pulse(t, 1.0, 1.8, 11.0, 11.6) * .85, 'mm')
    return img


def scene_site(t):
    img = Image.new('RGBA', (W, H), (247, 249, 253, 255)); d = ImageDraw.Draw(img)
    for cx, cy, r, c in ((.12, .18, 420, (226, 236, 255)), (.9, .85, 520, (230, 238, 255)), (.55, .05, 300, (238, 243, 255))): d.ellipse((W * cx - r * U, H * cy - r * U, W * cx + r * U, H * cy + r * U), fill=c + (255,))
    motes(img, t, 30, 13, (31, 91, 255), .35, .6)
    ims = [R.shot(n) for n in ('site_0', 'site_1', 'site_6')]
    bw, bh = int(1380 * U), int(840 * U); bx, by = int((W - bw) / 2), int(H * .22)
    k = clamp((t - 1.0) / 7.0) * 2.999; i = int(k); f = ease(k - i) if (k - i) > .72 else 0.0
    slide = ease(seg(t, 0, .9)); by2 = int(by + (1 - slide) * 90 * U + 8 * U * math.sin(t * .8))
    R.browser_frame(img, bx, by2, bw, bh, ims[i], alpha=slide)
    if f > 0 and ims[min(i + 1, 2)] is not None: R.browser_frame(img, bx, by2, bw, bh, ims[min(i + 1, 2)], alpha=f)
    text(img, (W / 2, H * .105), T['site'], font(FB, 70 * U), (11, 27, 58), pulse(t, .2, 1.0, 7.4, 8.2), 'mm')
    text(img, (W / 2, H * .165), T['site2'], font(FR_, 36 * U), (85, 101, 127), pulse(t, .4, 1.2, 7.4, 8.2), 'mm')
    return img


def scene_app(t):
    img = Image.new('RGBA', (W, H), (247, 249, 253, 255)); d = ImageDraw.Draw(img)
    for cx, cy, r, c in ((.85, .2, 480, (226, 236, 255)), (.08, .9, 520, (232, 239, 255))): d.ellipse((W * cx - r * U, H * cy - r * U, W * cx + r * U, H * cy + r * U), fill=c + (255,))
    motes(img, t, 30, 17, (31, 91, 255), .35, .6)
    names = ['app_pulse', 'app_quick', 'app_science']; step = 2.5; i = min(2, int(max(0, t - 1.0) / step)); lt = max(0, t - 1.0) - i * step
    px, py = int(W * .12), int(H * .09); slide = ease(seg(t, 0, .9)); pyy = int(py + (1 - slide) * 150 * U + 6 * U * math.sin(t * .9))
    cur = R.shot(names[i]); prev = R.shot(names[i - 1]) if i > 0 else None; k = seg(lt, 0, .55)
    if i == 0 or k >= 1 or prev is None or cur is None: R.phone(img, px, pyy, cur, alpha=slide)
    else:
        pw_ = 400 * U; sw = int(pw_ - 24 * U); sh_ = int(pw_ * 2.17 - 24 * U)
        a_ = prev.resize((sw, int(prev.height * sw / prev.width))).crop((0, 0, sw, sh_)); b_ = cur.resize((sw, int(cur.height * sw / cur.width))).crop((0, 0, sw, sh_)); R.phone(img, px, pyy, Image.blend(a_, b_, k), alpha=slide)
    tx = int(W * .12 + 400 * U + 170 * U)
    for j, (title, lines) in enumerate(T['steps']):
        s0 = 1.0 + j * step; a = pulse(t, s0 + .1, s0 + .7, s0 + step - .25, s0 + step + .05) if j < 2 else pulse(t, s0 + .1, s0 + .7, 8.4, 9.0)
        if a > 0:
            text(img, (tx + (1 - a) * 40 * U, H * .30), title, font(FB, 78 * U), (11, 27, 58), a, 'la')
            for q, ln in enumerate(lines): text(img, (tx + (1 - a) * 40 * U, H * .30 + 130 * U + q * 62 * U), ln, font(FR_, 40 * U), (85, 101, 127), a, 'la')
    for j, lab in enumerate(T['flow']):
        a = seg(t, 1.6 + j * .5, 2.2 + j * .5); pill(img, tx + j * 372 * U, H * .76, 356 * U, 62 * U, (232, 239, 255), a); text(img, (tx + j * 372 * U + 178 * U, H * .76 + 31 * U), lab, font(FS_, 27 * U), (31, 91, 255), a, 'mm')
    text(img, (px, H - 46 * U), T['demo'], font(FR_, 26 * U), (120, 135, 160), pulse(t, .8, 1.6, 8.0, 8.8), 'la')
    return img


def scene_final(t):
    p = .9; arr = R.sky_array(p); img = Image.fromarray(arr).convert('RGBA'); R.draw_sun(img, 1.0, p, .40, .40); rays(img, int(W * .80), int(H * .40), 1.0)
    a = np.array(img.convert('RGB')); a = R.draw_mountains(a, t + 60, p); img = Image.fromarray(a).convert('RGBA')
    R.draw_water(img, t + 20, p); FL6.draw(img, t + 10, light=False); R.draw_reeds(img, t); motes(img, t, 36, 21, (255, 236, 205), .8)
    fade_chips = 1 - seg(t, 4.6, 5.4)
    kinetic(img, (W / 2, H * .21), T['fin1'], font(FB, 90 * U), (255, 255, 255), t, .5, per=.03, hold=1 - seg(t, 4.8, 5.6))
    wd = [int(len(c) * 19 * U + 60 * U) for c in T['fchips']]; total = sum(wd) + 24 * U * (len(wd) - 1); x = (W - total) / 2
    for n, (c, w_) in enumerate(zip(T['fchips'], wd)):
        k = seg(t, 1.6 + n * .25, 2.1 + n * .25) * fade_chips; pill(img, x, H * .30, w_, 68 * U, (255, 255, 255), .92 * k); text(img, (x + w_ / 2, H * .30 + 34 * U), c, font(FS_, 32 * U), (11, 27, 58), k, 'mm'); x += w_ + 24 * U
    fin = seg(t, 4.6, 5.8)
    if fin > 0:
        img.alpha_composite(Image.new('RGBA', (W, H), (8, 20, 60, int(165 * fin)))); logo(img, W / 2, H * .30, 130 * U, fin)
        text(img, (W / 2, H * .45), 'WetlandEar', font(FB, 130 * U), (255, 255, 255), fin, 'mm'); text(img, (W / 2, H * .45 + 98 * U), T['tag'], font(FS_, 50 * U), (210, 228, 255), fin, 'mm')
        text(img, (W / 2, H * .66), 'wetlandear.tech   ·   app.wetlandear.tech', font(FS_, 44 * U), (255, 255, 255), seg(t, 5.2, 6.2), 'mm')
        text(img, (W / 2, H * .75), T['fin2'], font(FR_, 26 * U), (170, 195, 235), seg(t, 5.6, 6.6), 'mm')
        text(img, (W / 2, H * .80), T['photos'] + PH_CREDIT, font(FR_, 21 * U), (150, 175, 220), seg(t, 5.8, 6.8), 'mm')
    return img


SCENE_FN = [scene_dawn, scene_listen, scene_photos, scene_site, scene_app, scene_final]

# ---------------------------------------------------------------- transitions (A and B are RGB frames, p goes 0..1)
def rgb(x): return x.convert('RGB')


def tr_bars(A, Bm, p):
    a = np.asarray(rgb(A)); b = np.asarray(rgb(Bm)); n = 72; bw = W / n; col = (np.arange(W) / bw).astype(int)
    d = (np.arange(n) / n) * .34 + .10 * np.sin(np.arange(n) * 1.9) ** 2; hh = np.array([ease((p - d[i]) / .5) for i in range(n)]) * H
    hx = hh[col][None, :]; y = np.arange(H)[:, None]; m = y >= (H - hx)
    out = np.where(m[..., None], b, a).astype(np.float32); edge = (m & (y < (H - hx) + 10 * U)).astype(np.float32)[..., None]
    out = out * (1 - edge * .85) + 255 * edge * .85; return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def zoom(im, s):
    iw, ih = im.size; cw, ch = iw / s, ih / s; return im.crop((int((iw - cw) / 2), int((ih - ch) / 2), int((iw + cw) / 2), int((ih + ch) / 2))).resize((iw, ih), Image.BILINEAR)


def tr_zoomblur(A, Bm, p):
    k = ease(p); a = zoom(rgb(A), 1 + .35 * k).filter(ImageFilter.GaussianBlur(max(.1, 14 * U * math.sin(math.pi * min(1, k * 1.3)))))
    b = zoom(rgb(Bm), 1.3 - .3 * k).filter(ImageFilter.GaussianBlur(max(.1, 14 * U * (1 - k))))
    out = Image.blend(a, b, ease((p - .3) / .45)); fl = .38 * math.sin(math.pi * p) ** 2
    return Image.blend(out, Image.new('RGB', (W, H), (255, 255, 255)), fl)


def tr_push(A, Bm, p):
    k = ease(p); off = int(k * W); out = Image.new('RGB', (W, H)); out.paste(rgb(A), (-off, 0)); out.paste(rgb(Bm), (W - off, 0))
    lay = Image.new('RGBA', (int(90 * U), H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i in range(int(90 * U)): d.line((i, 0, i, H), fill=(11, 27, 58, int(70 * (1 - i / (90 * U)) ** 2)))
    o = out.convert('RGBA'); put(o, lay, W - off, 0); return o.convert('RGB')


_TRF = {}


def tr_flock(A, Bm, p):
    k = ease(p); xe = -200 * U + k * (W + 500 * U); a = np.asarray(rgb(A)).astype(np.float32); b = np.asarray(rgb(Bm)).astype(np.float32)
    x = np.arange(W)[None, :, None].astype(np.float32); w = np.clip((xe - x) / (260 * U) + .5, 0, 1); out = Image.fromarray(np.clip(a * (1 - w) + b * w, 0, 255).astype(np.uint8))
    if 'f' not in _TRF: _TRF['f'] = np.random.default_rng(3)
    o = out.convert('RGBA'); r = np.random.default_rng(12)
    for i in range(70):
        bx_ = xe + r.normal(0, 190 * U) - 40 * U; by_ = r.uniform(.06, .94) * H + 14 * U * math.sin(p * 9 + i); s_ = int(r.uniform(34, 96) * U // 4 * 4) or 8
        sp = R.sprite(int(((p * r.uniform(2.6, 4.0) * 3 + i * .37) % 1) * 16), False).resize((s_ * 2, s_), Image.BILINEAR); put(o, sp, bx_, by_)
    return o.convert('RGB')


def tr_flash(A, Bm, p):
    k = p
    if k < .5: out = Image.blend(zoom(rgb(A), 1 + .12 * ease(k * 2)), Image.new('RGB', (W, H), (255, 255, 255)), ease(k * 2))
    else: out = Image.blend(Image.new('RGB', (W, H), (255, 255, 255)), zoom(rgb(Bm), 1.1 - .1 * ease((k - .5) * 2)), ease((k - .5) * 2))
    return out


TRANS = [tr_bars, tr_zoomblur, tr_flock, tr_push, tr_flash]


def frame(i):
    t = i / FPS; s = max(k for k in range(len(B) - 1) if t >= B[k] - 1e-9); s = min(s, len(SCENE_FN) - 1)
    start = lambda k: B[k] - (TR if k > 0 else 0); lt = t - start(s); cur = SCENE_FN[s](lt)
    if s + 1 < len(SCENE_FN) and t > B[s + 1] - TR:
        p = (t - (B[s + 1] - TR)) / TR; nxt = SCENE_FN[s + 1](t - (B[s + 1] - TR)); cur = TRANS[s](cur, nxt, p)
    img = vignette(cur)
    if t < .9: img = Image.blend(Image.new('RGB', (W, H), (6, 14, 40)), img, ease(t / .9))
    if t > DUR - 1.4: img = Image.blend(img, Image.new('RGB', (W, H), (6, 14, 40)), ease((t - (DUR - 1.4)) / 1.4))
    return img


def main():
    if os.environ.get('FRAMES'):
        for s in os.environ['FRAMES'].split(','):
            frame(int(float(s) * FPS)).save(os.path.join(R.HERE, '_g_%s.png' % s.replace('.', '_'))); print('frame', s, flush=True)
        return
    out = os.path.join(R.HERE, 'intro2_silent_%s.mp4' % LANGV); n = int(DUR * FPS)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(n):
        pr.stdin.write(frame(i).tobytes())
        if i % 60 == 0: print('frame %d/%d' % (i, n), flush=True)
    pr.stdin.close(); pr.wait(); print('written', out, os.path.getsize(out) // 1024, 'KB')


if __name__ == '__main__':
    main()
