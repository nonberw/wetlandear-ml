"""Figures for the research report. Reads the real metrics of the shipped model (v5) and the real training folders; draws nothing from imagination.
Run with an environment that has matplotlib, numpy, scipy and soundfile:  python make_figures.py"""
import json, os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BASE = r'C:\Users\Timur\Documents\wetlandear-models'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs'); os.makedirs(OUT, exist_ok=True)
METRICS = os.environ.get('METRICS', BASE + r'\backup_v5\metrics_v5.json')
M = json.load(open(METRICS, encoding='utf-8'))
BLUE, LIGHT, GRAY, INK, ORANGE, GREEN, RED, SKY = '#1F5BFF', '#8AA4D6', '#55657F', '#0B1B3A', '#D98A1F', '#14804A', '#C2362F', '#E8EFFE'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#C9D3E6', 'axes.labelcolor': INK, 'xtick.color': GRAY, 'ytick.color': GRAY, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png')); plt.close(fig); print('fig', name)


def box(ax, x, y, w, h, text, fc=SKY, ec=BLUE, fs=8.2, bold=False, tc=INK, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.18', fc=fc, ec=ec, lw=1.3, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold' if bold else 'normal', linespacing=1.25)


def arrow(ax, a, b, color=BLUE, style='-|>', lw=1.4, rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw, connectionstyle='arc3,rad=%g' % rad, ls=ls))


# ---------- 1. system architecture ----------
def fig_arch():
    fig, ax = plt.subplots(figsize=(12.5, 7.2)); ax.set_xlim(0, 22); ax.set_ylim(0, 13); ax.axis('off')
    ax.text(2.5, 12.45, 'Источники звука', fontsize=10, fontweight='bold', color=BLUE, ha='center')
    box(ax, 0.4, 10.2, 4.2, 1.6, 'Рекордеры на станциях\n(файлы с карты\nпамяти)')
    box(ax, 0.4, 7.9, 4.2, 1.6, 'Телефон: запись\nс микрофона')
    box(ax, 0.4, 5.6, 4.2, 1.6, 'Архив старых записей\n(папка файлов)')
    ax.text(8.2, 12.45, 'Приложение (браузер)', fontsize=10, fontweight='bold', color=BLUE, ha='center')
    ax.add_patch(Rectangle((5.6, 4.2), 5.2, 7.9, fc='#F7F9FD', ec='#C9D3E6', lw=1))
    box(ax, 5.9, 10.2, 4.6, 1.6, 'Детектор событий\n(спектрограмма,\nпорог SNR)')
    box(ax, 5.9, 8.2, 4.6, 1.5, 'Журнал на устройстве\n(localStorage + IndexedDB)')
    box(ax, 5.9, 6.2, 4.6, 1.6, 'Режим без интернета:\nBirdNET + наш слой\nпрямо в браузере (LiteRT)', fc='#FFFFFF')
    box(ax, 5.9, 4.5, 4.6, 1.3, 'Интерфейс: проверка,\nкарта, отчёты,\nуправление', fc='#FFFFFF')
    ax.text(14.2, 12.45, 'Сервер (Vercel)', fontsize=10, fontweight='bold', color=BLUE, ha='center')
    ax.add_patch(Rectangle((11.9, 4.2), 4.6, 7.9, fc='#F7F9FD', ec='#C9D3E6', lw=1))
    box(ax, 12.1, 10.3, 4.2, 1.5, '/api/identify\nBirdNET + наш слой\n+ YAMNet')
    box(ax, 12.1, 8.6, 4.2, 1.4, '/api/alert, /api/morning\n(тревоги, отчёты)', fc='#FFFFFF')
    box(ax, 12.1, 6.9, 4.2, 1.4, '/api/telegram, /api/mcp\n(бот, ассистенты)', fc='#FFFFFF')
    box(ax, 12.1, 5.2, 4.2, 1.4, '/api/public, /api/expert\n(страницы по ссылке)', fc='#FFFFFF')
    ax.text(19.4, 12.45, 'База (Supabase)', fontsize=10, fontweight='bold', color=BLUE, ha='center')
    ax.add_patch(Rectangle((17.2, 4.2), 4.4, 7.9, fc='#F7F9FD', ec='#C9D3E6', lw=1))
    box(ax, 17.4, 9.9, 4.0, 1.9, 'Postgres\nсобытия, записи,\nстанции, команды,\nзадачи', fc='#FFFFFF')
    box(ax, 17.4, 7.9, 4.0, 1.5, 'Storage\nаудиоклипы (приватно)', fc='#FFFFFF')
    box(ax, 17.4, 5.9, 4.0, 1.5, 'Auth\nвход через Google', fc='#FFFFFF')
    box(ax, 17.4, 4.4, 4.0, 1.1, 'Правила доступа (RLS)', fc='#FFFFFF', fs=8.5)
    for y in (11.0, 8.7, 6.4):
        arrow(ax, (4.7, y), (5.8, 11.0 if y > 10 else 9.0 if y > 8 else 7.0))
    arrow(ax, (10.6, 10.9), (12.0, 10.9)); arrow(ax, (16.4, 10.9), (17.3, 10.9)); arrow(ax, (10.6, 8.9), (12.0, 9.3), rad=0.1)
    ax.text(0.4, 3.4, 'Выходы', fontsize=11, fontweight='bold', color=BLUE)
    box(ax, 0.4, 1.6, 4.6, 1.3, 'Telegram-бот:\nтревоги и отчёты', fs=8.2)
    box(ax, 5.6, 1.6, 4.6, 1.3, 'ИИ-ассистенты (MCP):\nClaude и другие', fs=8.2)
    box(ax, 10.8, 1.6, 4.6, 1.3, 'Публичные страницы,\nQR, отчёты спонсорам', fs=8.2)
    box(ax, 16.0, 1.6, 5.6, 1.3, 'Экспорт для науки:\nGBIF, eBird, PDF', fs=8.2)
    for x in (2.7, 7.9, 13.1, 18.8): arrow(ax, (x, 4.1), (x, 3.0), color=GRAY, rad=0.0)
    ax.text(11, 0.5, 'Приложение работает и без входа: журнал хранится на устройстве, вход нужен для синхронизации, сервера распознавания и команд.', ha='center', fontsize=8.5, color=GRAY)
    save(fig, 'f_architecture')


# ---------- 2. ML pipeline ----------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.9)); ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.axis('off')
    box(ax, 0.3, 6.6, 3.8, 2.6, 'Данные\nXeno-canto (API v3)\niNaturalist\nотбор лицензий', fs=8.5)
    box(ax, 4.9, 6.6, 3.8, 2.6, 'Окна по 3 секунды\n14 самых громких\n+ 3 фоновых\nна запись', fs=8.5)
    box(ax, 9.5, 6.6, 3.8, 2.6, 'BirdNET V2.4\nотпечаток 1024 числа\n+ баллы 6522 меток', fs=8.5)
    box(ax, 14.1, 6.6, 3.8, 2.6, 'Шумные копии\nкамыш, ветер, вода,\nдождь, чужие птицы', fs=8.5, fc='#FFFFFF')
    box(ax, 18.7, 6.6, 3.0, 2.6, 'Только\nотпечатки\n(звук не нужен)', fs=8.5, fc='#FFFFFF')
    for x in (4.2, 8.8, 13.4, 18.0): arrow(ax, (x, 7.9), (x + 0.6, 7.9))
    box(ax, 0.3, 2.4, 4.9, 2.6, 'Логистическая регрессия\n268 видов + «фон»\nбаланс классов, C = 2', fs=8.5)
    box(ax, 6.2, 2.4, 4.9, 2.6, 'Проверка:\n5 частей, разбиение\nпо ЗАПИСЯМ', fs=8.5)
    box(ax, 12.1, 2.4, 4.9, 2.6, 'Смешивание с баллами\nBirdNET (вес 0,15)\n+ температура', fs=8.5)
    box(ax, 18.0, 2.4, 3.7, 2.6, 'Экспорт\nhead.npz (сервер)\nhead_web (браузер)', fs=8.5)
    for x in (5.3, 11.2, 17.1): arrow(ax, (x - 0.15, 3.7), (x + 0.95, 3.7))
    arrow(ax, (20.2, 6.5), (3.0, 5.1), color=GRAY, rad=-0.18, ls='--')
    ax.text(11, 0.7, 'Нейросеть BirdNET не дообучается: обучается только маленький слой поверх её «отпечатков» (около 500 КБ).', ha='center', fontsize=9, color=GRAY)
    save(fig, 'f_pipeline')


# ---------- 3. real spectrogram with the windows ----------
def fig_spectro():
    import soundfile as sf
    from scipy.signal import resample_poly, spectrogram
    f = sorted(glob.glob(BASE + r'\train_audio\Grus_grus\XC*.mp3'))[3]
    x, sr = sf.read(f, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    x = resample_poly(x, 48000, sr).astype('float32'); n = 144000
    ws = [(i, x[i:i + n]) for i in range(0, len(x) - n + 1, n // 2)]
    en = np.array([float((w ** 2).mean()) for _, w in ws]); order = np.argsort(en)[::-1]
    strong = set(order[:min(14, len(ws))]); weak = set(order[-3:]) if len(ws) > 8 else set()
    fr, t, S = spectrogram(x, 48000, nperseg=1024, noverlap=768); S = 10 * np.log10(S + 1e-12)
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 5.6), gridspec_kw={'height_ratios': [1, 2.4]}, sharex=True)
    tt = np.arange(len(x)) / 48000
    ax[0].plot(tt[::40], x[::40], color=BLUE, lw=.4); ax[0].set_ylabel('амплитуда'); ax[0].set_yticks([])
    ax[1].imshow(S[(fr < 10000)], aspect='auto', origin='lower', extent=[0, t[-1], 0, 10], cmap='Blues', vmin=np.percentile(S, 40), vmax=np.percentile(S, 99.5))
    ax[1].set_ylabel('частота, кГц'); ax[1].set_xlabel('время, секунды')
    for k, (i, _) in enumerate(ws):
        col = ORANGE if k in strong else (GRAY if k in weak else None)
        if col: ax[1].add_patch(Rectangle((i / 48000, 0.15), 3, 9.7, fill=False, ec=col, lw=1.1 if k in strong else 1.4, ls='-' if k in strong else '--'))
        if k in strong: ax[0].axvspan(i / 48000, i / 48000 + 3, color=ORANGE, alpha=.08)
    ax[1].set_title('Настоящая запись серого журавля (Xeno-canto %s): оранжевые рамки — окна, которые идут в обучение, серый пунктир — фоновые' % os.path.basename(f)[:-4], fontsize=8.8, color=GRAY, loc='left')
    save(fig, 'f_windows')


# ---------- 4. data ----------
def counts():
    xc, inat, other = {}, {}, {}
    for d in os.listdir(BASE + r'\train_audio'):
        for fn in os.listdir(os.path.join(BASE, 'train_audio', d)):
            if re.match(r'^XC\d+\.', fn): xc[d] = xc.get(d, 0) + 1
            elif re.match(r'^INAT\d+\.', fn): inat[d] = inat.get(d, 0) + 1
            else: other[d] = other.get(d, 0) + 1
    return xc, inat, other


def fig_data():
    xc, inat, other = counts()
    per = M['per_species']; rec = np.array([r['recordings'] for r in per])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.15, 1]})
    bins = [2, 5, 10, 20, 30, 40, 60, 100]
    h = [int(((rec >= bins[i]) & (rec < bins[i + 1])).sum()) for i in range(len(bins) - 1)] + [int((rec >= bins[-1]).sum())]
    lab = ['2–4', '5–9', '10–19', '20–29', '30–39', '40–59', '60–99', '100+']
    cols = [RED if i < 2 else ORANGE if i == 2 else BLUE for i in range(len(h))]
    ax[0].bar(lab, h, color=cols)
    for i, v in enumerate(h): ax[0].text(i, v + 1, str(v), ha='center', fontsize=9, color=INK)
    ax[0].set_xlabel('число записей вида в обучении'); ax[0].set_ylabel('число видов'); ax[0].set_title('Сколько видов сколько записей имеют (v5, %d видов)' % len(rec), fontsize=10, loc='left')
    tot = sorted(xc.items(), key=lambda kv: -kv[1])[:0]
    s_xc, s_in, s_ot = sum(xc.values()), sum(inat.values()), sum(other.values())
    ax[1].barh(['Xeno-canto\n(в v5)', 'iNaturalist\n(добавлено для v6)', 'ручной этап\n(Commons и др.)'], [s_xc, s_in, s_ot], color=[BLUE, LIGHT, GRAY])
    for i, v in enumerate([s_xc, s_in, s_ot]): ax[1].text(v + 60, i, '%d' % v, va='center', fontsize=9)
    ax[1].set_xlabel('число файлов'); ax[1].set_title('Откуда файлы', fontsize=10, loc='left'); ax[1].set_xlim(0, s_xc * 1.18)
    fig.tight_layout(); save(fig, 'f_data')
    return {'xc': s_xc, 'inat': s_in, 'other': s_ot, 'species_inat': len(inat)}


# ---------- 5. versions ----------
def fig_versions():
    v = ['v1', 'v2', 'v3', 'v4', 'v5']; sp = [24, 123, 213, 222, 268]; rc = [267, 1706, 2960, 6214, 7822]
    bn = [68.8, 59.5, 56.9, 58.2, 56.3]; hd = [78.6, 74.4, 71.5, 74.7, 74.2]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(5); w = .36
    ax[0].bar(x - w / 2, bn, w, color=LIGHT, label='BirdNET: расчёт проекта (с фильтром места)'); ax[0].bar(x + w / 2, hd, w, color=BLUE, label='слой WetlandEar')
    for i in range(5): ax[0].text(x[i] - w / 2, bn[i] + .8, '%.1f' % bn[i], ha='center', fontsize=8); ax[0].text(x[i] + w / 2, hd[i] + .8, '%.1f' % hd[i], ha='center', fontsize=8)
    fair = M['robustness']['summary']['clean']['birdnet_restricted'] * 100
    ax[0].plot([4 - w, 4], [fair, fair], color=ORANGE, lw=3, label='v5: BirdNET без фильтра места, на нашем списке (%.1f)' % fair)
    ax[0].set_xticks(x); ax[0].set_xticklabels(v); ax[0].set_ylim(40, 92); ax[0].set_ylabel('точность по 3-секундным окнам, %'); ax[0].legend(frameon=False, fontsize=7.4, loc='upper right')
    ax[0].set_title('Точность по версиям (проверка по записям)', fontsize=10, loc='left')
    ax[1].plot(v, sp, 'o-', color=BLUE, label='видов'); ax[1].set_ylabel('число видов', color=BLUE)
    a2 = ax[1].twinx(); a2.plot(v, rc, 's--', color=ORANGE, label='записей'); a2.set_ylabel('число записей', color=ORANGE); a2.spines['right'].set_visible(True)
    ax[1].set_title('Рост набора данных', fontsize=10, loc='left')
    fig.tight_layout(); save(fig, 'f_versions')


# ---------- 6. robustness ----------
def fig_noise():
    R = M['robustness']; bt = R['by_noise_type']; bs = R['by_snr']
    names = {'reeds': 'камыш', 'wind': 'ветер', 'water': 'вода', 'rain': 'дождь', 'bg': 'фон поля', 'chorus': 'чужие птицы', 'mix': 'смесь'}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.3, 1]})
    x = np.arange(len(bt)); w = .27
    ax[0].bar(x - w, [r['birdnet'] * 100 for r in bt], w, color=LIGHT, label='BirdNET'); ax[0].bar(x, [r['clean_only'] * 100 for r in bt], w, color='#5B86F2', label='слой, обучен на чистых')
    ax[0].bar(x + w, [r['with_noise'] * 100 for r in bt], w, color=BLUE, label='слой, обучен с шумом')
    ax[0].set_xticks(x); ax[0].set_xticklabels([names.get(r['type'], r['type']) for r in bt], fontsize=9); ax[0].set_ylabel('точность, %'); ax[0].set_ylim(30, 85); ax[0].legend(frameon=False, fontsize=8, ncol=1, loc='upper right')
    ax[0].set_title('Точность на шумных окнах по типу шума', fontsize=10, loc='left')
    lab = ['%d…%d' % (r['lo'], r['hi']) for r in bs]
    ax[1].plot(lab, [r['birdnet'] * 100 for r in bs], 'o-', color=LIGHT, label='BirdNET'); ax[1].plot(lab, [r['clean_only'] * 100 for r in bs], 'o-', color='#5B86F2', label='обучен на чистых'); ax[1].plot(lab, [r['with_noise'] * 100 for r in bs], 'o-', color=BLUE, label='обучен с шумом')
    ax[1].set_xlabel('отношение сигнал/шум, дБ (слева шумнее)'); ax[1].set_ylabel('точность, %'); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title('Точность и уровень шума', fontsize=10, loc='left')
    fig.tight_layout(); save(fig, 'f_noise')


# ---------- 7. calibration ----------
def fig_calibration():
    cal = M['calibration']; cov = M['coverage']; covn = M['coverage_noisy']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    mid = [(r['lo'] + r['hi']) / 2 for r in cal]; right = [r['right'] for r in cal]; share = [r['share'] for r in cal]
    ax[0].plot([0, 1], [0, 1], '--', color=GRAY, lw=1, label='идеальная калибровка')
    ax[0].plot(mid, right, 'o-', color=BLUE, label='наша модель')
    for m_, r_, s_ in zip(mid, right, share): ax[0].text(m_, r_ + .04, '%d%% окон' % round(s_ * 100), fontsize=7.5, ha='center', color=GRAY)
    ax[0].set_xlabel('уверенность, которую выдала модель'); ax[0].set_ylabel('доля верных ответов'); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.08); ax[0].legend(frameon=False, fontsize=8.5, loc='lower right')
    ax[0].set_title('Можно ли верить процентам', fontsize=10, loc='left')
    ax[1].plot([c['t'] for c in cov], [c['precision'] * 100 for c in cov], 'o-', color=BLUE, label='точность (чистые окна)'); ax[1].plot([c['t'] for c in covn], [c['precision'] * 100 for c in covn], 'o-', color=LIGHT, label='точность (шумные окна)')
    a2 = ax[1].twinx(); a2.plot([c['t'] for c in cov], [c['covers'] * 100 for c in cov], 's--', color=ORANGE, label='доля окон с ответом'); a2.spines['right'].set_visible(True); a2.set_ylabel('доля окон с ответом, %', color=ORANGE)
    ax[1].set_xlabel('порог уверенности'); ax[1].set_ylabel('точность, %', color=BLUE); ax[1].set_title('Порог: точнее, но отвечаем реже', fontsize=10, loc='left')
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.8, loc='lower left')
    fig.tight_layout(); save(fig, 'f_calibration')


# ---------- 8. per species ----------
def fig_species():
    P = [r for r in M['per_species']]
    masked = set(json.load(open(os.path.join(OUT, 'masked.json')))) if os.path.exists(os.path.join(OUT, 'masked.json')) else set()
    inb = [r for r in P if r['in_birdnet'] and r['sci'] not in masked]; mk = [r for r in P if r['in_birdnet'] and r['sci'] in masked]; notb = [r for r in P if not r['in_birdnet']]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={'width_ratios': [1, 1.05]})
    ax[0].plot([0, 100], [0, 100], '--', color=GRAY, lw=1)
    sc = ax[0].scatter([r['birdnet'] * 100 for r in inb], [r['head'] * 100 for r in inb], c=[np.log10(r['recordings']) for r in inb], cmap='Blues', edgecolor=BLUE, lw=.4, s=26, vmin=.2, vmax=2.4)
    ax[0].scatter([-2 for r in mk], [r['head'] * 100 for r in mk], marker='^', color=GRAY, s=24, label='BirdNET знает, но фильтр места обнулил (%d)' % len(mk))
    ax[0].scatter([-5 for r in notb], [r['head'] * 100 for r in notb], marker='D', color=ORANGE, s=22, label='вида нет в BirdNET (%d)' % len(notb))
    worse = sorted(inb, key=lambda r: r['head'] - r['birdnet'])[:4]
    for r in worse: ax[0].annotate(r['en'][:20], (r['birdnet'] * 100, r['head'] * 100), fontsize=6.8, color=RED, xytext=(3, -8), textcoords='offset points')
    ax[0].set_xlim(-9, 102); ax[0].set_xlabel('точность BirdNET по виду, % (расчёт проекта)'); ax[0].set_ylabel('точность нашего слоя по виду, %'); ax[0].legend(frameon=False, fontsize=7.3, loc='lower right')
    ax[0].set_title('Каждая точка — вид (выше диагонали — слой точнее)', fontsize=9.5, loc='left')
    cb = fig.colorbar(sc, ax=ax[0], fraction=.04, pad=.02); cb.set_label('записей (log10)', fontsize=8)
    rec = np.array([r['recordings'] for r in P]); acc = np.array([r['head'] * 100 for r in P])
    ax[1].scatter(rec, acc, s=18, color=BLUE, alpha=.65, edgecolor='none'); ax[1].set_xscale('log')
    edges = [2, 5, 10, 20, 40, 80, 400]
    for i in range(len(edges) - 1):
        m = (rec >= edges[i]) & (rec < edges[i + 1])
        if m.sum(): ax[1].plot([edges[i], edges[i + 1] - .01], [np.median(acc[m])] * 2, color=ORANGE, lw=2.2)
    ax[1].set_xlabel('записей этого вида в обучении (лог. шкала)'); ax[1].set_ylabel('точность по виду, %'); ax[1].set_title('Больше записей — выше точность (оранжевое — медиана)', fontsize=9.5, loc='left')
    fig.tight_layout(); save(fig, 'f_species')
    d = np.array([r['head'] - r['birdnet'] for r in inb])
    return {'better': int((d > .005).sum()), 'equal': int((np.abs(d) <= .005).sum()), 'worse': int((d < -.005).sum()), 'notb': len(notb), 'masked': len(mk)}


# ---------- 9. lineage ----------
def fig_lineage():
    fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 28.6); ax.set_ylim(0, 11.5); ax.axis('off')
    xs = [0.3, 5.1, 9.9, 14.7, 19.5]
    V = [('v1', '24 вида\n267 записей\n78,6% против 68,8%'), ('v2', '123 вида\n1 706 записей\n74,4% против 59,5%'), ('v3', '213 видов\n2 960 записей\n71,5% против 56,9%\n+ калибровка'),
         ('v4', '222 вида\n6 214 записей\n74,7% против 58,2%\n+ шумовые копии'), ('v5', '268 видов\n7 822 записи\n74,2% против 56,3%\n(в приложении)')]
    for x, (n, t) in zip(xs, V):
        box(ax, x, 6.4, 4.2, 3.6, n + '\n' + t, fs=8.4, fc=SKY if n != 'v5' else '#DCE8FF')
    for a_, b_ in zip(xs[:-1], xs[1:]): arrow(ax, (a_ + 4.25, 8.2), (b_ - 0.05, 8.2), lw=2)
    box(ax, 24.6, 6.4, 3.8, 3.6, 'v6\nвыложена:\n+2 042 записи\niNaturalist, +10 п.п.\nна полевых записях', fs=8.4, fc='#FFFFFF', ls='--', ec=ORANGE)
    arrow(ax, (23.75, 8.2), (24.55, 8.2), color=ORANGE, ls='--', lw=1.8)
    rej = [(3.0, 'Нейросеть MLP:\n56,5% — хуже\nлогистической\nрегрессии (v2)'), (6.9, 'Отпечатки YAMNet\nв добавок: 72,4%\nпротив 72,5%,\nпользы нет (v2)'), (10.8, 'Баллы BirdNET как\nпризнаки (stacking):\nлучше по записи,\nне по окну (v2)')]
    for x, t in rej:
        box(ax, x, 1.2, 3.6, 3.4, '✕  отвергнуто\n' + t, fs=7.8, fc='#F4F6FA', ec=GRAY, tc=GRAY, ls='--')
        arrow(ax, (7.2, 6.3), (x + 1.8, 4.7), color=GRAY, ls='--', lw=1.0)
    box(ax, 19.5, 1.2, 4.2, 3.4, '✕  отвергнуто\nСвой вес BirdNET\nна каждый вид:\n71,2% → 70,9%,\nне помогло (v5)', fs=7.8, fc='#F4F6FA', ec=GRAY, tc=GRAY, ls='--')
    arrow(ax, (21.6, 6.3), (21.6, 4.7), color=GRAY, ls='--', lw=1.0)
    ax.text(0.3, 10.9, 'Ветви развития: что вошло в основную линию (синее) и что пробовали и отвергли (серое)', fontsize=10.5, fontweight='bold', color=INK)
    save(fig, 'f_lineage')


# ---------- 10. alerts ----------
def fig_alerts():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={'width_ratios': [1, 1.3]})
    ax[0].bar(['выстрелы\n(3 записи)', 'двигатели,\nпилы (порог 0,5)', 'двигатели,\nпилы (порог 0,3)'], [37.5, 33.9, 48.2], color=[LIGHT, LIGHT, BLUE])
    for i, v in enumerate([37.5, 33.9, 48.2]): ax[0].text(i, v + 1, '%.1f%%' % v, ha='center', fontsize=9)
    ax[0].set_ylabel('доля окон, поднявших тревогу, %'); ax[0].set_ylim(0, 62); ax[0].set_title('Что ловится (очень малая выборка)', fontsize=10, loc='left')
    s = ['птицы (1 169 окон)', 'гроза (158)', 'дождь (80)', 'ветер (69)']; g = [0.2, 1.9, 0, 0]; e = [0, 0, 0, 0]
    x = np.arange(4); ax[1].bar(x - .18, g, .36, color=ORANGE, label='ложная «стрельба»'); ax[1].bar(x + .18, [0.0] * 4, .36, color=BLUE, label='ложный «двигатель»')
    for i, v in enumerate(g): ax[1].text(i - .18, v + .06, '%.1f%%' % v, ha='center', fontsize=8.5)
    for i in range(4): ax[1].text(i + .18, .06, '0%', ha='center', fontsize=8.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(s, fontsize=8.8); ax[1].set_ylabel('ложные тревоги, % окон'); ax[1].set_ylim(0, 2.6); ax[1].legend(frameon=False, fontsize=8.5); ax[1].set_title('Что тревожит зря', fontsize=10, loc='left')
    fig.tight_layout(); save(fig, 'f_alerts')


# ---------- 11. database ----------
def fig_db():
    fig, ax = plt.subplots(figsize=(12, 6.4)); ax.set_xlim(0, 24); ax.set_ylim(0, 12.2); ax.axis('off')
    groups = [('Люди и команды', 0.2, [('profiles', 'имя, организация'), ('organizations', 'заповедник, филиал'), ('org_members', 'роль в команде'), ('shared_prefs', 'индикаторные виды,\nзаметки станций')]),
              ('Наблюдения', 6.2, [('stations', 'название, координаты'), ('recordings', 'запись, индексы звука'), ('detections', 'событие: вид, статус,\nуверенность'), ('review_log', 'кто и когда подтвердил')]),
              ('Подключения', 12.2, [('telegram_links', 'чат бота, тревоги,\nсписок наблюдения'), ('api_tokens', 'ключи ИИ-ассистентов\n(хранится хеш)'), ('training_shares', 'добровольные отпечатки\nдля обучения')]),
              ('Управление', 18.2, [('tasks', 'задачи рейнджерам'), ('expert_requests', 'ссылка эксперту, ответ'), ('public_pages', 'публичные страницы\nи отчёты')])]
    for title, x, items in groups:
        ax.text(x + 2.8, 11.3, title, ha='center', fontsize=11, fontweight='bold', color=BLUE)
        for i, (n, d) in enumerate(items):
            ax.add_patch(FancyBboxPatch((x, 8.5 - i * 2.35), 5.6, 2.0, boxstyle='round,pad=0.02,rounding_size=0.18', fc=SKY if i % 2 == 0 else '#FFFFFF', ec=BLUE, lw=1.3))
            ax.text(x + 2.8, 9.9 - i * 2.35, n, ha='center', va='center', fontsize=9.2, fontweight='bold', color=INK)
            ax.text(x + 2.8, 9.15 - i * 2.35, d, ha='center', va='center', fontsize=8, color=GRAY, linespacing=1.2)
    ax.text(12, 0.5, 'У каждой таблицы включены правила доступа (RLS): человек видит свои строки и строки коллег по организации;\nпубличные страницы и страница эксперта читаются только через сервер.', ha='center', fontsize=8.8, color=GRAY, linespacing=1.3)
    save(fig, 'f_database')


if __name__ == '__main__':
    fig_arch(); fig_pipeline(); fig_spectro(); stats = fig_data(); fig_versions(); fig_noise(); fig_calibration(); sp = fig_species(); fig_lineage(); fig_alerts(); fig_db()
    json.dump({'data': stats, 'species': sp}, open(os.path.join(OUT, 'stats.json'), 'w'), ensure_ascii=False)
    print(stats, sp)
