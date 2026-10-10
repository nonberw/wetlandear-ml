"""Soundtrack v2 for the intro video, synthesised from scratch (no samples, so no licences): 96 BPM, Am - F - C - G, a plucked arpeggio, a bell melody, pad, bass,
soft drums that grow with the scenes, risers and low hits on the scene changes, and a reverb. Writes intro2.wav (44.1 kHz stereo)."""
import os, wave
import numpy as np

SR = 44100; HERE = os.path.dirname(os.path.abspath(__file__))
BPM = 96.0; BEAT = 60 / BPM; BAR = 4 * BEAT                      # 2.5 s per bar
DUR = float(os.environ.get('DUR', '50')); N = int(SR * DUR)
SCENES = [0, 10, 17.5, 27.5, 35, 42.5, 50]                      # same boundaries as render2.py
rng = np.random.default_rng(7)
t_all = np.arange(N) / SR


def bus(): return np.zeros(N), np.zeros(N)


mel, mer = bus(); padl, padr = bus(); drl, drr = bus(); fxl, fxr = bus()   # melody+plucks (to reverb), pad, drums (dry), effects


def add(L, R, a, sig, pan=0.0, gain=1.0):
    if a < 0: sig = sig[-a:]; a = 0
    n = min(len(sig), N - a)
    if n <= 0: return
    L[a:a + n] += sig[:n] * gain * (1 - pan) / 2 * 2 ** .5 * .707
    R[a:a + n] += sig[:n] * gain * (1 + pan) / 2 * 2 ** .5 * .707


def hz(name):
    n = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}; k = n[name[0]] + (1 if '#' in name else 0); o = int(name[-1])
    return 440.0 * 2 ** ((k + 12 * (o + 1) - 69) / 12)


CH = [('A2', ['A3', 'C4', 'E4'], ['A4', 'C5', 'E5', 'A5']), ('F2', ['F3', 'A3', 'C4'], ['F4', 'A4', 'C5', 'F5']), ('C3', ['E3', 'G3', 'C4'], ['G4', 'C5', 'E5', 'G5']), ('G2', ['D3', 'G3', 'B3'], ['G4', 'B4', 'D5', 'G5'])]


def pluck(f, dur=1.1, amp=1.0):
    n = int(dur * SR); x = np.arange(n) / SR; s = np.zeros(n)
    for k in range(1, 9):
        s += np.sin(2 * np.pi * f * k * x + k) * np.exp(-x * (3.2 + 2.2 * k)) / k ** 1.15
    return s * amp * np.minimum(x / .004, 1)


def bell(f, dur=2.2, amp=1.0):
    n = int(dur * SR); x = np.arange(n) / SR
    mod = np.sin(2 * np.pi * f * 3.5 * x) * 2.4 * np.exp(-x * 4.0)
    s = np.sin(2 * np.pi * f * x + mod) * np.exp(-x * 1.5) + .35 * np.sin(2 * np.pi * f * 2 * x + mod * .5) * np.exp(-x * 2.4)
    return s * amp * np.minimum(x / .006, 1)


def pad_note(f, dur, amp=1.0):
    n = int(dur * SR); x = np.arange(n) / SR; s = np.zeros(n)
    for det in (-.9, 0, .8):
        ph = rng.uniform(0, 6.28)
        s += (np.sin(2 * np.pi * (f + det) * x + ph) + .45 * np.sin(2 * np.pi * 2 * (f + det) * x + ph) + .18 * np.sin(2 * np.pi * 3 * (f + det) * x + ph)) / 3
    env = np.minimum(x / (BAR * .55), 1) * np.minimum((dur - x) / (BAR * .6), 1)
    return s * env * amp


def bass_note(f, dur, amp=1.0):
    n = int(dur * SR); x = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * x) + .25 * np.sin(2 * np.pi * 2 * f * x)) * np.exp(-x * 1.3) * np.minimum(x / .01, 1) * amp


def kick(amp=1.0):
    n = int(.42 * SR); x = np.arange(n) / SR; f = 48 + 90 * np.exp(-x * 24)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 9) * amp + .15 * np.random.default_rng(1).standard_normal(n) * np.exp(-x * 90) * amp


def clap(amp=1.0):
    n = int(.3 * SR); x = np.arange(n) / SR; z = np.random.default_rng(2).standard_normal(n)
    z = np.diff(z, prepend=0) * .7 + z * .3
    env = np.exp(-x * 22) + .6 * np.exp(-np.maximum(x - .012, 0) * 30) * (x > .012) + .5 * np.exp(-np.maximum(x - .026, 0) * 30) * (x > .026)
    return z * env * amp * .5


def hat(amp=1.0, open_=False):
    n = int((.25 if open_ else .06) * SR); x = np.arange(n) / SR; z = np.diff(np.random.default_rng(3).standard_normal(n + 1))
    return z * np.exp(-x * (14 if open_ else 70)) * amp * .5


def riser(a, dur, amp=1.0):
    n = int(dur * SR); x = np.arange(n) / SR; u = x / dur
    z = np.diff(rng.standard_normal(n + 1)) * (u ** 2.2) * .5
    sw = np.sin(2 * np.pi * np.cumsum(180 * 2 ** (u * 3.6)) / SR) * u ** 2 * .35
    return (z + sw) * amp


def boom(amp=1.0):
    n = int(2.2 * SR); x = np.arange(n) / SR; f = 38 + 60 * np.exp(-x * 6)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 1.7)
    z = np.diff(np.random.default_rng(4).standard_normal(n + 1)) * np.exp(-x * 3.2) * .35
    return (s + z) * amp


def chirp(f0, f1, dur, vib=18):
    n = int(dur * SR); x = np.arange(n) / SR; u = x / dur; f = f0 + (f1 - f0) * u ** 1.4 + 90 * np.sin(2 * np.pi * vib * x)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * u) ** 1.5


bars = int(DUR / BAR)
# ---------------- pad and bass (every bar), chords follow Am F C G; the pad is strong in the quiet parts and ducks under the kick
for b in range(bars):
    bass, notes, arp = CH[b % 4]; a = int(b * BAR * SR)
    lvl = .55 if b < 4 else .42
    for nt in notes: add(padl, padr, a, pad_note(hz(nt), BAR * 1.15), pan=rng.uniform(-.5, .5), gain=.07 * lvl)
    if b >= 2: add(drl, drr, a, bass_note(hz(bass), BAR * .98), 0, .30 if b >= 4 else .16)
    if b >= 4:
        for q in (2, 3): add(drl, drr, a + int((q + .5) * BEAT * SR), bass_note(hz(bass), BEAT * .6), 0, .12)
# ---------------- arpeggio: 8th-note plucks, from bar 1, richer from bar 7
for b in range(1, bars):
    arp = CH[b % 4][2]; pat = [0, 1, 2, 3, 2, 1, 2, 3] if b >= 7 else [0, 2, 1, 3, 0, 2, 1, 3]
    for i, k in enumerate(pat):
        if b < 4 and i % 2: continue
        a = int((b * BAR + i * BEAT / 2) * SR); add(mel, mer, a, pluck(hz(arp[k]), 1.0), pan=(-.4 if i % 2 else .4), gain=.10 if b >= 7 else .075)
# ---------------- bell melody for the photo, site, app and final scenes (A minor pentatonic), two-bar phrases
PH = [  # (beat within the 2-bar phrase, note, beats)
    [(0, 'E5', 1.5), (2, 'G5', 1), (3, 'A5', 1.5), (5, 'G5', 1), (6, 'E5', 2)],
    [(0, 'D5', 1.5), (2, 'E5', 1), (3, 'G5', 1.5), (5, 'E5', 1), (6, 'D5', 2)],
    [(0, 'C5', 1.5), (2, 'E5', 1), (3, 'G5', 1.5), (5, 'A5', 1), (6, 'G5', 2)],
    [(0, 'B4', 1.5), (2, 'D5', 1), (3, 'G5', 1.5), (5, 'E5', 1), (6, 'D5', 2)]]
for k, b in enumerate(range(7, 17, 2)):
    for beat, nt, dur in PH[k % 4]:
        a = int((b * BAR + beat * BEAT) * SR); add(mel, mer, a, bell(hz(nt), 2.4), pan=.15, gain=.11)
# the finale: a high octave line over the last chords
for beat, nt in ((0, 'A5'), (2, 'E5'), (4, 'C5'), (6, 'E5'), (8, 'A5'), (12, 'E6')):
    a = int((17 * BAR + beat * BEAT) * SR); add(mel, mer, a, bell(hz(nt), 3.5), pan=-.1, gain=.10)
# ---------------- drums: a heartbeat in the listening scene, a groove from the photo scene, a snare roll before the finale
kicks = []
for b in range(bars):
    for q in range(4):
        tt = b * BAR + q * BEAT
        on = (b >= 2 and q == 0 and b < 4) or (4 <= b < 7 and q in (0, 2) and False) or (7 <= b < 17 and q in (0, 2)) or (b == 4 or b == 5) and q == 0 or (b == 6 and q in (0, 2, 3)) or b == 17 and q == 0
        if on: add(drl, drr, int(tt * SR), kick(), 0, .75 if b >= 7 else .5); kicks.append(tt)
for b in range(7, 17):
    for q in (1, 3): add(drl, drr, int((b * BAR + q * BEAT) * SR), clap(), .1, .30)
    for i in range(8): add(drl, drr, int((b * BAR + i * BEAT / 2) * SR), hat(1.0, open_=(i % 4 == 3)), -.2 + .05 * i, .16 if i % 2 else .11)
for b in range(4, 7):
    for i in range(8): add(drl, drr, int((b * BAR + i * BEAT / 2) * SR), hat(.8), .15, .07)
for i in range(16): add(drl, drr, int((15.5 * BAR + i * BEAT / 4) * SR), clap(.7 + i * .03), 0, .20)   # roll into the app scene finale
# ---------------- risers into each scene change and a low hit on the change itself
for sb in SCENES[1:-1]:
    add(fxl, fxr, int((sb - 2.4) * SR), riser(0, 2.4), 0, .22)
    add(fxl, fxr, int(sb * SR), boom(), 0, .55); add(drl, drr, int(sb * SR), kick(1.0), 0, .6)
add(fxl, fxr, int(0.2 * SR), riser(0, 1.6), 0, .12)
# ---------------- wind and a few birds under the landscape scenes
def lowpass_noise(cut):
    f = np.fft.rfft(rng.standard_normal(N)); fr = np.fft.rfftfreq(N, 1 / SR); f *= 1 / (1 + (fr / cut) ** 2) ** 1.5
    x = np.fft.irfft(f, N); return x / np.abs(x).max()
wl, wr = lowpass_noise(420), lowpass_noise(360)
wg = np.interp(t_all, [0, 1.5, 8.5, 10, 42, 43, DUR - 3, DUR], [0, .9, .9, .1, .1, .8, .6, 0]) * (.5 + .5 * np.sin(2 * np.pi * .06 * t_all + 1))
fxl += wl * .06 * wg; fxr += wr * .06 * wg
for t0, f0, f1, pan in ((1.6, 3200, 4300, -.5), (1.85, 3500, 4600, -.5), (3.3, 4300, 5400, .6), (3.52, 4000, 5200, .6), (5.1, 2700, 3900, -.2), (5.4, 3000, 4100, -.2), (7.0, 3800, 5000, .4),
                       (43.3, 3300, 4400, -.4), (43.55, 3600, 4700, -.4), (45.4, 4400, 5600, .5), (47.0, 2800, 3800, 0.0)):
    add(fxl, fxr, int(t0 * SR), chirp(f0, f1, .14), pan, .035)
# ---------------- sidechain-style ducking of pad and melody under the kick
duck = np.ones(N)
for tt in kicks:
    a = int(tt * SR); n = int(.35 * SR)
    if a + n < N: duck[a:a + n] -= .4 * np.exp(-np.arange(n) / SR * 11)
padl *= duck; padr *= duck; mel *= (1 - (1 - duck) * .6); mer *= (1 - (1 - duck) * .6)
# ---------------- reverb (decaying noise impulse response) on pad, melody and effects; a little on the drums
def reverb(x, wet_len=2.6, seed=0):
    r = np.random.default_rng(seed); n = int(wet_len * SR); ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 2.3); ir[:int(.012 * SR)] *= 0
    ir = np.diff(ir, prepend=0) * .6 + ir * .4; ir /= np.abs(ir).sum() ** .5
    m = N + n; return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:N]
wetL = reverb(mel + padl * .7 + fxl * .5, seed=11); wetR = reverb(mer + padr * .7 + fxr * .5, seed=12)
L = drl + padl + mel * .8 + fxl + wetL * .55 + reverb(drl, 1.2, 21) * .10
R = drr + padr + mer * .8 + fxr + wetR * .55 + reverb(drr, 1.2, 22) * .10
# ---------------- master: fade in and out, soft clipping, normalise
fade = np.clip(t_all / 1.2, 0, 1) * np.clip((DUR - t_all) / 3.0, 0, 1) ** 1.3
L *= fade; R *= fade
peak = max(np.abs(L).max(), np.abs(R).max()); g = .85 / peak
L = np.tanh(L * g * 1.35) / np.tanh(1.35); R = np.tanh(R * g * 1.35) / np.tanh(1.35)
pcm = (np.stack([L, R], 1) * 32000).astype('<i2')
with wave.open(os.path.join(HERE, 'intro2.wav'), 'wb') as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())
rms = [float(np.sqrt(np.mean(L[int(i * SR):int((i + 2.5) * SR)] ** 2))) for i in range(0, int(DUR - 2), 5)]
print('intro2.wav %.1f s, peak %.2f, loudness by 5 s step:' % (DUR, np.abs(L).max()), ' '.join('%.2f' % v for v in rms))
