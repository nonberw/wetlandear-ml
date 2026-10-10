"""Synthesised soundtrack for the intro video: soft pad, wind and a few bird-like chirps. Nothing here is a recording, so there are no licence strings attached.
Writes intro.wav (44.1 kHz stereo)."""
import os, wave
import numpy as np

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
DUR = float(os.environ.get('DUR', '44'))
rng = np.random.default_rng(5)
N = int(SR * DUR)
t = np.arange(N) / SR
L = np.zeros(N); R = np.zeros(N)


def env(x, a, d):
    """attack/decay envelope for an array of times starting at 0"""
    return np.minimum(x / a, 1.0) * np.exp(-np.maximum(x - a, 0) / d)


# ---- pad: chords change with the scenes (Am, F, C, G, Am); each note is a few slightly detuned sines
chords = [(0, [110.0, 164.8, 220.0, 261.6, 329.6]), (9, [87.3, 130.8, 174.6, 261.6, 349.2]), (17, [65.4, 130.8, 196.0, 261.6, 329.6]), (27, [98.0, 146.8, 196.0, 293.7, 392.0]), (37, [110.0, 164.8, 220.0, 329.6, 440.0])]
for k, (t0, notes) in enumerate(chords):
    t1 = chords[k + 1][0] if k + 1 < len(chords) else DUR
    a, b = int(t0 * SR), int(min(t1 + 1.5, DUR) * SR)
    x = t[a:b] - t0
    gate = np.clip(x / 1.8, 0, 1) * np.clip((t1 + 1.5 - t[a:b]) / 1.8, 0, 1)
    for f in notes:
        for det, pan in ((-0.6, -0.4), (0.0, 0.0), (0.7, 0.4)):
            ph = rng.uniform(0, 6.28)
            s = (np.sin(2 * np.pi * (f + det) * x + ph) + 0.25 * np.sin(2 * np.pi * 2 * (f + det) * x + ph * 0.5)) * 0.018 * gate * (0.8 + 0.2 * np.sin(2 * np.pi * 0.11 * x + ph))
            L[a:b] += s * (1 - pan) / 2; R[a:b] += s * (1 + pan) / 2

# ---- wind: low-passed noise that swells under the landscape scenes
def lowpass_noise(cut):
    f = np.fft.rfft(rng.standard_normal(N)); fr = np.fft.rfftfreq(N, 1 / SR)
    f *= 1 / (1 + (fr / cut) ** 2) ** 1.5
    x = np.fft.irfft(f, N); return x / np.abs(x).max()


w1, w2 = lowpass_noise(420), lowpass_noise(380)
swell = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + 1))
scene_gain = np.interp(t, [0, 1.5, 8, 10, 36, 38, DUR], [0, 1.0, 1.0, 0.25, 0.25, 1.0, 0.6])
L += w1 * 0.07 * swell * scene_gain; R += w2 * 0.07 * swell * scene_gain


# ---- bird-like chirps (frequency sweeps with a little vibrato), mostly while birds fly over the mountains
def chirp(t0, f0, f1, dur, amp, pan, vib=18):
    a = int(t0 * SR); n = int(dur * SR)
    if a + n >= N: return
    x = np.arange(n) / SR; u = x / dur
    f = f0 + (f1 - f0) * u ** 1.4 + 90 * np.sin(2 * np.pi * vib * x)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * amp * np.sin(np.pi * u) ** 1.5
    L[a:a + n] += s * (1 - pan) / 2 * 2; R[a:a + n] += s * (1 + pan) / 2 * 2


def song(t0, base, n, gap, amp, pan):
    for i in range(n):
        chirp(t0 + i * gap, base * rng.uniform(0.95, 1.1), base * rng.uniform(1.15, 1.6) * (1 if i % 2 else 0.8), rng.uniform(0.09, 0.16), amp, pan, rng.uniform(10, 30))


song(1.8, 3200, 4, 0.22, 0.045, -0.5); song(3.4, 4300, 3, 0.18, 0.04, 0.6); song(5.2, 2700, 5, 0.25, 0.04, -0.2); song(7.0, 3800, 3, 0.2, 0.035, 0.4)
song(38.2, 3300, 4, 0.22, 0.04, -0.4); song(39.9, 4400, 3, 0.18, 0.035, 0.5); song(41.5, 2800, 4, 0.24, 0.035, 0.0)
# crane-like two-tone call in the middle of the first scene
for t0, pan in ((4.4, 0.3), (6.0, -0.3)):
    chirp(t0, 780, 1050, 0.55, 0.03, pan, 6); chirp(t0 + 0.12, 1560, 2100, 0.5, 0.012, pan, 6)


# ---- soft interface ticks at the scene changes and when species labels appear
def tick(t0, f=880.0, amp=0.05):
    a = int(t0 * SR); n = int(0.35 * SR)
    if a + n >= N: return
    x = np.arange(n) / SR; s = (np.sin(2 * np.pi * f * x) + 0.4 * np.sin(2 * np.pi * 2 * f * x)) * np.exp(-x * 11) * amp
    L[a:a + n] += s; R[a:a + n] += s


for t0 in (9.0, 17.0, 27.0, 37.0): tick(t0, 660, 0.05)
for t0, f in ((12.2, 784), (13.6, 880), (15.0, 988), (16.2, 1175)): tick(t0, f, 0.04)
for t0 in (29.4, 32.6, 35.4): tick(t0, 740, 0.035)

# ---- master: slow fade in and out, gentle limiter
fade = np.clip(t / 1.5, 0, 1) * np.clip((DUR - t) / 2.5, 0, 1)
L *= fade; R *= fade
peak = max(np.abs(L).max(), np.abs(R).max()); g = 0.8 / peak if peak > 0 else 1
L = np.tanh(L * g * 1.2) / np.tanh(1.2); R = np.tanh(R * g * 1.2) / np.tanh(1.2)
pcm = (np.stack([L, R], 1) * 32000).astype('<i2')
with wave.open(os.path.join(HERE, 'intro.wav'), 'wb') as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())
print('intro.wav', DUR, 's')
