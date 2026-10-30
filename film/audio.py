"""Synthesize a subtle cinematic ambient bed for the shot (10s, 48k stereo)."""
import numpy as np
import wave
import os

SR = 48000
DUR = 10.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(11)

L = np.zeros(N, np.float32)
R = np.zeros(N, np.float32)


def add(sig, gain, pan=0.0):
    global L, R
    l = np.sqrt((1 - pan) / 2) * 1.414
    r = np.sqrt((1 + pan) / 2) * 1.414
    L += (sig * gain * l).astype(np.float32)
    R += (sig * gain * r).astype(np.float32)


def env(points):
    """Piecewise-linear envelope from (time, value) pairs."""
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    return np.interp(t, xs, ys).astype(np.float32)


# --- low cinematic drone (A1-ish), slow beating
f0 = 55.0
for k, (mult, g) in enumerate([(1.0, 0.30), (2.0, 0.16), (3.0, 0.07), (4.02, 0.035)]):
    ph = rng.uniform(0, 2 * np.pi)
    lfo = 1.0 + 0.18 * np.sin(2 * np.pi * (0.055 + 0.013 * k) * t + ph)
    add(np.sin(2 * np.pi * f0 * mult * t + ph) * lfo, g * 0.5, pan=(-0.25 + 0.1 * k))

# slow detuned fifth for width
add(np.sin(2 * np.pi * 82.4 * t + 1.1) * (1 + 0.2 * np.sin(2 * np.pi * 0.043 * t)), 0.055, pan=0.35)
add(np.sin(2 * np.pi * 41.2 * t + 2.7) * (1 + 0.25 * np.sin(2 * np.pi * 0.031 * t)), 0.10, pan=0.0)

# --- air / breath noise: band-passed noise with slow swells
def band_noise(lo, hi, q_soft=True):
    x = rng.normal(0, 1, N).astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1 / SR)
    if q_soft:
        mask = np.exp(-0.5 * ((np.log(freqs + 1) - np.log((lo * hi) ** 0.5)) / 0.55) ** 2)
    else:
        mask = ((freqs > lo) & (freqs < hi)).astype(np.float32)
    return np.fft.irfft(spec * mask, N).astype(np.float32)


air = band_noise(280, 2600)
air /= (np.abs(air).max() + 1e-6)
air_env = env([(0, 0.0), (1.4, 0.45), (3.0, 0.25), (4.6, 0.55), (6.5, 0.30), (8.2, 0.20), (10, 0.0)])
air_env *= 1 + 0.28 * np.sin(2 * np.pi * 0.19 * t + 0.4)
add(air * air_env, 0.055, pan=-0.15)
air2 = band_noise(900, 5200)
air2 /= (np.abs(air2).max() + 1e-6)
add(air2 * air_env * 0.7, 0.020, pan=0.30)

# --- breath accents (soft whoosh) around blinks and the head turn
def breath_swell(center, width, gain, pan=0.0):
    x = band_noise(200, 1800)
    x /= (np.abs(x).max() + 1e-6)
    e = np.exp(-0.5 * ((t - center) / width) ** 2).astype(np.float32)
    add(x * e, gain, pan)


breath_swell(0.95, 0.22, 0.055, -0.1)    # first blink
breath_swell(2.30, 0.45, 0.045, 0.1)     # inhale, posture shift
breath_swell(4.30, 0.35, 0.040, 0.0)
breath_swell(5.42, 0.20, 0.060, 0.2)     # second blink
breath_swell(7.40, 0.55, 0.035, -0.1)

# --- high shimmer pad (very quiet, adds "expensive" air)
for f, ph, g in [(440 * 2, 0.2, 0.008), (660 * 2, 1.4, 0.006), (990 * 2, 2.4, 0.004)]:
    sig = np.sin(2 * np.pi * f * t + ph) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + ph))
    add(sig, g * env([(0, 0), (2.0, 1.0), (6.0, 0.9), (10, 0)]), pan=rng.uniform(-0.5, 0.5))

# --- soft sub pulse (heartbeat-ish), very low
for c in [1.8, 4.15, 6.55, 8.35]:
    e = np.exp(-0.5 * ((t - c) / 0.11) ** 2).astype(np.float32)
    add(np.sin(2 * np.pi * 46 * t) * e, 0.16, pan=0.0)

# --- master: gentle low shelf rolloff + fade in/out
def one_pole_lp(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    # vectorised via lfilter-like recursion (numpy loop is fine at 480k)
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x).astype(np.float32)


from scipy.signal import lfilter  # noqa: E402

L = one_pole_lp(L, 9000)
R = one_pole_lp(R, 9000)
master = env([(0, 0.0), (1.1, 0.85), (8.6, 0.95), (10.0, 0.0)])
L *= master
R *= master

peak = max(np.abs(L).max(), np.abs(R).max())
gain = 0.42 / (peak + 1e-6)
L *= gain
R *= gain

stereo = np.stack([L, R], -1)
pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'score.wav')
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote', out, pcm.shape)
