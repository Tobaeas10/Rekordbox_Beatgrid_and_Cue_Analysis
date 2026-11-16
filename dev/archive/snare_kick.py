import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks

FILE = sys.argv[1] if len(sys.argv) > 1 else "evar - power (fugitives remix).mp3"
START, DURATION = 55.0, 3.6923
SNARE_BAND = (2000, 12000)   # broadband snare/clap energy
KICK_BAND  = (30, 150)
SMOOTH_MS  = 0.5             # envelope smoothing (lower = sharper, noisier)
FRAC       = 0.25            # onset = envelope reaches floor + FRAC*(plateau-floor)
ZOOM_MS    = 80

y, sr = librosa.load(FILE, sr=None, mono=True, offset=START, duration=DURATION)

def band(lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), y)

def env(x, ms):
    k = max(1, int(sr * ms / 1000))
    return np.convolve(np.abs(hilbert(x)), np.ones(k) / k, "same")

sn, ke = band(*SNARE_BAND), env(band(*KICK_BAND), 0.5)
fine = env(sn, SMOOTH_MS)

# 1) find "big jump in broadband energy": mean after / mean before, 30 ms each side
w = int(0.03 * sr)
c = np.concatenate([[0], np.cumsum(env(sn, 1.0))])
idx = np.arange(w, len(y) - w)
ratio = np.zeros(len(y))
ratio[idx] = (c[idx + w] - c[idx]) / (c[idx] - c[idx - w] + 1e-9)
after = np.zeros(len(y)); after[idx] = (c[idx + w] - c[idx]) / w
# need a big relative jump AND a genuinely loud level (>=70% of the loudest hit)
pk, _ = find_peaks(ratio * (after >= 0.7 * after.max()), height=3, distance=int(0.25 * sr))

# 2) refine each to the first sample where the fine envelope leaves the noise floor
hits = []
for p in pk:
    floor = np.percentile(fine[p - int(.04*sr): p - int(.015*sr)], 90)
    plateau = np.median(fine[p + int(.005*sr): p + int(.025*sr)])
    thr = floor + FRAC * (plateau - floor)
    a = p - int(.015 * sr)               # only look just before the jump
    i = a + np.argmax(fine[a:p + int(.01 * sr)] > thr)
    hits.append(i)

print("kick+snare onsets (s rel. to START | % of window | absolute):")
for i in hits:
    print(f"  {i/sr:.5f}s | {i/sr/DURATION*100:5.2f}% | {START + i/sr:.5f}s")

fig, ax = plt.subplots(3, len(hits), figsize=(7 * len(hits), 8), squeeze=False)
h = int(ZOOM_MS / 2000 * sr)
for k, i in enumerate(hits):
    a, b = i - h, i + h
    tt = (np.arange(a, b) - i) / sr * 1000
    for r, (sig, ttl) in enumerate([(sn[a:b], "snare band waveform"),
                                    (fine[a:b], "snare band envelope"),
                                    (ke[a:b], "kick band envelope")]):
        ax[r, k].plot(tt, sig, lw=.6); ax[r, k].axvline(0, c="r", lw=.8)
        ax[r, k].set_title(f"{ttl}  (hit at {START + i/sr:.4f}s)" if r == 0 else ttl)
        ax[r, k].grid(alpha=.3)
    ax[2, k].set_xlabel("ms from detected onset")
plt.tight_layout(); plt.savefig("kick_snare_onsets.png", dpi=110)
