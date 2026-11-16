"""Detect kicks by their pitch-drop signature (sine sweeping ~100-220 Hz -> 40-55 Hz in <=150 ms).
Usage: python kick_sweep.py file.mp3 [start_s] [duration_s]
Steady/wobbling sub-bass is rejected because it never *starts* high."""
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, fftconvolve, find_peaks, hilbert

FILE  = sys.argv[1] if len(sys.argv) > 1 else "Inaktiv - Inner Fears.mp3"
START = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
DUR   = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

# --- what a kick looks like (edit to taste) ---
F_START_RANGE = (100, 240)   # pitch at the very start of the kick
F_END_RANGE   = (25, 60)     # pitch it settles to
MAX_SWEEP_MS  = 150
MIN_SCORE     = 0.6          # matched-filter similarity (0..1)

y, sr = librosa.load(FILE, sr=None, mono=True, offset=START, duration=DUR)
x = sosfiltfilt(butter(4, [25, 300], btype="band", fs=sr, output="sos"), y)

# 1) matched-filter bank of exponential pitch-drop sines (phase-invariant via complex template)
L = int(0.12 * sr); t = np.arange(L) / sr
score = np.zeros(len(x))
for f0 in (140, 170, 200, 230):
    for fend in (45, 52):
        for tau in (0.015, 0.025, 0.04):
            f = fend + (f0 - fend) * np.exp(-t / tau)
            a = np.exp(-t / 0.12)
            T = a * np.exp(2j * np.pi * np.cumsum(f) / sr)
            c = np.abs(fftconvolve(x, np.conj(T)[::-1])[L - 1:L - 1 + len(x)])
            E = fftconvolve(x ** 2, (a ** 2)[::-1])[L - 1:L - 1 + len(x)]
            score = np.maximum(score, c / np.sqrt((a ** 2).sum() * E + 1e-9))
pk, _ = find_peaks(score, height=MIN_SCORE, distance=int(0.12 * sr))

# 2) verify each candidate with the measured instantaneous frequency
z = hilbert(x); amp = np.abs(z)
fi = np.gradient(np.unwrap(np.angle(z))) * sr / (2 * np.pi)
k = int(0.006 * sr)
fs_ = np.convolve(fi, np.ones(k) / k, "same")
def f_at(i):
    w = int(0.005 * sr); return np.average(fs_[i - w:i + w], weights=amp[i - w:i + w])

# onset refinement: first rise of the 80-300 Hz envelope (the part of the kick the sub can't fake)
hi = np.abs(hilbert(sosfiltfilt(butter(4, [80, 300], btype="band", fs=sr, output="sos"), y)))
hi = np.convolve(hi, np.ones(int(.002 * sr)) / int(.002 * sr), "same")

kicks = []
print(" onset(s)  abs(s)  score | f@0  f@25  f@50  f@100 f@150 Hz | verdict")
for p in pk:
    fl = [f_at(p + int(d * sr)) for d in (0.003, .025, .05, .1, .15)]
    ok = (F_START_RANGE[0] <= fl[0] <= F_START_RANGE[1] and fl[2] <= 75 and
          F_END_RANGE[0] <= fl[4] <= F_END_RANGE[1] + 10)
    # refine onset
    a = p - int(.02 * sr); b = p + int(.02 * sr)
    floor = np.percentile(hi[p - int(.05 * sr):p - int(.02 * sr)], 90)
    plateau = np.max(hi[p:p + int(.05 * sr)])
    i = a + np.argmax(hi[a:b] > floor + 0.25 * (plateau - floor))
    print(f" {i/sr:8.4f} {START+i/sr:7.3f} {score[p]:5.2f} | " + " ".join(f"{v:5.0f}" for v in fl) +
          f" | {'KICK' if ok else 'reject (no high-start sweep)'}")
    if ok: kicks.append(i)

# 3) picture: low-end reassigned spectrogram + kick markers
fr, tm, mg = librosa.reassigned_spectrogram(y, sr=sr, n_fft=2048, hop_length=64, window="hann")
db = librosa.amplitude_to_db(mg, ref=np.max)
m = np.isfinite(fr) & np.isfinite(tm) & (db > -40) & (fr >= 25) & (fr <= 400)
fig, ax = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
ax[0].scatter(tm[m], fr[m], c=db[m], s=1, cmap="magma"); ax[0].set_yscale("log"); ax[0].set_ylim(25, 400)
for i in kicks: ax[0].axvline(i / sr, c="cyan", lw=1.6, alpha=.9)
ax[0].set_title("25-400 Hz reassigned (cyan = detected kicks)")
ax[1].plot(np.arange(len(x)) / sr, score); ax[1].axhline(MIN_SCORE, c="gray", ls="--")
ax[1].plot(pk / sr, score[pk], "v", c="orange"); ax[1].set_title("pitch-drop matched filter score")
plt.tight_layout(); plt.savefig("kick_sweep.png", dpi=80)
