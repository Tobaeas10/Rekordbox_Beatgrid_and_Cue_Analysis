"""adfeats.py (wip40): AFTER-DROP features from raw dB band-group levels (NOT per-band z-scores, which wash out the 1-2 dB balance step).
User fact: drop (~32 bars full) -> after-drop (~16 bars, same drums/bass, main synth missing or muffled: low end unchanged, mid/high ~1-2 dB lower, 'more bass relative to mids/highs') -> verse/outro (low end falls ~10 dB).
Measured on 39 '+16 late' tracks: at the true end low level -0.3 dB, tilt (low minus mid/hi) +1.2 dB; at the predicted (wrong) end low falls -10.8 dB, tilt falls back 3 dB.
Candidate i features: tilt/level chunk differences (w=8,16,32), LOOK-AHEAD (the big fall 8/16/24 bars later), chunk level relative to the track's loudest 16-bar chunk, tilt uniformity."""
import numpy as np
def ad_feats(t):
    BX = t['BX'].astype(np.float64); nb = len(BX) // 4
    lv = BX[:nb * 4].reshape(nb, 8, 22)[:, :, :10].mean(1)
    p = lambda a, b: 10 * np.log10((10 ** (lv[:, a:b] / 10)).sum(1) + 1e-12)
    L = np.stack([p(0, 4), p(4, 7), p(7, 9), p(9, 10)], 1)                      # low, mid, hi(4-8k), top(8-12k) dB per bar
    tl = np.stack([L[:, 0] - L[:, 1], L[:, 0] - L[:, 2], L[:, 0] - L[:, 3], L[:, 0] - 0.5 * (L[:, 1] + L[:, 2])], 1)   # tilts
    S = np.concatenate([L, tl], 1)                                              # 8 series
    i = np.arange(nb + 1)
    c = np.cumsum(np.vstack([np.zeros((1, 8)), S]), 0)
    def m(a_, b_):
        a_ = np.clip(a_, 0, nb); b_ = np.clip(b_, 0, nb); return (c[b_] - c[a_]) / np.maximum(b_ - a_, 1)[:, None]
    F = []
    for w in (8, 16, 32):
        F.append(m(i, i + w) - m(i - w, i))                                      # 8 per w
    for s in (8, 16, 24):                                                        # look-ahead steps at i+s (w=8)
        F.append(m(i + s, i + s + 8) - m(i + s - 8, i + s))
    F.append(m(i, i + 16) - m(i + 16, i + 32))                                   # chunk i vs next chunk
    roll = (c[16:] - c[:-16]) / 16.0 if nb >= 16 else np.zeros((1, 8))           # rolling 16-bar means
    ref = np.percentile(roll[:, :4], 90, 0) if len(roll) else np.zeros(4)        # loudest chunk per group
    F.append(m(i, i + 16)[:, :4] - ref); F.append(m(i - 16, i)[:, :4] - ref)     # level vs track's full-drop level
    c2 = np.cumsum(np.vstack([np.zeros((1, 8)), S ** 2]), 0)
    def sdv(a_, b_):
        a_ = np.clip(a_, 0, nb); b_ = np.clip(b_, 0, nb); n_ = np.maximum(b_ - a_, 1)[:, None]
        mu = (c[b_] - c[a_]) / n_; return np.sqrt(np.maximum((c2[b_] - c2[a_]) / n_ - mu ** 2, 0))[:, 4:]
    F += [sdv(i - 16, i), sdv(i, i + 16)]
    return np.concatenate(F, 1).astype(np.float32)
