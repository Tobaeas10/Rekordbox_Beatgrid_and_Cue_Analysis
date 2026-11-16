#!/usr/bin/env python3
"""Offline lab for downbeat (beat 1) detection.
Needs eval_out/eval_report.json plus eval_out/cache/<id>.feat.npz (run local_eval.py --feat once).
Prints, per track, the beat index mod 4 (relative to the detected anchor) of the true downbeat (Rekordbox) and the guess of each method.

  python downbeat_lab.py [--out eval_out]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def beat_onsets(env, rate, beat_t):
    """Positive log-energy jump at every beat (max over [-20, +40] ms), per band -> (n_beats, n_bands)."""
    e = np.log(env.astype(np.float32) + 1e-4)
    fl = np.maximum(e[:, 3:] - e[:, :-3], 0)
    fl = np.concatenate([np.zeros((e.shape[0], 3), np.float32), fl], 1)
    out = np.zeros((len(beat_t), env.shape[0]), np.float32)
    for j, bt in enumerate(beat_t):
        i0, i1 = int((bt - 0.02) * rate), int((bt + 0.04) * rate) + 1
        if i0 < 0 or i1 >= fl.shape[1]:
            continue
        out[j] = fl[:, i0:i1].max(1)
    return out


def beat_energy(env, rate, beat_t, T):
    out = np.zeros((len(beat_t), env.shape[0]), np.float32)
    for j, bt in enumerate(beat_t):
        i0, i1 = int(bt * rate), max(int(bt * rate) + 1, int((bt + T) * rate))
        if i0 < 0 or i1 > env.shape[1]:
            continue
        out[j] = env[:, i0:i1].mean(1)
    return out


def phase_votes(x, idx, L=4):
    v = np.zeros(4)
    for j, val in zip(idx, x):
        v[j % 4] += val
    return v


def methods(env, rate, s):
    T, a = s['period_s'], s['anchor_s']
    k0 = int(np.ceil((s['start_s'] - a) / T))
    k1 = int(np.floor((s['end_s'] - a) / T))
    idx = np.arange(k0, k1 + 1)
    bt = a + idx * T
    ons = beat_onsets(env, rate, bt)
    en = beat_energy(env, rate, bt, T)
    res = {}
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    res['strong_low'] = phase_votes(ons[:, 1], idx)
    res['strong_sub+low'] = phase_votes(ons[:, 0] + ons[:, 1], idx)
    res['strong_all'] = phase_votes(ons.sum(1), idx)
    for L in (4, 8):
        nov = np.zeros(len(idx))
        e_low = en[:, 0] + en[:, 1]
        for m in range(L, len(idx) - L):
            nov[m] = abs(e_low[m:m + L].mean() - e_low[m - L:m].mean())
        res[f'novelty_low{L}'] = phase_votes(nov, idx)
    res['combo'] = z(res['strong_low']) + z(res['novelty_low8']) + z(res['strong_all'])
    res['first_beat'] = np.eye(4)[int(idx[0]) % 4] if False else np.eye(4)[(int(np.ceil(-a / T - 0.015 / T))) % 4]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='eval_out')
    a = ap.parse_args()
    out = Path(a.out)
    rows = json.loads((out / 'eval_report.json').read_text(encoding='utf-8'))
    tot, hit, n = {}, {}, 0
    for t in rows:
        f = out / 'cache' / f"{t['id']}.feat.npz"
        if not f.exists() or not t.get('sections'):
            continue
        z = np.load(f)
        env, rate = z['env'].astype(np.float32), float(z['sr']) / float(z['hop'])
        s = max(t['sections'], key=lambda x: x['end_s'] - x['start_s'])
        s = dict(s, period_s=60.0 / s['bpm'], anchor_s=s['anchor_ms'] / 1000.0)
        g = t['rb_grid']
        T = 60000.0 / t['rekordbox_bpm']
        true_p = int(round(((g['first_downbeat_ms'] - t['offset_model']['ms']) - s['anchor_ms']) / T)) % 4 if 'anchor_ms' in s else None
        r = methods(env, rate, s)
        n += 1
        line = f"{t['artist'][:14]:14s} true {true_p} |"
        for k, v in r.items():
            p = int(np.argmax(v))
            line += f" {k} {p}{'*' if p == true_p else ' '}"
            hit[k] = hit.get(k, 0) + (p == true_p)
        print(line)
    print('\naccuracy over', n, 'tracks (random = 25%):', {k: f"{100 * v / max(n, 1):.0f}%" for k, v in hit.items()})


if __name__ == '__main__':
    main()
