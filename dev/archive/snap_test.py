import sys
import numpy as np
sys.path.insert(0, '..')
import pipeline as P
import beatgrid as B


def fit_fixed(t, w, a, bpm, tol=0.012, iters=6):
    half = 60 / bpm / 2
    for _ in range(iters):
        k = np.round((t - a) / half)
        res = t - (a + k * half)
        inl = np.abs(res) < tol
        if inl.sum() < 5:
            return None
        a = a + np.average(res[inl], weights=w[inl])
    k = np.round((t - a) / half)
    res = t - (a + k * half)
    inl = np.abs(res) < tol
    return a, 1000 * np.sqrt(np.mean(res[inl] ** 2)), int(inl.sum())


for f in sys.argv[1:]:
    ev = P.get_events(f)
    t, w = ev['t'], ev['w']
    for s in P.analyze(f)['sections']:
        m = (t >= s['start_s']) & (t < s['end_s'])
        for cand in sorted({round(s['bpm']), round(s['bpm'], 1), round(s['bpm'], 2)}):
            fx = fit_fixed(t[m], w[m], s['anchor_s'], cand)
            if fx:
                print('%s %5.0f-%5.0f free %9.4f rms %.2f inl %3d | fixed %8.2f rms %.2f inl %3d' % (
                    f[:22], s['start_s'], s['end_s'], s['bpm'], s['rms_ms'], s['n_inl'], cand, fx[1], fx[2]))
