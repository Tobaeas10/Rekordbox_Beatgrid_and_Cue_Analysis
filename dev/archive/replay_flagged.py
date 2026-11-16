"""Replay fit -> phase_fix -> snare_override from a cache dir (ff2.pkl with nov_*) for tracks flagged in an eval_report.json.
usage: python dev/replay_flagged.py eval_report.json cache_dir"""
import sys, json, glob, pickle
sys.path.insert(0, '.')
import pipeline as P
d = json.load(open(sys.argv[1]))
fs = {f.split('/')[-1].split('.')[0]: f for f in glob.glob(sys.argv[2] + '/**/*.ff2.pkl', recursive=True)}
for r in d:
    if not any(s.get('phase_shift_ms') or s.get('snare_override_ms') for s in r['sections']) or r['id'] not in fs:
        continue
    ev = pickle.load(open(fs[r['id']], 'rb'))
    raw = P.find_sections(P.block_scans(ev))
    fits = P.clean_sections(ev, raw, [P.fit_section(ev, x) for x in raw])
    out = []
    for f in fits:
        g = P.snare_override(ev, P.phase_fix(ev, f))
        out.append((round(g.get('phase_shift_ms') or 0, 1), g.get('snare_override_ms'), g.get('lattice_kicks')))
    print(r['title'][:34], out, flush=True)
