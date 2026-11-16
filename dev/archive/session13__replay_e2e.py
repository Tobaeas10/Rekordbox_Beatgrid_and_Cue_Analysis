"""End-to-end offline replay with the new pipeline: python dev/replay_e2e.py REPORT CACHE OUT.json [FOLDS.json MODELDIR]
With folds, each track uses the out-of-fold model phase_model_fold<k>.json (honest estimate); otherwise the bundled model."""
import sys, json, pickle, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline as P, ranker
from replay import pseudo_stored
rep = json.load(open(sys.argv[1])); cd = sys.argv[2]; out = sys.argv[3]
folds = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
md = sys.argv[5] if len(sys.argv) > 5 else None
models = {}
res = json.load(open(out)) if os.path.exists(out) else {}
n = 0
for r in rep:
    i = r['id']
    if i in res: continue
    try:
        ev = pickle.loads((Path(cd) / f"{i}.ff2.pkl").read_bytes())
        if folds is not None and i in folds:
            k = folds[i]
            if k not in models: models[k] = ranker.GB.load(f"{md}/phase_model_fold{k}.json")
            m = models[k]
        else:
            m = P.PHASE_MODEL or P.load_phase_model()
        blocks = P.block_scans(ev); raw = P.find_sections(blocks)
        fits = P.clean_sections(ev, raw, [P.fit_section(ev, x) for x in raw])
        fits = P.fix_rb_ratio(ev, raw, fits, r.get('rekordbox_bpm'))
        secs = [P.refine_anchor(ev, P.select_phase(ev, P.fix_parity(ev, f, first=(k2 == 0)), model=m)) for k2, f in enumerate(fits)]
        built = P.build_grid(secs, r['offset_model']['ms'], ev['dur'], ev)
        st = pseudo_stored(r)
        dr = P.dry_compare(built, st) if st else None
        res[i] = dict(dr=dr, bpms=[s['bpm'] for s in secs], moved=[(s.get('phase_select') or {}).get('moved_ms') for s in secs])
    except Exception as e:
        res[i] = dict(error=repr(e))
    n += 1
    if n % 25 == 0: json.dump(res, open(out, 'w'))
json.dump(res, open(out, 'w'))
