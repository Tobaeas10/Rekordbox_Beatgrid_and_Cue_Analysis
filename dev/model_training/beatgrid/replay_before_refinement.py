"""Replay to the pre-refine stage without phase_fix/snare_override: python dev/model_training/beatgrid/replay_before_refinement.py REPORT CACHE OUT.json
Stores per track the fit sections after fix_rekordbox_ratio and fix_parity (anchor before refine_anchor) and the refined anchor."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys, json, pickle, os
from pathlib import Path
from lib.beatgrid import grid_pipeline

report = json.load(open(sys.argv[1])); cache_folder = sys.argv[2]; output_path = sys.argv[3]
results = json.load(open(output_path)) if os.path.exists(output_path) else {}
processed_count = 0
for report_row in report:
    if report_row['id'] in results:
        continue
    try:
        events = pickle.loads((Path(cache_folder) / f"{report_row['id']}.ff2.pkl").read_bytes())
        blocks = grid_pipeline.block_scans(events)
        raw_sections = grid_pipeline.find_sections(blocks)
        fits = grid_pipeline.clean_sections(events, raw_sections, [grid_pipeline.fit_section(events, raw_section) for raw_section in raw_sections])
        fits = grid_pipeline.fix_rekordbox_ratio(events, raw_sections, fits, report_row.get('rekordbox_bpm'))
        sections = []
        for section_index, fit in enumerate(fits):
            parity_fixed = grid_pipeline.fix_parity(events, dict(fit), first=(section_index == 0))
            refined = grid_pipeline.refine_anchor(events, dict(parity_fixed))
            sections.append(dict(start_s=parity_fixed['start_s'], end_s=parity_fixed['end_s'], bpm=parity_fixed['bpm'], period_s=parity_fixed['period_s'], anchor_s=parity_fixed['anchor_s'],
                           rms_ms=parity_fixed['rms_ms'], anchor_ref_s=refined['anchor_s'], refine=refined.get('refine'),
                           parity={key: value for key, value in (parity_fixed.get('parity') or {}).items() if not isinstance(value, (list, dict))},
                           rb_ratio_fix=parity_fixed.get('rb_ratio_fix')))
        results[report_row['id']] = dict(secs=sections)
    except Exception as e:
        results[report_row['id']] = dict(error=repr(e))
    processed_count += 1
    if processed_count % 25 == 0:
        json.dump(results, open(output_path, 'w'))
json.dump(results, open(output_path, 'w'))
