"""Offline replay of the full current detect order from the cache: python dev/evaluation/replay_full_pipeline.py REPORT CACHE OUT.json [--limit N]
Models are the bundled ones (trained on all tracks), so results on training tracks are in-sample."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys, json, pickle, os
from pathlib import Path
from lib.beatgrid import grid_pipeline
from dev.evaluation.replay_cached_events import pseudo_stored
report = json.load(open(sys.argv[1])); cache_folder = sys.argv[2]; output_path = sys.argv[3]
limit = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
results = json.load(open(output_path)) if os.path.exists(output_path) else {}
for report_index, report_row in enumerate(report[:limit]):
    track_id = report_row['id']
    if track_id in results: continue
    try:
        events = pickle.loads((Path(cache_folder) / f"{track_id}.ff2.pkl").read_bytes())
        raw_sections = grid_pipeline.find_sections(grid_pipeline.block_scans(events))
        fits = grid_pipeline.clean_sections(events, raw_sections, [grid_pipeline.fit_section(events, raw_section) for raw_section in raw_sections])
        fits = grid_pipeline.fix_rekordbox_ratio(events, raw_sections, fits, report_row.get('rekordbox_bpm'))
        fits = grid_pipeline.fix_whole_ratio(events, raw_sections, fits)
        fits = grid_pipeline.fix_rekordbox_near(events, fits, report_row.get('rekordbox_bpm'))
        fits = grid_pipeline.drop_ghost_tail(events, fits)
        if not [fit for fit in fits if fit]:
            forced_section = grid_pipeline.forced_rekordbox_section(events, report_row.get('rekordbox_bpm')); fits = [forced_section] if forced_section else fits
        sections = [grid_pipeline.fine_adjust(events, grid_pipeline.refine_anchor(events, grid_pipeline.select_phase(events, grid_pipeline.fix_parity(events, fit, first=(section_index == 0))))) for section_index, fit in enumerate(fits)]
        built = grid_pipeline.build_grid(sections, report_row['offset_model']['ms'], events['dur'], events)
        stored_grid = pseudo_stored(report_row)
        results[track_id] = dict(dr=grid_pipeline.dry_compare(built, stored_grid) if stored_grid else None, bpms=[section['bpm'] for section in sections], fine=[section.get('fine_shift_ms') for section in sections], db=sections[0].get('downbeat') if sections else None)
    except Exception as e:
        results[track_id] = dict(error=repr(e))
    if report_index % 50 == 0: json.dump(results, open(output_path, 'w'))
json.dump(results, open(output_path, 'w'))
