"""Offline replay of the detector from cached events (no audio): python dev/evaluation/replay_cached_events.py REPORT.json CACHE_DIR OUT.json [--track-ids a,b] [--limit N]
Pseudo-stored grid is rebuilt from rb_grid (constant tempo only); results are comparable to the report's dry_run."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys, json, pickle, time, os
from pathlib import Path
import numpy as np
from lib.beatgrid import grid_pipeline


def pseudo_stored(report_row):
    rekordbox_grid = report_row['rb_grid']
    if len(rekordbox_grid['tempos_bpm']) != 1:
        return None
    period_ms = 60000.0 / rekordbox_grid['tempos_bpm'][0]
    beat_count = rekordbox_grid['n_beats']
    downbeat_offset = round((rekordbox_grid['first_downbeat_ms'] - rekordbox_grid['first_beat_ms']) / period_ms) % 4
    return [(((beat_index - downbeat_offset) % 4) + 1, int(round(rekordbox_grid['tempos_bpm'][0] * 100)), rekordbox_grid['first_beat_ms'] + beat_index * period_ms) for beat_index in range(beat_count)]


def detect_offline(events, rekordbox_bpm, pipeline_module=grid_pipeline):
    blocks = pipeline_module.block_scans(events)
    raw_sections = pipeline_module.find_sections(blocks)
    fits = pipeline_module.clean_sections(events, raw_sections, [pipeline_module.fit_section(events, raw_section) for raw_section in raw_sections])
    fits = pipeline_module.fix_rekordbox_ratio(events, raw_sections, fits, rekordbox_bpm)
    return [pipeline_module.refine_anchor(events, pipeline_module.fix_parity(events, pipeline_module.snare_override(events, pipeline_module.phase_fix(events, fit)), first=(section_index == 0))) for section_index, fit in enumerate(fits)]


def replay_one(report_row, cache_dir, pipeline_module=grid_pipeline, detect=detect_offline):
    events = pickle.loads((Path(cache_dir) / f"{report_row['id']}.ff2.pkl").read_bytes())
    sections = detect(events, report_row.get('rekordbox_bpm'), pipeline_module)
    offset_model_ms = report_row['offset_model']['ms']
    built = pipeline_module.build_grid(sections, offset_model_ms, events['dur'], events)
    stored_grid = pseudo_stored(report_row)
    dry_run = pipeline_module.dry_compare(built, stored_grid) if stored_grid else None
    return sections, built, dry_run


if __name__ == '__main__':
    report = json.load(open(sys.argv[1])); cache_folder = sys.argv[2]; output_path = sys.argv[3]
    selected_ids = None; limit = None
    if '--track-ids' in sys.argv: selected_ids = set(sys.argv[sys.argv.index('--track-ids') + 1].split(','))
    if '--limit' in sys.argv: limit = int(sys.argv[sys.argv.index('--limit') + 1])
    results = {}
    if os.path.exists(output_path): results = json.load(open(output_path))
    processed_count = 0
    for r in report:
        if selected_ids and r['id'] not in selected_ids: continue
        if r['id'] in results: continue
        if limit and processed_count >= limit: break
        start_time = time.time()
        try:
            sections, built, dr = replay_one(r, cache_folder)
            results[r['id']] = dict(dr=dr, secs=[dict(start_s=section['start_s'], end_s=section['end_s'], bpm=section['bpm'], anchor_s=section['anchor_s'], period_s=section['period_s'], rms=section['rms_ms'],
                                                  q=(section.get('downbeat') or {}).get('q')) for section in sections], sec=round(time.time() - start_time, 1))
        except Exception as e:
            results[r['id']] = dict(error=repr(e))
        processed_count += 1
        if processed_count % 20 == 0: json.dump(results, open(output_path, 'w'))
    json.dump(results, open(output_path, 'w'))
