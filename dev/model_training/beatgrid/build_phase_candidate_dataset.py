"""Phase candidate dataset v2: ours = parity-fixed anchor before refine (no phase_fix / snare_override). Needs base2.json from replay_before_refinement.py."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json, pickle, sys
import numpy as np
import pandas as pd
from lib.beatgrid import rankers

event_cache_folder = str(paths.EVENT_CACHES / 'old961') + '/'
report_path = str(paths.EVAL_REPORTS / 'eval_report.json')
TOLERANCE_MS = 8.0


def truth_abs(report_row, report_section):
    """Truth anchor (s, decoded time) for a report section."""
    return (report_section['anchor_ms'] + report_section['err_vs_model_ms']) / 1000.0


def build(base_path, rows_path=str(paths.WORK / 'scope.json'), out=str(paths.WORK / 'phase_cand2.pkl')):
    report = json.load(open(report_path))
    base_sections = json.load(open(base_path))
    scope = {feature_row['id'] for feature_row in json.load(open(rows_path))}
    candidate_rows, track_info = [], []
    for report_row in report:
        track_id = report_row['id']
        if track_id not in scope or track_id not in base_sections or 'secs' not in base_sections[track_id] or not base_sections[track_id]['secs'] or not report_row['sections']:
            continue
        section = max(base_sections[track_id]['secs'], key=lambda report_section: report_section['end_s'] - report_section['start_s'])
        # the report section covering the same span supplies the truth
        matching_section = max(report_row['sections'], key=lambda report_section: min(report_section['end_s'], section['end_s']) - max(report_section['start_s'], section['start_s']))
        if 'err_vs_model_ms' not in matching_section or abs(matching_section['bpm'] - section['bpm']) > 0.6:
            continue
        period_seconds = section['period_s']
        true_offset_ms = rankers.wrap(truth_abs(report_row, matching_section) - section['anchor_s'], period_seconds) * 1000.0      # truth minus ours (pre-refine), ms
        events = pickle.load(open(event_cache_folder + track_id + '.ff2.pkl', 'rb'))
        positions, features = rankers.phase_candidates(events, section)
        if positions is None:
            continue
        track_info.append(dict(id=track_id, tru=true_offset_ms, T=period_seconds * 1000, ours_ok=int(abs(true_offset_ms) <= TOLERANCE_MS), refine_final=(section['anchor_ref_s'] - section['anchor_s']) * 1000))
        for candidate_position, feature_row in zip(positions, features):
            record = dict(zip(rankers.PHASE_FEATS, feature_row))
            record.update(id=track_id, pos=candidate_position, label=int(abs(rankers.wrap((candidate_position - true_offset_ms) / 1000.0, period_seconds) * 1000.0) <= TOLERANCE_MS), dpos=rankers.wrap((candidate_position - true_offset_ms) / 1000.0, period_seconds) * 1000.0, tru=true_offset_ms)
            candidate_rows.append(record)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_pickle(out)
    pd.DataFrame(track_info).to_pickle(out.replace('.pkl', '_info.pkl'))
    return candidates


if __name__ == '__main__':
    candidates = build(sys.argv[1] if len(sys.argv) > 1 else str(paths.WORK / 'base2.json'))
    best_label_per_track = candidates.groupby('id').label.max()
    ours_label_per_track = candidates[candidates.is_ours == 1].groupby('id').label.max()
    print(candidates.shape, candidates.id.nunique(), 'candidate covers truth', round(best_label_per_track.mean(), 4), 'ours covers', round(ours_label_per_track.mean(), 4))
