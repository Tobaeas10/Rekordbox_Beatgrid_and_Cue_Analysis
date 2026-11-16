"""Periodicity candidate dataset: candidates = ranker.phase_candidates positions of the FINAL section (report anchor_ms), label = within LABEL_TOLERANCE_MS of truth
(truth = anchor_ms + err_vs_model_ms, report rows only). Adds bar-consistency / jitter / autocorrelation features. Run: python periodicity_candidate_dataset.py OUT.pkl"""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import json, pickle, sys, glob
import numpy as np, pandas as pd
from lib import paths
from lib.beatgrid import rankers
from dev.model_training.beatgrid.track_metrics import ACAPELLA_PATTERN, TEMPO_CHANGE_PATTERN
REPORT_AND_CACHE_SETS = [(str(paths.EVAL_REPORTS / 'eval_report_-_first_800_dubstep_songs___all_initial_dnb__ukg__2step.json'), str(paths.EVENT_CACHES / 'old961') + '/'),
        (str(paths.EVAL_REPORTS / 'eval_report_-_698_more_dubstep_songs.json'), str(paths.EVENT_CACHES / 'new698') + '/'),
        (str(paths.EVAL_REPORTS / 'eval_report__extra_subgenres_.json'), str(paths.EVENT_CACHES / 'extra') + '/')]
LABEL_TOLERANCE_MS = 8.0


def periodicity_features(events, anchor_seconds, period_seconds, start_seconds, end_seconds, window_seconds=(6.0,)):
    """features of a lattice (anchor a [s], period T [s]) from kick/snare events"""
    in_section = (events['t'] >= start_seconds) & (events['t'] <= end_seconds)
    event_times, event_weights, event_classes = events['t'][in_section], events['w'][in_section], events['c'][in_section]
    beat_count = max(int((end_seconds - start_seconds) / period_seconds), 8)
    features = []
    for event_class in (0, 1):
        class_times, class_weights = event_times[event_classes == event_class], event_weights[event_classes == event_class]
        if len(class_times) == 0:
            features += [0] * 9; continue
        beat_index = np.round((class_times - anchor_seconds) / period_seconds); error_ms = (class_times - anchor_seconds - beat_index * period_seconds) * 1000.0
        is_hit = np.abs(error_ms) <= 6.0
        position_in_bar = (beat_index % 4).astype(int); bar_index = np.floor(beat_index / 4).astype(int)
        bar_count = max(beat_count / 4, 1)
        bar_coverage = [len(set(bar_index[is_hit & (position_in_bar == position)])) / bar_count for position in range(4)]
        is_near = np.abs(error_ms) <= 25.0
        jitter_ms = float(np.median(np.abs(error_ms[is_near & is_hit]))) if (is_hit & is_near).any() else 6.0
        clutter = float(((np.abs(error_ms) > 6) & is_near).sum() / (is_hit.sum() + 1))
        # beat hit train autocorr at lags 4 and 8 beats
        hit_beat_indices = (beat_index[is_hit] - beat_index.min()).astype(int) if is_hit.any() else np.array([], int)
        beat_span = int(beat_index.max() - beat_index.min()) + 1
        hit_train = np.zeros(max(beat_span, 1)); hit_train[hit_beat_indices[hit_beat_indices < beat_span]] = 1
        autocorrelation_lag4 = float((hit_train[4:] * hit_train[:-4]).sum() / (hit_train.sum() + 1)) if len(hit_train) > 8 else 0.
        features += [max(bar_coverage), sum(bar_coverage), bar_coverage[0], bar_coverage[2], jitter_ms, clutter, autocorrelation_lag4, float(event_weights[event_classes == event_class][is_hit].sum()) / bar_count if is_hit.any() else 0., float(is_hit.mean())]
    return features


PERIODICITY_FEATURE_NAMES = [f'{class_letter}_{feature_suffix}' for class_letter in 'ks' for feature_suffix in ('dmax', 'dsum', 'd0', 'd2', 'jit', 'clut', 'ac4', 'wbar', 'hitfrac')]


def build(output_path):
    records, track_info, seen_ids = [], [], set()
    for report_path, cache_folder in REPORT_AND_CACHE_SETS:
        for report_row in json.load(open(report_path)):
            track_id = report_row['id']
            if track_id in seen_ids or ACAPELLA_PATTERN.search(report_row['artist'] + report_row['title']) or TEMPO_CHANGE_PATTERN.search(report_row['artist'] + report_row['title']): continue
            seen_ids.add(track_id)
            if len(report_row['sections']) != 1: continue
            first_section = report_row['sections'][0]
            if 'err_vs_model_ms' not in first_section or first_section['bpm'] >= 160: continue
            try:
                events = pickle.load(open(cache_folder + track_id + '.ff2.pkl', 'rb'))
            except Exception:
                continue
            period_seconds = 60.0 / first_section['bpm']; anchor_seconds = first_section['anchor_ms'] / 1000.0
            section = dict(period_s=period_seconds, anchor_s=anchor_seconds, start_s=first_section['start_s'], end_s=first_section['end_s'], rms_ms=first_section.get('fit_rms_ms', 3.0))
            true_offset_ms = rankers.wrap((first_section['anchor_ms'] + first_section['err_vs_model_ms']) / 1000.0 - anchor_seconds, period_seconds) * 1000.0
            candidate_positions_ms, candidate_features = rankers.phase_candidates(events, section)
            if candidate_positions_ms is None: continue
            track_info.append(dict(id=track_id, name=report_row['artist'] + ' - ' + report_row['title'], tru=true_offset_ms, T=period_seconds * 1000, bpm=first_section['bpm']))
            for position_ms, feature_values in zip(candidate_positions_ms, candidate_features):
                row = dict(zip(rankers.PHASE_FEATS, feature_values))
                row.update(dict(zip(PERIODICITY_FEATURE_NAMES, periodicity_features(events, anchor_seconds + position_ms / 1000.0 + rankers.CALIBRATION_MILLISECONDS / 1000.0 * 0, period_seconds, first_section['start_s'], first_section['end_s']))))
                row.update(id=track_id, pos=position_ms, dpos=rankers.wrap((position_ms - true_offset_ms) / 1000.0, period_seconds) * 1000.0)
                row['label'] = int(abs(row['dpos']) <= LABEL_TOLERANCE_MS)
                records.append(row)
    pd.DataFrame(records).to_pickle(output_path); pd.DataFrame(track_info).to_pickle(output_path.replace('.pkl', '_info.pkl'))
    table = pd.DataFrame(records); candidate_coverage = table.groupby('id').label.max(); ours_coverage = table[table.is_ours == 1].groupby('id').label.max()
    print(table.shape, table.id.nunique(), 'cand covers truth', round(candidate_coverage.mean(), 4), 'ours ok', round(ours_coverage.mean(), 4), 'ours wrong', int((ours_coverage == 0).sum()))


if __name__ == '__main__':
    build(sys.argv[1])
