"""Learned candidate rankers (numpy only at run time).

phase_candidates(ev, sec)  -> candidate peaks of the folded low/mid novelty with a feature row each (same code for training and use)
downbeat_candidates(ev, sec, t0) -> one feature row per beat-1 phase q in 0..3
GB            -> gradient-boosted tree ensemble exported from sklearn HistGradientBoostingClassifier (JSON), evaluated with numpy
"""
import json
import numpy as np

CALIBRATION_MILLISECONDS = 4.0          # mid-band fold peak lies this much after the visible attack
MAX_CANDIDATES = 8


def wrap(values, period):
    return (values + period / 2) % period - period / 2


def _fold(novelty, step_seconds, period_seconds, anchor_seconds, window_start, window_end, slot_count):
    length = len(novelty)
    first_index, last_index = max(0, int(window_start / step_seconds)), min(length, int(window_end / step_seconds) + 1)
    times = np.arange(first_index, last_index) * step_seconds
    slot_index = (((times - anchor_seconds) % period_seconds) / period_seconds * slot_count).astype(int) % slot_count
    return np.bincount(slot_index, weights=novelty[first_index:last_index].astype(np.float64), minlength=slot_count)


def _smooth(histogram, window):
    padding = window // 2
    return np.convolve(np.concatenate([histogram[-padding:], histogram, histogram[:padding]]), np.ones(window) / window, 'valid')


def _peaks(histogram, threshold):
    previous_values, next_values = np.roll(histogram, 1), np.roll(histogram, -1)
    return [int(index) for index in np.nonzero((histogram >= threshold) & (histogram >= previous_values) & (histogram > next_values))[0]]


def _at(histogram, center, half_width):
    length = len(histogram)
    return float(histogram[[(center + offset) % length for offset in range(-half_width, half_width + 1)]].max())


def _ring_mean(histogram, center, offsets):
    length = len(histogram)
    return float(np.mean(histogram[[(center + offset) % length for offset in offsets]]))


def _share(event_times, event_weights, anchor_seconds, period_seconds, center_milliseconds, tolerance_milliseconds):
    """(weight share, count) of events within tol_ms of the lattice shifted by center_ms."""
    if len(event_times) == 0 or event_weights.sum() <= 0:
        return 0.0, 0
    within_tolerance = np.abs(wrap(event_times - anchor_seconds - center_milliseconds / 1000.0, period_seconds)) * 1000.0 <= tolerance_milliseconds
    return float(event_weights[within_tolerance].sum() / event_weights.sum()), int(within_tolerance.sum())


PHASE_FEATS = ['is_ours', 'dist', 'fm', 'fl', 'fm_half', 'fl_half', 'fm_rank', 'fm_sharp', 'fl_sharp', 'kshare', 'sshare', 'kn', 'sn',
               'kshare_half', 'sshare_half', 'ours_k', 'ours_s', 'ours_kn', 'ours_sn', 'T', 'rms', 'lnk', 'lns', 'mid_strength', 'low_strength']


def phase_candidates(events, section, max_candidates=MAX_CANDIDATES):
    """Candidates for the beat phase of one section. Positions are ms relative to sec['anchor_s'] (calibrated to the visible attack).
    Returns (positions[ms], X[n, len(PHASE_FEATS)]) or (None, None) when the cache lacks the novelty arrays."""
    if 'nov_low' not in events:
        return None, None
    period_seconds, anchor_seconds, section_start, section_end = section['period_s'], section['anchor_s'], section['start_s'], section['end_s']
    slot_count = int(round(period_seconds * 1000))
    step_seconds = events['nov_dt']
    mid_fold = _smooth(_fold(events['nov_mid'], step_seconds, period_seconds, anchor_seconds, section_start, section_end, slot_count), 3)
    low_fold = _smooth(_fold(events['nov_low'], step_seconds, period_seconds, anchor_seconds, section_start, section_end, slot_count), 3)
    if mid_fold.max() <= 0 or low_fold.max() <= 0:
        return None, None
    mid_fold_normalized, low_fold_normalized = mid_fold / mid_fold.max(), low_fold / low_fold.max()
    combined_strength = lambda slot: mid_fold_normalized[slot] + 0.5 * low_fold_normalized[slot]
    candidate_slots = set(_peaks(mid_fold_normalized, 0.25)) | set(_peaks(low_fold_normalized, 0.45))
    kept_slots = sorted([slot for slot in candidate_slots if slot != 0], key=lambda slot: -combined_strength(slot))[:max_candidates - 1]
    selected_slots = []
    for slot in [0] + kept_slots:
        if all(min(abs(slot - other_slot), slot_count - abs(slot - other_slot)) >= 5 for other_slot in selected_slots):
            selected_slots.append(slot)
    in_section = (events['t'] >= section_start) & (events['t'] <= section_end)
    event_times, event_weights, event_classes = events['t'][in_section], events['w'][in_section], events['c'][in_section]
    kick_times, kick_weights, snare_times, snare_weights = event_times[event_classes == 0], event_weights[event_classes == 0], event_times[event_classes == 1], event_weights[event_classes == 1]
    ours_k, ours_s = _share(kick_times, kick_weights, anchor_seconds, period_seconds, 0.0, 4), _share(snare_times, snare_weights, anchor_seconds, period_seconds, 0.0, 4)
    mid_median, low_median = float(np.median(mid_fold)) + 1e-12, float(np.median(low_fold)) + 1e-12
    rank_order = sorted(selected_slots, key=lambda slot: -mid_fold_normalized[slot])
    far_offsets = [offset for offset in range(-25, 26, 5) if abs(offset) >= 10]
    positions, feature_rows = [], []
    for slot in selected_slots:
        position_ms = wrap(slot / slot_count * period_seconds, period_seconds) * 1000.0 - CALIBRATION_MILLISECONDS
        half = (slot + slot_count // 2) % slot_count
        kick_share, kick_count = _share(kick_times, kick_weights, anchor_seconds, period_seconds, position_ms, 4)
        snare_share, snare_count = _share(snare_times, snare_weights, anchor_seconds, period_seconds, position_ms, 4)
        kick_share_half, _ = _share(kick_times, kick_weights, anchor_seconds, period_seconds, position_ms + period_seconds * 500.0, 4)
        snare_share_half, _ = _share(snare_times, snare_weights, anchor_seconds, period_seconds, position_ms + period_seconds * 500.0, 4)
        feature_rows.append([float(abs(position_ms) <= 6), abs(position_ms), _at(mid_fold_normalized, slot, 2), _at(low_fold_normalized, slot, 3), _at(mid_fold_normalized, half, 3), _at(low_fold_normalized, half, 3), rank_order.index(slot),
                     mid_fold_normalized[slot] / (_ring_mean(mid_fold_normalized, slot, far_offsets) + 1e-9), low_fold_normalized[slot] / (_ring_mean(low_fold_normalized, slot, far_offsets) + 1e-9), kick_share, snare_share, np.log1p(kick_count), np.log1p(snare_count),
                     kick_share_half, snare_share_half, ours_k[0], ours_s[0], np.log1p(ours_k[1]), np.log1p(ours_s[1]), period_seconds * 1000.0, section.get('rms_ms', 3.0),
                     np.log1p(events['nk']), np.log1p(events['ns']), mid_fold.max() / mid_median, low_fold.max() / low_median])
        positions.append(position_ms)
    return np.array(positions), np.array(feature_rows, float)


class GB:
    """Gradient-boosted trees (sklearn HistGradientBoostingClassifier, binary) exported with export_hgb()."""

    def __init__(self, exported):
        self.base = exported['base']
        self.trees = [{key: np.array(value) for key, value in tree.items()} for tree in exported['trees']]

    @classmethod
    def load(cls, path):
        with open(path) as file:
            return cls(json.load(file))

    def raw(self, features):
        features = np.asarray(features, float)
        raw_scores = np.full(len(features), self.base)
        for tree in self.trees:
            node = np.zeros(len(features), int)
            while True:
                leaf = tree['leaf'][node]
                if leaf.all():
                    break
                feature_indices = tree['feat'][node]
                go_left = features[np.arange(len(features)), feature_indices] <= tree['thr'][node]
                next_nodes = np.where(go_left, tree['left'][node], tree['right'][node])
                node = np.where(leaf, node, next_nodes)
            raw_scores += tree['val'][node]
        return raw_scores

    def proba(self, features):
        return 1.0 / (1.0 + np.exp(-self.raw(features)))


def export_hgb(model):
    """sklearn HistGradientBoostingClassifier (binary) -> dict for GB."""
    trees = []
    for predictor in model._predictors:
        nodes = predictor[0].nodes
        trees.append(dict(feat=nodes['feature_idx'].astype(int).tolist(), thr=nodes['num_threshold'].astype(float).tolist(),
                          left=nodes['left'].astype(int).tolist(), right=nodes['right'].astype(int).tolist(),
                          val=nodes['value'].astype(float).tolist(), leaf=nodes['is_leaf'].astype(bool).tolist()))
    return dict(base=float(np.ravel(model._baseline_prediction)[0]), trees=trees)


# ---------------------------------------------------------------- downbeat

SIGNALS = ['kick', 'snare', 'bass', 'low', 'mid', 'en4', 'en8']
DOWNBEAT_FEATS = ['is_q0'] + [f'{s}{r}' for s in SIGNALS for r in range(4)] + ['entry', 'entry_rel', 'nbeats', 'ksum', 'ssum']


def _beat_signals(events, section, first_beat_time):
    """Per-beat evidence arrays over the section (beat index n counted from t0, the first grid beat)."""
    period_seconds = section['period_s']
    first_beat_number = max(0, int(np.ceil((section['start_s'] - first_beat_time) / period_seconds - 1e-9)))
    last_beat_number = int(np.floor((section['end_s'] - first_beat_time) / period_seconds + 1e-9))
    beat_numbers = np.arange(first_beat_number, last_beat_number + 1)
    if len(beat_numbers) < 8:
        return None
    beat_times = first_beat_time + beat_numbers * period_seconds
    event_times, event_weights, event_classes = events['t'], events['w'], events['c']
    kick_per_beat, snare_per_beat = np.zeros(len(beat_numbers)), np.zeros(len(beat_numbers))
    nearest_beat_number = np.rint((event_times - first_beat_time) / period_seconds).astype(int)
    is_on_beat = (np.abs((event_times - first_beat_time) / period_seconds - nearest_beat_number) < 0.12) & (nearest_beat_number >= first_beat_number) & (nearest_beat_number <= last_beat_number)
    np.add.at(kick_per_beat, nearest_beat_number[is_on_beat & (event_classes == 0)] - first_beat_number, event_weights[is_on_beat & (event_classes == 0)])
    np.add.at(snare_per_beat, nearest_beat_number[is_on_beat & (event_classes == 1)] - first_beat_number, event_weights[is_on_beat & (event_classes == 1)])
    signals = dict(ns=beat_numbers, kick=kick_per_beat, snare=snare_per_beat)
    bass = events.get('bass')
    if bass is not None:
        rate = events.get('bass_rate', 100.0)
        bass_log = np.log(bass.astype(np.float32) + 1e-4)
        flux = np.concatenate([np.zeros(3, np.float32), np.maximum(bass_log[3:] - bass_log[:-3], 0)])
        bass_flux = np.zeros(len(beat_numbers))
        for beat_slot, time_seconds in enumerate(beat_times):
            first_frame, last_frame = int((time_seconds - 0.02) * rate), int((time_seconds + 0.04) * rate) + 1
            if first_frame >= 0 and last_frame < len(flux):
                bass_flux[beat_slot] = flux[first_frame:last_frame].max()
        signals['bass'] = bass_flux
        # per-beat log energy for the structural novelty
        log_energy = np.zeros(len(beat_numbers) + 16)
        for beat_slot in range(-8, len(beat_numbers) + 8):
            time_seconds = first_beat_time + (first_beat_number + beat_slot) * period_seconds
            first_frame, last_frame = int(time_seconds * rate), int((time_seconds + period_seconds) * rate) + 1
            log_energy[beat_slot + 8] = np.log(bass[max(0, first_frame):max(1, min(len(bass), last_frame))].astype(np.float32).mean() + 1e-4) if 0 <= first_frame < len(bass) else np.nan
        log_energy = np.where(np.isnan(log_energy), np.nanmedian(log_energy) if np.isfinite(np.nanmedian(log_energy)) else 0.0, log_energy)
        for signal_name, span_beats in (('en4', 4), ('en8', 8)):
            increase = np.zeros(len(beat_numbers))
            for beat_slot in range(len(beat_numbers)):
                shifted_index = beat_slot + 8
                increase[beat_slot] = max(0.0, log_energy[shifted_index:shifted_index + span_beats].mean() - log_energy[shifted_index - span_beats:shifted_index].mean())
            signals[signal_name] = increase
    for signal_name in ('low', 'mid'):
        novelty_array = events.get('nov_' + signal_name)
        if novelty_array is None:
            continue
        step_seconds = events['nov_dt']
        cumulative_novelty = np.concatenate([[0.0], np.cumsum(novelty_array.astype(np.float64))])
        first_frame = np.clip(((beat_times - 0.015) / step_seconds).astype(int), 0, len(novelty_array))
        last_frame = np.clip(((beat_times + 0.06) / step_seconds).astype(int), 0, len(novelty_array))
        signals[signal_name] = cumulative_novelty[last_frame] - cumulative_novelty[first_frame]
    return signals


def downbeat_candidates(events, section, first_beat_time):
    """Feature rows for q = 0..3 (beat index of beat 1 counted from the first grid beat t0). Each signal enters as its share per beat
    position relative to q (position 0 = candidate beat 1), so trees can learn kick-on-1 / snare-on-2+4 / snare-on-3 patterns."""
    signals = _beat_signals(events, section, first_beat_time)
    if signals is None:
        return None
    beat_numbers = signals['ns']
    shares = {}
    for signal_name in SIGNALS:
        values = signals.get(signal_name)
        if values is None:
            values = np.zeros(len(beat_numbers))
        sums_by_position = np.array([values[beat_numbers % 4 == position].sum() for position in range(4)])
        shares[signal_name] = sums_by_position / (sums_by_position.sum() + 1e-9)
    # first drum entry: first beat with kick or low-band evidence above a fraction of its typical level
    entry_evidence = signals.get('kick', 0) / (np.percentile(signals['kick'], 90) + 1e-9) + signals.get('low', np.zeros(len(beat_numbers))) / (np.percentile(signals.get('low', np.ones(len(beat_numbers))), 90) + 1e-9)
    evidence_indices = np.nonzero(entry_evidence >= 0.5)[0]
    entry = int(beat_numbers[evidence_indices[0]]) if len(evidence_indices) else 0
    feature_rows = []
    for candidate_position in range(4):
        row = [float(candidate_position == 0)]
        for signal_name in SIGNALS:
            row += [float(shares[signal_name][(candidate_position + offset) % 4]) for offset in range(4)]
        row += [float(entry % 4 == candidate_position), float(((entry - candidate_position) % 4)), float(len(beat_numbers)), float(signals['kick'].sum()), float(signals['snare'].sum())]
        feature_rows.append(row)
    return np.array(feature_rows, float)


class LRModel:
    """Logistic regression over downbeat_candidates rows (standardised features), numpy only."""
    def __init__(self, exported):
        self.mu, self.sd, self.coef, self.b = (np.array(exported['mu']), np.array(exported['sd']), np.array(exported['coef']), float(exported['b']))
        self.margin = float(exported.get('margin', 0.3))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path)))

    def proba(self, features):
        logit = ((np.asarray(features, float) - self.mu) / self.sd) @ self.coef + self.b
        return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------- fine offset

FINE_LIMIT_MILLISECONDS = 25
FINE_MISSING_VALUE = -99.0


def _fine_fold(novelty, step_seconds, anchor_seconds, period_seconds, section_start, section_end, limit=FINE_LIMIT_MILLISECONDS):
    period_indices = np.arange(int(np.ceil((section_start - anchor_seconds) / period_seconds)), int(np.floor((section_end - anchor_seconds) / period_seconds)) + 1)
    length = len(novelty)
    first_samples = np.rint((anchor_seconds + period_indices * period_seconds - limit / 1000.0) / step_seconds).astype(int)
    is_inside = (first_samples >= 0) & (first_samples + 2 * limit + 1 <= length)
    if not is_inside.any():
        return np.zeros(2 * limit + 1), 0
    sample_indices = first_samples[is_inside][:, None] + np.arange(2 * limit + 1)[None, :]
    return novelty[sample_indices].mean(0), int(is_inside.sum())


def _mode(residuals, weights, limit=FINE_LIMIT_MILLISECONDS, sigma=1.5):
    within_limit = np.abs(residuals) < limit
    residuals, weights = residuals[within_limit], weights[within_limit]
    if len(residuals) < 8:
        return FINE_MISSING_VALUE, int(len(residuals)), FINE_MISSING_VALUE
    candidate_offsets = np.arange(-limit, limit, 0.25)
    density = (weights[None, :] * np.exp(-0.5 * ((candidate_offsets[:, None] - residuals[None, :]) / sigma) ** 2)).sum(1)
    return float(candidate_offsets[np.argmax(density)]), int(len(residuals)), float(np.average(residuals, weights=weights))


def fine_features(events, section):
    """Feature vector describing the residual structure around the final anchor (events, 1 ms folded low/mid novelty)."""
    period_seconds, anchor_seconds = section['period_s'], section['anchor_s']
    event_times, event_weights, event_classes = events['t'], events['w'], events['c']
    low_threshold_times = events.get('tl', event_times)
    in_section = (event_times >= section['start_s']) & (event_times <= section['end_s'])
    half = period_seconds / 2
    features = []
    for times in (event_times, low_threshold_times):
        times_in_section, weights_in_section, classes_in_section = times[in_section], event_weights[in_section], event_classes[in_section]
        residual_milliseconds = (times_in_section - (anchor_seconds + np.round((times_in_section - anchor_seconds) / half) * half)) * 1000.0
        for event_class in (1, 0):
            selection = classes_in_section == event_class
            mode_milliseconds, count, mean_milliseconds = _mode(residual_milliseconds[selection], weights_in_section[selection])
            features += [mode_milliseconds, float(count), mean_milliseconds]
    mid_profile, fold_count = _fine_fold(events['nov_mid'], events['nov_dt'], anchor_seconds, period_seconds, section['start_s'], section['end_s'])
    low_profile, _ = _fine_fold(events['nov_low'], events['nov_dt'], anchor_seconds, period_seconds, section['start_s'], section['end_s'])
    mid_profile, low_profile = mid_profile / (mid_profile.max() + 1e-9), low_profile / (low_profile.max() + 1e-9)
    refine_info = section.get('refine') or {}
    features += [float(section.get('fit_rms_ms', 0.0)), 60.0 / period_seconds, float(fold_count), float(refine_info.get('shift_ms', 0.0)), float(refine_info.get('rejected_shift_ms', 0.0)),
          float(np.argmax(mid_profile) - FINE_LIMIT_MILLISECONDS), float(np.argmax(low_profile) - FINE_LIMIT_MILLISECONDS), float(np.argmax(mid_profile > 0.5) - FINE_LIMIT_MILLISECONDS), float(np.argmax(low_profile > 0.5) - FINE_LIMIT_MILLISECONDS)]
    features += list(mid_profile[::2]) + list(low_profile[::2])
    return np.array(features, float)
