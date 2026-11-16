"""Advanced analysis mode (opt-in, slower, more accurate timing): a second fine-adjust model that looks at the bar-folded raw audio.

Modes (mode()): auto (default) = basic, plus this model only when needs() says the basic pass is unsure (~16% of tracks); advanced (--force-advanced) = every track; basic (--basic-only) = never. The advanced step
decodes the audio once more, cuts 1 ms band-envelope / 0.25 ms windows around the half-beat lattice of each section (audio_features.windows), folds them
over the beats (median over beats = what repeats survives, rhythmically varying synths drop out) and predicts the residual anchor shift with
0.5 * HGB + 0.5 * mean(3 MLPs) (advanced_fine_adjust_model.json). CV (1682 tracks, dev/model_training/beatgrid): tracks with |error| > 5 ms 29 -> 20, > 8 ms 14 -> 6.
The shift REPLACES the basic fine shift of the section (same dead zone / cap); basic shift stays in sec['fine_basic_ms'].
"""
from lib import paths
import os
import numpy as np

ADVANCED_FINE_ADJUST_MODEL = None
MIN_SECTION_SECONDS = 30.0
ENABLED_ENV = 'BEATGRID_ADVANCED_ENABLED'


ADVANCED_TRIGGER_RAW_MILLISECONDS = 0.5   # basic fine model unsure: |raw prediction| >= this (15.6% of tracks; 8 of the 10 tracks advanced fixes in CV, and the 1 it breaks)
ADVANCED_TRIGGER_RMS_MILLISECONDS = 3.5   # loose lattice fit


def mode():
    """'auto' (default): basic analysis, then the advanced fine model only for tracks the basic pass is unsure about (needs());
    'advanced': advanced model on every track (--force-advanced); 'basic': never (--basic-only). Env BEATGRID_ANALYSIS_MODE; legacy BEATGRID_ADVANCED_ENABLED=1 = advanced."""
    analysis_mode = os.environ.get('BEATGRID_ANALYSIS_MODE', '').strip().lower()
    if analysis_mode in ('auto', 'basic', 'advanced'):
        return analysis_mode
    if os.environ.get(ENABLED_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return 'advanced'
    return 'auto'


def needs(sections):
    """Second-pass trigger from the basic result: any section >= MIN_SECTION_S whose basic fine model is unsure or whose fit is loose."""
    for section in sections or []:
        if section is None or section['end_s'] - section['start_s'] < MIN_SECTION_SECONDS:
            continue
        raw = section.get('fine_raw_ms')
        if (raw is not None and abs(raw) >= ADVANCED_TRIGGER_RAW_MILLISECONDS) or float(section.get('fit_rms_ms', section.get('rms_ms', 0.0))) >= ADVANCED_TRIGGER_RMS_MILLISECONDS:
            return True
    return False


def _crossing(profile, fraction):
    peak_index = int(np.argmax(profile))
    peak_value = profile[peak_index]
    crossing_index = peak_index
    while crossing_index > 0 and profile[crossing_index] > fraction * peak_value:
        crossing_index -= 1
    return crossing_index - 30


def _profile_features(profiles):
    features = []
    for band in range(4):
        profile = profiles[band]
        peak_index = int(np.argmax(profile))
        features += [peak_index - 30, _crossing(profile, 0.5), _crossing(profile, 0.25), profile[peak_index], np.median(profile[:20]), profile[30]]
    return features


def window_features(windows, beat_indices, parities, raw_windows):
    """(eng, prof) from audio_features windows; identical to the training code (dev/model_training/beatgrid/build_advanced_fine_adjust_rows.py). None when too few on-beat windows."""
    windows = np.asarray(windows, np.float32)
    raw_windows_float = np.asarray(raw_windows, np.float32)
    is_on_beat = np.asarray(parities) == 0
    if is_on_beat.sum() < 12:
        return None
    on_beat_windows = windows[is_on_beat]
    beat_in_bar = (np.asarray(beat_indices)[is_on_beat] // 2) % 4
    on_beat_raw_windows = raw_windows_float[is_on_beat]
    all_median = np.median(on_beat_windows, 0)
    kick_scores = [np.median(on_beat_windows[beat_in_bar == beat_position][:, 1, 25:36].max(1)) if (beat_in_bar == beat_position).sum() >= 3 else -1 for beat_position in range(4)]
    beat_position = int(np.argmax(kick_scores))
    kick_median = np.median(on_beat_windows[beat_in_bar == beat_position], 0)
    raw_kick_median = np.median(on_beat_raw_windows[beat_in_bar == beat_position], 0)
    peak_offsets = on_beat_windows[beat_in_bar == beat_position][:, 1, :].argmax(1) - 30
    persistence = [np.std(peak_offsets), np.mean(np.abs(peak_offsets - np.median(peak_offsets)) <= 3)]
    second_peak_offsets = on_beat_windows[:, 2, :].argmax(1) - 30
    persistence += [np.std(second_peak_offsets), np.mean(np.abs(second_peak_offsets - np.median(second_peak_offsets)) <= 3)]
    summary_features = _profile_features(all_median) + _profile_features(kick_median) + persistence + [kick_scores[beat_position] - sorted(kick_scores)[-2], beat_position]
    profile_features = np.concatenate([all_median.ravel() / (all_median.max() + 1e-6), kick_median.ravel() / (kick_median.max() + 1e-6), raw_kick_median.reshape(-1, 4).mean(1) / (raw_kick_median.max() + 1e-6)])
    return np.asarray(summary_features, float), profile_features


class Model:
    def __init__(self, exported):
        from lib.beatgrid import rankers as ranker
        self.hgb = ranker.GB(exported['hgb'])
        self.mlps = [dict(mean=np.array(mlp['mean']), scale=np.array(mlp['scale']), W=[np.array(weight) for weight in mlp['W']], b=[np.array(bias) for bias in mlp['b']]) for mlp in exported['mlps']]
        self.w_mlp = float(exported.get('w_mlp', 0.5))

    def predict(self, features):
        features = np.nan_to_num(np.asarray(features, float))
        hgb_raw = self.hgb.raw(features)
        mlp_predictions = []
        for mlp in self.mlps:
            activation = np.clip((features - mlp['mean']) / mlp['scale'], -5, 5)
            for layer_index, (weight, bias) in enumerate(zip(mlp['W'], mlp['b'])):
                activation = activation @ weight + bias
                if layer_index < len(mlp['W']) - 1:
                    activation = np.maximum(activation, 0)
            mlp_predictions.append(activation.ravel())
        return self.w_mlp * np.mean(mlp_predictions, 0) + (1 - self.w_mlp) * hgb_raw


def load_model(path=None):
    global ADVANCED_FINE_ADJUST_MODEL
    import json
    path = path or str(paths.MODELS_BEATGRID / 'advanced_fine_adjust_model.json')
    with open(path) as file:
        ADVANCED_FINE_ADJUST_MODEL = Model(json.load(file))
    return ADVANCED_FINE_ADJUST_MODEL


def apply(events, sections, audio):
    """Replace the basic fine shift of every section >= MIN_SECTION_S by the advanced prediction. Returns secs (edited in place).
    Any failure leaves the section with its basic result."""
    from lib import audio_features
    from lib.beatgrid import rankers as ranker
    from lib.beatgrid import tempo_detection
    from lib.beatgrid import grid_pipeline
    try:
        model = ADVANCED_FINE_ADJUST_MODEL or load_model()
        samples, sample_rate = tempo_detection.load_audio(audio)
    except Exception:
        return sections
    for section in sections:
        try:
            if section is None or section['end_s'] - section['start_s'] < MIN_SECTION_SECONDS:
                continue
            basic = float(section.get('fine_shift_ms') or 0.0)
            pre_adjusted_section = dict(section)
            pre_adjusted_section['anchor_s'] = section['anchor_s'] - basic / 1000.0
            window_data, _, _, _, _ = audio_features.windows(events, pre_adjusted_section, pre_adjusted_section['anchor_s'], samples, sample_rate)
            window_feature_result = window_features(window_data['W'], window_data['Wk'], window_data['Wpar'], window_data['Wraw'])
            if window_feature_result is None:
                continue
            feature_vector = np.concatenate([ranker.fine_features(events, pre_adjusted_section), window_feature_result[0], window_feature_result[1]])
            shift_ms = float(model.predict(feature_vector[None, :])[0])
        except Exception:
            continue
        section['fine_basic_ms'] = round(basic, 2)
        section['fine_adv_raw_ms'] = round(shift_ms, 3)
        if abs(shift_ms) < grid_pipeline.FINE_DEADZONE_MS:
            shift_ms = 0.0
        shift_ms = max(-grid_pipeline.FINE_CAP_MS, min(grid_pipeline.FINE_CAP_MS, shift_ms))
        section['anchor_s'] = pre_adjusted_section['anchor_s'] + shift_ms / 1000.0
        section['fine_shift_ms'] = round(shift_ms, 2)
        section['fine_raw_ms'] = section['fine_adv_raw_ms']
        section['fine_adv'] = True
    return sections


# ------------------------------------------------------------------ downbeat (bar-folded per-beat band energies)
DOWNBEAT_MODEL = None
DOWNBEAT_TRIGGER_MARGIN = 0.9   # basic downbeat margin below this -> run the advanced downbeat model (7.9% of tracks, covers 25 of the 29 tracks it fixes in CV)
DOWNBEAT_MIN_SECTION_SECONDS = 30.0


def _position_features(beat_energies, position_indices):
    """BE (n,4,3) per-beat band energies, idx = beat position (0..3) of each beat -> (4, 14): share per position of the median band/phase
    energy (12) + low-band onset consistency (2)."""
    median_energies = np.zeros((4, 12))
    consistency = np.zeros((4, 2))
    global_median = np.median(beat_energies[:, :2, 1], 0) + 1e-9
    for position in range(4):
        beats_at_position = beat_energies[position_indices == position]
        if len(beats_at_position) < 3:
            continue
        median_energies[position] = np.median(beats_at_position, 0).reshape(-1)
        consistency[position] = (beats_at_position[:, :2, 1] > 0.5 * global_median).mean(0)
    shares = median_energies / (median_energies.sum(0, keepdims=True) + 1e-9)
    return np.hstack([shares, consistency])


def downbeat_new_features(beat_energies, beat_energy_start, period_seconds, first_beat_time, start_seconds, end_seconds):
    """(4, 56) rows for q = 0..3 (beat 1 = beat index q counted from t0, the first grid beat of the downbeat frame); None when < 24 beats."""
    beat_energies = np.asarray(beat_energies, np.float32)
    beat_count = len(beat_energies)
    beat_times = beat_energy_start + np.arange(beat_count) * period_seconds
    in_section = (beat_times >= start_seconds) & (beat_times <= end_seconds)
    if in_section.sum() < 24:
        return None
    shift = int(round((beat_energy_start - first_beat_time) / period_seconds))
    positions = ((np.arange(beat_count) + shift) % 4)[in_section]
    position_features = _position_features(beat_energies[in_section], positions)
    return np.stack([np.concatenate([position_features[(candidate_position + offset) % 4] for offset in range(4)]) for candidate_position in range(4)])


class DownbeatModel:
    def __init__(self, exported):
        from lib.beatgrid import rankers as ranker
        self.lr = ranker.LRModel(exported['lr'])
        self.hgb = ranker.GB(exported['hgb'])
        self.margin = float(exported.get('margin', 0.3))

    def proba(self, features):
        features = np.asarray(features, float)
        return 0.5 * self.lr.proba(features) + 0.5 * self.hgb.proba(features)


def load_downbeat_model(path=None):
    global DOWNBEAT_MODEL
    import json
    path = path or str(paths.MODELS_BEATGRID / 'advanced_downbeat_model.json')
    with open(path) as file:
        DOWNBEAT_MODEL = DownbeatModel(json.load(file))
    return DOWNBEAT_MODEL


def wants_downbeat(sections):
    """Second-pass trigger for the downbeat model, from the basic downbeat decision stored in secs[0]['downbeat'] by build_grid."""
    analysis_mode = mode()
    if analysis_mode == 'basic' or not sections:
        return False
    first_section = sections[0]
    if first_section['end_s'] - first_section['start_s'] < DOWNBEAT_MIN_SECTION_SECONDS:
        return False
    if analysis_mode == 'advanced':
        return True
    downbeat_info = first_section.get('downbeat')
    if not downbeat_info or not downbeat_info.get('scores'):
        return False
    scores, decided_position = downbeat_info['scores'], downbeat_info.get('q', 0)
    return scores[decided_position] - max(score for index, score in enumerate(scores) if index != decided_position) < DOWNBEAT_TRIGGER_MARGIN


def attach_band_energies(events, sections, audio):
    """Compute the per-beat band energies of the first section and keep them in ev['adv_be'] for pipeline.pick_downbeat. True on success."""
    from lib import audio_features
    from lib.beatgrid import tempo_detection
    try:
        samples, sample_rate = tempo_detection.load_audio(audio)
        band_envelopes, frame_seconds = audio_features.band_envelopes_1ms(samples, sample_rate)
        first_section = sections[0]
        beat_energies, first_beat_time = audio_features.beat_energies(band_envelopes, frame_seconds, first_section['anchor_s'], first_section['period_s'], events['dur'])
        events['adv_be'] = dict(BE=beat_energies, t0=float(first_beat_time), period_s=float(first_section['period_s']))
        return True
    except Exception:
        return False


def downbeat_proba(events, section, first_beat_time, features):
    """Advanced downbeat probabilities for q = 0..3 or None (no stored energies / too short / model error)."""
    advanced_energies = events.get('adv_be') if events is not None else None
    if not advanced_energies:
        return None
    try:
        new_features = downbeat_new_features(advanced_energies['BE'], advanced_energies['t0'], advanced_energies['period_s'], first_beat_time, section['start_s'], section['end_s'])
        if new_features is None:
            return None
        model = DOWNBEAT_MODEL or load_downbeat_model()
        return model.proba(np.concatenate([features, new_features], 1)), model.margin
    except Exception:
        return None
