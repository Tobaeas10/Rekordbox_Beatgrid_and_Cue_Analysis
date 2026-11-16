"""cue_predictor.py: production cue (drop/fakeout) predictor (no training data needed).
  grid_features(audio, beats)  -> track dict (keys BE, BX, pl, bpm) + grid info, features computed on the STORED grid (PQTZ beats [(beat_in_bar,tempo_x100,ms)])
  predict(t)                   -> dict(D,e1,H,e2,fk1,fk2,conf_*) in BARS from the stored grid's first downbeat (bar 0), or None
  cues_for(pred)               -> (hot{'A'..'H':bar}, mem[bars])  via cue_rules.derive
Models: data/models/cues/drop_start_model.pkl, drop_end_model.pkl (cand6), fakeout_end_model.pkl."""
import pickle
import numpy as np
from lib import paths
LENGTH_PENALTIES = {16: -0.5, 24: -1.5, 32: 0, 40: -1.5, 48: -0.3, 56: -2, 64: -0.3}
DECODER_PENALTIES = dict(s0=0.0, e0=-1.0, len=LENGTH_PENALTIES, single=-3.0, gap=0.15, b16=0.0)   # identical to cdec6.py
_MODELS = {}
def models():
    if not _MODELS:
        for model_key, file_name in (('s', 'drop_start_model.pkl'), ('e', 'drop_end_model.pkl')): _MODELS[model_key] = pickle.load(open(paths.MODELS_CUES / file_name, 'rb'))
    return _MODELS
def log_odds(probabilities): probabilities = np.clip(probabilities, 1e-3, 1 - 1e-3); return np.log(probabilities / (1 - probabilities))
def decode_phase(start_scores, end_scores, bar_count, bar_phase, penalties):
    candidate_bars = np.arange(bar_phase, bar_count + 1, 8); candidate_count = len(candidate_bars)
    if candidate_count < 2: return (-1e9, None)
    start_scores = start_scores[candidate_bars] - penalties['s0']; end_scores = end_scores[candidate_bars] - penalties['e0']; best = (-1e9, None); pair_scores = {}
    for i in range(candidate_count):
        for j in range(i + 1, candidate_count):
            length_bars = int(candidate_bars[j] - candidate_bars[i])
            if length_bars > 128: break
            if length_bars < 16: continue
            pair_scores[(i, j)] = start_scores[i] + end_scores[j] + penalties['len'].get(length_bars, -3) + (penalties['b16'] if length_bars % 16 == 0 else 0)
    for (first_start_index, first_end_index), first_pair_score in pair_scores.items():
        if first_pair_score + penalties['single'] > best[0]: best = (first_pair_score + penalties['single'], (int(candidate_bars[first_start_index]), int(candidate_bars[first_end_index]), None, None))
        for second_start_index in range(first_end_index, candidate_count):
            if candidate_bars[second_start_index] - candidate_bars[first_end_index] < 8: continue
            for second_end_index in range(second_start_index + 1, candidate_count):
                if (second_start_index, second_end_index) in pair_scores:
                    total_score = first_pair_score + pair_scores[(second_start_index, second_end_index)] - penalties['gap'] * abs(candidate_bars[second_start_index] - candidate_bars[first_end_index] - 16) / 8
                    if total_score > best[0]: best = (total_score, (int(candidate_bars[first_start_index]), int(candidate_bars[first_end_index]), int(candidate_bars[second_start_index]), int(candidate_bars[second_end_index])))
    return best
def decode(start_probabilities, end_probabilities, bar_count, penalties=DECODER_PENALTIES):
    start_scores = log_odds(start_probabilities); end_scores = log_odds(end_probabilities)
    return max([decode_phase(start_scores, end_scores, bar_count, bar_phase, penalties) for bar_phase in range(8)], key=lambda candidate: candidate[0])[1]

def grid_info(beats):
    times = [beat[2] for beat in beats]
    first_downbeat_index = next((beat_index for beat_index, beat in enumerate(beats) if beat[0] == 1), None)
    tempos = {beat[1] for beat in beats}
    return times, first_downbeat_index, len(tempos) > 1

def bar_ms(times, first_downbeat_index, bar):
    """time in ms of (fractional) bar from the stored grid's first downbeat (constant tempo assumed)."""
    period_ms = (times[-1] - times[0]) / (len(times) - 1)
    return times[0] + (first_downbeat_index + 4 * bar) * period_ms

def grid_features(audio, beats, playlists):
    from lib.beatgrid import tempo_detection
    from lib import audio_features
    times, first_downbeat_index, multi = grid_info(beats)
    if first_downbeat_index is None or multi or len(times) < 64: return None, 'no usable constant grid'
    period_seconds = (times[-1] - times[0]) / (len(times) - 1) / 1000.0; anchor_seconds = times[0] / 1000.0
    samples, sample_rate = tempo_detection.load_audio(audio); duration_seconds = len(samples) / sample_rate
    envelope, frame_seconds = audio_features.band_envelopes_1ms(samples, sample_rate); envelope_frame_count = envelope.shape[1]
    beat_energy_table, first_beat_time = audio_features.beat_energies(envelope, frame_seconds, anchor_seconds, period_seconds, duration_seconds); del envelope
    beat_energy_table = beat_energy_table.astype(np.float32)
    if sample_rate == 44100: resampled_samples, resampled_rate = samples, sample_rate
    else: resampled_samples, resampled_rate = tempo_detection.load_audio(audio, 44100)
    band_energy_table = audio_features.compute_band_energies(resampled_samples, resampled_rate, first_beat_time, period_seconds, len(beat_energy_table))
    leading_beat_count = int(np.ceil(-anchor_seconds / period_seconds)); origin = first_downbeat_index - leading_beat_count        # lattice index of the stored bar 0
    if origin < 0:
        beat_energy_table = np.vstack([np.repeat(beat_energy_table[:1], -origin, 0), beat_energy_table]); band_energy_table = np.concatenate([np.repeat(band_energy_table[:1], -origin, 0), band_energy_table]); origin = 0
    beat_energy_table = beat_energy_table[origin:]; band_energy_table = band_energy_table[origin:]
    if len(beat_energy_table) < 16 * 8: return None, 'track too short'
    return dict(BE=beat_energy_table, BX=band_energy_table, pl=list(playlists), bpm=60.0 / period_seconds, T=period_seconds, phase=0, length=duration_seconds), None

def predict(track):
    from lib.cues.drop_features import track_features
    from lib.cues.fakeout_predictor import predict_fakeout_end
    candidates, bar_count, features = track_features(track); drop_models = models()
    start_probabilities = drop_models['s'].predict_proba(features)[:, 1]; end_probabilities = drop_models['e'].predict_proba(features)[:, 1]
    decoded = decode(start_probabilities, end_probabilities, bar_count)
    if decoded is None: return None
    first_drop_start, first_drop_end, second_drop_start, second_drop_end = decoded
    prediction = dict(D=first_drop_start, e1=first_drop_end, H=second_drop_start, e2=second_drop_end, conf_D=float(start_probabilities[first_drop_start]), conf_e1=float(end_probabilities[first_drop_end]), conf_H=None if second_drop_start is None else float(start_probabilities[second_drop_start]), conf_e2=None if second_drop_end is None else float(end_probabilities[second_drop_end]))
    prediction['fk1'], prediction['pfk1'] = predict_fakeout_end(track, first_drop_start)
    if second_drop_start is not None: prediction['fk2'], prediction['pfk2'] = predict_fakeout_end(track, second_drop_start, is_second_drop=True)
    else: prediction['fk2'], prediction['pfk2'] = None, 0.0
    # a fakeout end must lie before the drop end
    for key, drop_end_bar in (('fk1', first_drop_end), ('fk2', second_drop_end)):
        if prediction[key] is not None and drop_end_bar is not None and prediction[key] >= drop_end_bar: prediction[key] = None
    return prediction

def cues_for(prediction):
    from lib.cues.cue_rules import derive
    return derive(prediction['D'], prediction['e1'], prediction['H'], prediction['e2'], prediction.get('fk1'), prediction.get('fk2'))
