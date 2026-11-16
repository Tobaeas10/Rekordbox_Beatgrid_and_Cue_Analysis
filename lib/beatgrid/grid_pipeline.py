if __name__ == "__main__":
    import _bootstrap  # only when this file is run directly; makes `import lib...` work
from lib import paths
import os
import sys
import json
import pickle
import hashlib
import numpy as np
from lib.beatgrid import tempo_detection

BLOCK_SECONDS, HOP_SECONDS = 30.0, 15.0
MIN_Z_SCORE, MIN_EVENT_COUNT = 8.0, 10
BPM_TOLERANCE = 0.25
SNAP_BPM_TOLERANCE = 0.1


ONSET_FRACTION = float(os.environ.get('BEATGRID_ONSET_FRACTION', 0.25))
VERSION = 'session9'
REFINE_FRACTION = float(os.environ.get('BEATGRID_REFINE_FRACTION', 0.1))
CACHE_TAG = ('' if ONSET_FRACTION == 0.25 else f'_of{ONSET_FRACTION:g}') + ('' if REFINE_FRACTION == 0.1 else f'_rf{REFINE_FRACTION:g}')


def get_events(path, cache=True, onset_fraction=None):
    onset_fraction = ONSET_FRACTION if onset_fraction is None else onset_fraction
    cache_path = path + f'.events2{CACHE_TAG}.pkl'
    if cache and os.path.exists(cache_path):
        return pickle.load(open(cache_path, 'rb'))
    samples, sample_rate = tempo_detection.load_audio(path)
    kicks = tempo_detection.detect_kicks(samples, sample_rate, onset_fraction=onset_fraction)
    snares = tempo_detection.detect_snares(samples, sample_rate, onset_fraction=onset_fraction, low_threshold_fraction=REFINE_FRACTION or None)
    bias, pair_count = tempo_detection.estimate_kick_bias(kicks, snares)
    event_times, event_weights, event_classes, low_threshold_times = tempo_detection.make_events(kicks, snares, bias)
    events = dict(t=event_times, w=event_weights, c=event_classes, tl=low_threshold_times, dur=len(samples) / sample_rate, bias=bias, npairs=pair_count, nk=len(kicks), ns=len(snares))
    if cache:
        pickle.dump(events, open(cache_path, 'wb'))
    return events


def block_scans(events):
    event_times, event_weights, event_classes, duration_seconds = events['t'], events['w'], events['c'], events['dur']
    bpm_candidates = np.arange(tempo_detection.BPM_LOWER_LIMIT, tempo_detection.BPM_UPPER_LIMIT, 0.03)
    blocks = []
    block_start = 0.0
    while block_start < duration_seconds - 8:
        in_block = (event_times >= block_start) & (event_times < block_start + BLOCK_SECONDS)
        scan = tempo_detection.scan_block(event_times[in_block], event_weights[in_block], event_classes[in_block], block_start, bpm_candidates)
        is_valid = bool(scan and scan['n'] >= MIN_EVENT_COUNT and scan['z'] >= MIN_Z_SCORE)
        blocks.append(dict(start=block_start, end=min(block_start + BLOCK_SECONDS, duration_seconds), res=scan, valid=is_valid))
        block_start += HOP_SECONDS
    return blocks


def find_sections(blocks, max_sections=4):
    """Group consecutive valid blocks that agree on tempo. Invalid blocks are skipped.
    A section ends at the first valid block that disagrees (two in a row, or one with very high z)."""
    sections, current_section, disagreeing_blocks = [], None, []
    for block in blocks:
        if not block['valid']:
            continue
        bpm, z_score = block['res']['bpm'], block['res']['z']
        if current_section is None:
            current_section = dict(start=block['start'], end=block['end'], bpms=[bpm], blocks=[block]); disagreeing_blocks = []
            continue
        median_bpm = float(np.median(current_section['bpms']))
        if abs(bpm - median_bpm) < BPM_TOLERANCE:
            current_section['end'] = block['end']; current_section['bpms'].append(bpm); current_section['blocks'].append(block); disagreeing_blocks = []
        else:
            disagreeing_blocks.append(block)
            if len(disagreeing_blocks) >= 2 or z_score >= 15:
                current_section['broke_at'] = disagreeing_blocks[0]['start']
                sections.append(current_section)
                if len(sections) >= max_sections:
                    return sections
                current_section = dict(start=disagreeing_blocks[0]['start'], end=disagreeing_blocks[-1]['end'], bpms=[disagreeing_block['res']['bpm'] for disagreeing_block in disagreeing_blocks], blocks=list(disagreeing_blocks))
                disagreeing_blocks = []
    if current_section is not None:
        sections.append(current_section)
    return sections


def fit_fixed(event_times, event_weights, anchor_seconds, bpm, tolerance=0.012, iteration_count=6):
    """Phase-only fit on the eighth-note lattice at a fixed bpm. Returns (anchor, rms_ms, n_inliers) or None."""
    half = 60.0 / bpm / 2
    for _ in range(iteration_count):
        eighth_indices = np.round((event_times - anchor_seconds) / half)
        residuals = event_times - (anchor_seconds + eighth_indices * half)
        inlier_mask = np.abs(residuals) < tolerance
        if inlier_mask.sum() < 5:
            return None
        anchor_seconds = anchor_seconds + np.average(residuals[inlier_mask], weights=event_weights[inlier_mask])
    eighth_indices = np.round((event_times - anchor_seconds) / half)
    residuals = event_times - (anchor_seconds + eighth_indices * half)
    inlier_mask = np.abs(residuals) < tolerance
    return float(anchor_seconds), float(1000 * np.sqrt(np.mean(residuals[inlier_mask] ** 2))), int(inlier_mask.sum())


def fit_section(events, section):
    event_times, event_weights, event_classes = events['t'], events['w'], events['c']
    median_bpm = float(np.median(section['bpms']))
    seed = section['blocks'][0]
    seed_scan = seed['res']
    in_section = (event_times >= section['start']) & (event_times < section['end'])
    section_bpm, section_anchor, template_name = tempo_detection.refine_window(event_times, event_weights, event_classes, section['start'], section['end'], median_bpm, span=0.25, step=0.002)
    section_bpm, section_anchor, template_name = tempo_detection.refine_window(event_times, event_weights, event_classes, section['start'], section['end'], section_bpm, span=0.01, step=0.0002)
    fit = tempo_detection.lattice_fit(event_times[in_section], event_weights[in_section], section_anchor, section_bpm, tolerance=0.015)
    if fit is None:
        return None
    snapped = False
    if abs(fit['bpm'] - round(fit['bpm'])) <= SNAP_BPM_TOLERANCE:
        fixed_fit = fit_fixed(event_times[in_section], event_weights[in_section], fit['a'], float(round(fit['bpm'])))
        if fixed_fit:
            fit = dict(bpm=float(round(fit['bpm'])), a=fixed_fit[0], rms_ms=fixed_fit[1], n_inl=fixed_fit[2])
            snapped = True
    beat_period = 60.0 / fit['bpm']
    # drift check: median residual per 15 s window
    half = beat_period / 2
    on_lattice = in_section & (np.abs(((event_times - fit['a'] + half / 2) % half) - half / 2) < 0.012)
    lattice_times = event_times[on_lattice]; residuals = ((lattice_times - fit['a'] + half / 2) % half) - half / 2
    drift_windows = []
    for window_start in np.arange(section['start'], section['end'] - 5, 15.0):
        in_window = (lattice_times >= window_start) & (lattice_times < window_start + 15)
        if in_window.sum() >= 8:
            drift_windows.append((float(window_start), int(in_window.sum()), float(1000 * np.median(residuals[in_window]))))
    return dict(start_s=round(section['start'], 2), end_s=round(section['end'], 2), bpm=round(fit['bpm'], 4), anchor_s=float(fit['a']),
               period_s=beat_period, snapped=snapped, rms_ms=round(fit['rms_ms'], 2), n_inl=fit['n_inl'], template=template_name,
               broke_at=section.get('broke_at'), drift=[(drift_start, drift_count, round(drift_median_ms, 2)) for drift_start, drift_count, drift_median_ms in drift_windows])


SNARE_OVERRIDE_MIN_MS = 25.0
OVERRIDE_MAX_RMS_MS = 3.0
OVERRIDE_KICK_FRACTION = 0.8


def snare_override(events, section, tolerance_milliseconds=6.0, min_fraction=0.6, min_count=40):
    """The grid follows snares/claps. When most snares sit on one phase of the beat lattice that lies further than the refine window
    from the fitted anchor (kick-driven fit locked onto a late kick tail), move the anchor onto the snare phase.
    A shift near half a beat is rejected (that would be off-beat snares)."""
    event_times, event_classes = events['t'], events['c']
    period_seconds = section['period_s']
    is_snare_in_section = (event_classes == 1) & (event_times >= section['start_s']) & (event_times <= section['end_s'])
    if is_snare_in_section.sum() < min_count:
        return section
    lattice_kick_count, lattice_kick_fraction = lattice_kicks(events, section)
    if (section['rms_ms'] <= OVERRIDE_MAX_RMS_MS and lattice_kick_count >= 20 and lattice_kick_fraction >= OVERRIDE_KICK_FRACTION) or \
            (section['rms_ms'] <= OVERRIDE_MAX_RMS2_MS and lattice_kick_count >= PHASE_MAX_LATTICE_KICKS):
        return section
    residual_ms = ((event_times[is_snare_in_section] - section['anchor_s'] + period_seconds / 2) % period_seconds - period_seconds / 2) * 1000.0
    best_offset_ms, best_count = 0.0, 0
    for offset_ms in np.arange(-0.5 * period_seconds * 1000, 0.5 * period_seconds * 1000, 1.0):
        count = int((np.abs(residual_ms - offset_ms) <= tolerance_milliseconds).sum())
        if count > best_count:
            best_offset_ms, best_count = offset_ms, count
    if best_count < min_fraction * is_snare_in_section.sum() or abs(best_offset_ms) < SNARE_OVERRIDE_MIN_MS or abs(best_offset_ms) > 0.35 * period_seconds * 1000:
        return section
    near = residual_ms[np.abs(residual_ms - best_offset_ms) <= tolerance_milliseconds]
    shift = float(np.median(near)) / 1000.0
    section = dict(section, anchor_s=section['anchor_s'] + shift, snare_override_ms=round(shift * 1000, 1))
    return section


PHASE_MIN_SHIFT_MS = 25.0
PHASE_AGREE_MS = 15.0


PHASE_MAX_LATTICE_KICKS = 50
OVERRIDE_MAX_RMS2_MS = 3.5


def lattice_kicks(events, section, tolerance_milliseconds=4.0):
    """(count, fraction) of kick events within tol_ms of the section lattice."""
    event_times, event_classes, period_seconds = events['t'], events['c'], section['period_s']
    is_kick_in_section = (event_classes == 0) & (event_times >= section['start_s']) & (event_times <= section['end_s'])
    if not is_kick_in_section.any():
        return 0, 0.0
    is_on_lattice = np.abs(((event_times[is_kick_in_section] - section['anchor_s'] + period_seconds / 2) % period_seconds - period_seconds / 2) * 1000.0) <= tolerance_milliseconds
    return int(is_on_lattice.sum()), float(is_on_lattice.mean())


def add_novelty(events, path):
    """Attach 1 ms positive-flux novelty of the 40-300 Hz and 300-3000 Hz bands (float16) for the phase vote. True when computed."""
    if 'nov_low' in events:
        return False
    from scipy.signal import butter, sosfilt
    samples, sample_rate = tempo_detection.load_audio(path)
    frame_size = max(1, int(round(sample_rate / 1000.0)))
    for band_name, (low_edge_hz, high_edge_hz) in (('low', (40, 300)), ('mid', (300, 3000))):
        rectified = np.abs(sosfilt(butter(4, [low_edge_hz, high_edge_hz], 'band', fs=sample_rate, output='sos'), samples))
        frame_count = len(rectified) // frame_size
        smoothed = np.convolve(rectified[:frame_count * frame_size].reshape(frame_count, frame_size).mean(1), [0.25] * 4, 'same')
        events['nov_' + band_name] = np.maximum(0, smoothed - np.concatenate([np.zeros(6), smoothed[:-6]])).astype(np.float16)
    events['nov_dt'] = frame_size / float(sample_rate)
    return True


def phase_peaks(events, section):
    """Peak offset (ms, relative to the section lattice) of the novelty folded over one beat, per band."""
    period_seconds, anchor_seconds, step_seconds = section['period_s'], section['anchor_s'], events['nov_dt']
    peaks = {}
    bins = np.arange(-period_seconds / 2, period_seconds / 2 + 1e-9, 0.002)
    for band_name in ('low', 'mid'):
        novelty = events['nov_' + band_name].astype(np.float64)
        times = np.arange(len(novelty)) * step_seconds
        in_section = (times >= section['start_s']) & (times <= section['end_s'])
        phase_deviation = ((times[in_section] - anchor_seconds + period_seconds / 2) % period_seconds) - period_seconds / 2
        histogram, _ = np.histogram(phase_deviation, bins=bins, weights=novelty[in_section])
        histogram = np.convolve(histogram, np.ones(5) / 5, 'same')
        peaks[band_name] = float(1000 * (bins[int(np.argmax(histogram))] + 0.001))
    return peaks


def phase_fix(events, section):
    """When the folded kick-band and snare-band novelty peaks both lie more than 25 ms from the lattice and agree with each other,
    the fit locked onto hats or bass re-entries: move the anchor to the snare-band peak (refine_anchor does the fine tuning)."""
    if 'nov_low' not in events:
        return section
    peaks = phase_peaks(events, section)
    lattice_kick_count, _ = lattice_kicks(events, section)
    period_ms = section['period_s'] * 1000.0
    low_peak_ms, mid_peak_ms = peaks['low'], peaks['mid']
    section = dict(section, phase_peaks={band: round(offset_ms, 1) for band, offset_ms in peaks.items()}, lattice_kicks=lattice_kick_count)
    if lattice_kick_count < PHASE_MAX_LATTICE_KICKS and (abs(low_peak_ms) > PHASE_MIN_SHIFT_MS and abs(mid_peak_ms) > PHASE_MIN_SHIFT_MS and abs(low_peak_ms - mid_peak_ms) <= PHASE_AGREE_MS
            and abs(mid_peak_ms) < 0.4 * period_ms and np.sign(low_peak_ms) == np.sign(mid_peak_ms)):
        section['anchor_s'] += mid_peak_ms / 1000.0
        section['phase_shift_ms'] = round(mid_peak_ms, 1)
    return section



PHASE_MODEL = None
PHASE_MARGIN = 0.1


def load_phase_model(path=None):
    global PHASE_MODEL
    import os
    from lib.beatgrid import rankers as ranker
    path = path or str(paths.MODELS_BEATGRID / 'grid_phase_model.json')
    PHASE_MODEL = ranker.GB.load(path)
    return PHASE_MODEL


def select_phase(events, section, model=None, margin=None):
    """Learned arbiter between the current anchor and competing peaks of the folded low/mid novelty (replaces phase_fix and
    snare_override). Moves the anchor only when the best candidate beats the current one by `margin` in model probability;
    refine_anchor does the fine alignment afterwards."""
    from lib.beatgrid import rankers as ranker
    model = model or PHASE_MODEL or load_phase_model()
    margin = PHASE_MARGIN if margin is None else margin
    positions_ms, candidate_features = ranker.phase_candidates(events, section)
    if positions_ms is None:
        return section
    probabilities = model.proba(candidate_features)
    ours = np.nonzero(candidate_features[:, 0] > 0)[0]
    ours_probability = float(probabilities[ours].max()) if len(ours) else -1.0
    best_index = int(np.argmax(probabilities))
    info = dict(p_best=round(float(probabilities[best_index]), 3), p_ours=round(ours_probability, 3), n_cand=int(len(probabilities)))
    if candidate_features[best_index, 0] == 0 and probabilities[best_index] - ours_probability > margin:
        section = dict(section, anchor_s=section['anchor_s'] + positions_ms[best_index] / 1000.0)
        info['moved_ms'] = round(float(positions_ms[best_index]), 1)
    return dict(section, phase_select=info)


REKORDBOX_RATIO_MAX_RMS = 6.5
REKORDBOX_RATIOS = (4 / 3, 3 / 4, 3 / 2, 2 / 3, 2.0, 0.5, 5 / 4, 4 / 5)


def fix_rekordbox_ratio(events, raw_sections, fits, rekordbox_bpm, tolerance=0.01):
    """Rekordbox's tempo is reliable: a section whose bpm is a simple ratio of it (e.g. 180 for 135) is refit at the RB tempo."""
    if not rekordbox_bpm or rekordbox_bpm < 60:
        return fits
    fixed_fits = []
    for fit in fits:
        if fit is not None and abs(fit['bpm'] - rekordbox_bpm) > 0.6 and any(abs(fit['bpm'] / rekordbox_bpm - ratio) < tolerance * ratio for ratio in REKORDBOX_RATIOS):
            section = max(raw_sections, key=lambda candidate_section: min(candidate_section['end'], fit['end_s']) - max(candidate_section['start'], fit['start_s']))
            if min(section['end'], fit['end_s']) > max(section['start'], fit['start_s']):
                refit = fit_section(events, dict(section, bpms=[float(round(rekordbox_bpm))], start=fit['start_s'], end=fit['end_s']))
                if refit is not None and refit["rms_ms"] <= REKORDBOX_RATIO_MAX_RMS:
                    refit['rb_ratio_fix'] = round(fit['bpm'], 3)
                    fit = refit
        fixed_fits.append(fit)
    return fixed_fits


WHOLE_RATIOS = (4 / 3, 3 / 4, 3 / 2, 2 / 3, 5 / 4, 4 / 5)
WHOLE_BPM_RANGE = (125.0, 185.0)   # genre tempo range; only a section outside it is converted
WHOLE_TOLERANCE = 0.03


def fix_whole_ratio(events, raw_sections, fits):
    """A section with a non-whole bpm outside the genre range whose ratio (4/3, 3/2, 5/4, ...) is a whole bpm inside it was most likely
    detected at a wrong multiple (107.25 -> 143): refit at the whole bpm. Sections in range are left alone (true non-whole tempos exist)."""
    fixed_fits = []
    for fit in fits:
        if fit is not None and abs(fit['bpm'] - round(fit['bpm'])) >= 0.1 and not (WHOLE_BPM_RANGE[0] <= fit['bpm'] <= WHOLE_BPM_RANGE[1]):
            candidates = [fit['bpm'] * ratio for ratio in WHOLE_RATIOS]
            candidates = [candidate for candidate in candidates if abs(candidate - round(candidate)) < WHOLE_TOLERANCE and WHOLE_BPM_RANGE[0] <= candidate <= WHOLE_BPM_RANGE[1]]
            if len(candidates) == 1:
                section = max(raw_sections, key=lambda candidate_section: min(candidate_section['end'], fit['end_s']) - max(candidate_section['start'], fit['start_s']))
                if min(section['end'], fit['end_s']) > max(section['start'], fit['start_s']):
                    refit = fit_section(events, dict(section, bpms=[float(round(candidates[0]))], start=fit['start_s'], end=fit['end_s']))
                    if refit is not None and refit['rms_ms'] <= REKORDBOX_RATIO_MAX_RMS:
                        refit['whole_ratio_fix'] = round(fit['bpm'], 3)
                        fit = refit
        fixed_fits.append(fit)
    return fixed_fits


GHOST_MAX_DURATION_SECONDS, GHOST_MIN_RMS_MS, GHOST_RMS_FACTOR, GHOST_FRACTION_TOLERANCE = 120.0, 3.0, 1.15, 0.1


def _rms(fit):
    return float(fit.get('rms_ms', fit.get('fit_rms_ms', 0.0)))


def drop_ghost_tail(events, fits):
    """Drop a last tempo section that looks like an outro with no real rhythm (reese phasing etc.): shorter than GHOST_MAX_DUR_S, non-whole bpm,
    loosely fitted (rms >= 3 ms and clearly worse than the longest section). The previous section's grid then continues to the end.
    Evidence (1850 tracks): only 3 last sections are non-whole; 2 of them are such ghosts (Falling 122.47, Public Itching 122.5), the third
    (Supersonic 150.58, rms 2.1) is genuine and kept by the rms rule."""
    live = [fit for fit in fits if fit]
    if len(live) < 2:
        return fits
    last = live[-1]
    main = max(live[:-1], key=lambda fit: fit['end_s'] - fit['start_s'])
    duration_seconds = last['end_s'] - last['start_s']
    if (duration_seconds <= GHOST_MAX_DURATION_SECONDS and abs(last['bpm'] - round(last['bpm'])) >= GHOST_FRACTION_TOLERANCE
            and _rms(last) >= GHOST_MIN_RMS_MS and _rms(last) >= GHOST_RMS_FACTOR * _rms(main)):
        previous_fit = live[-2]
        previous_fit['end_s'] = max(previous_fit['end_s'], float(events['dur']))
        previous_fit['ghost_tail_dropped'] = dict(bpm=round(last['bpm'], 3), start_s=round(last['start_s'], 1), rms_ms=round(_rms(last), 2))
        return [fit for fit in fits if fit is not last]
    return fits


def parity_scores(events, section, tolerance=0.012):
    """Weighted kick+snare evidence at the beat lattice (a + kT) and at the off-beat lattice (a + T/2 + kT)."""
    event_times, event_weights, event_classes = events['t'], events['w'], events['c']
    period_seconds, anchor_seconds = section['period_s'], section['anchor_s']
    in_section = (event_times >= section['start_s']) & (event_times < section['end_s'])
    beat_phase = ((event_times[in_section] - anchor_seconds) / period_seconds) % 1.0
    on_beat_mask = (beat_phase < tolerance / period_seconds) | (beat_phase > 1 - tolerance / period_seconds)
    offbeat_mask = np.abs(beat_phase - 0.5) < tolerance / period_seconds
    section_weights, section_classes = event_weights[in_section], event_classes[in_section]
    scores = {}
    for name, selection in (('on', on_beat_mask), ('off', offbeat_mask)):
        scores[name + '_kick'] = float(section_weights[selection & (section_classes == 0)].sum())
        scores[name + '_snare'] = float(section_weights[selection & (section_classes == 1)].sum())
    return scores


BASS_PARITY_RATIO = 2.0
DOWNBEAT_MIN_MARGIN = 0.1


def bass_parity(events, section, anchor_seconds):
    """Bass-band onset energy at the off-beat lattice divided by that at the beat lattice a + kT (window +-0.06 beat).
    Used as an override for weak kick/snare parity decisions: a ratio >= BASS_PARITY_RATIO means the chosen lattice sits on the
    off-beats (session 11: 178 correct tracks all had <= 1.47, the half-beat failures Silva Bumpa and Kanine had 2.1 and 2.4)."""
    if 'bass' not in events:
        return None
    envelope = np.asarray(events['bass'], float)
    rate = float(events['bass_rate'])
    envelope = np.log1p(envelope / (np.median(envelope) + 1e-9))
    onset_flux = np.maximum(np.diff(envelope, prepend=envelope[0]), 0)
    onset_flux = np.convolve(onset_flux, np.ones(3) / 3, 'same')
    times = np.arange(len(onset_flux)) / rate
    in_section = (times >= section['start_s']) & (times < section['end_s'])
    period_seconds = section['period_s']

    def score(anchor):
        beat_phase = ((times[in_section] - anchor) / period_seconds) % 1.0
        return onset_flux[in_section][(beat_phase < 0.06) | (beat_phase > 0.94)].sum()
    return float((score(anchor_seconds + period_seconds / 2) + 1e-9) / (score(anchor_seconds) + 1e-9))


def fix_parity(events, section, first=True):
    """The eighth-note lattice only fixes the anchor mod T/2. Decide beat vs off-beat lattice.
    First section: the library's tracks start on beat 1, so when one lattice has a beat clearly closest to t=0
    (within NEAR_ZERO_FRAC of a beat period) that lattice wins. Otherwise kick+snare weights decide, and near-zero
    breaks ties (weights are unreliable: off-beat hats are detected as snares)."""
    evidence = parity_scores(events, section)
    on_beat_score = evidence['on_kick'] + evidence['on_snare']
    offbeat_score = evidence['off_kick'] + evidence['off_snare']
    ratio = (on_beat_score + 1e-9) / (offbeat_score + 1e-9)
    period_seconds, anchor_seconds = section['period_s'], section['anchor_s']

    def distance_to_beat(position):
        return abs((position + period_seconds / 2) % period_seconds - period_seconds / 2)

    distance_on, distance_off = distance_to_beat(anchor_seconds), distance_to_beat(anchor_seconds + period_seconds / 2)
    near_flip = distance_off < distance_on
    if ratio >= STRONG_RATIO or ratio <= 1 / STRONG_RATIO:
        flip, method = ratio < 1, 'weights (strong)'
    elif first and distance_off <= EXACT_ZERO_FRACTION * period_seconds and distance_on >= 0.25 * period_seconds and 1 / EXACT_ZERO_RATIO <= ratio < EXACT_ZERO_RATIO:
        flip, method = True, 'exact-zero'
    elif 1.0 <= ratio < KEEP_RATIO:
        flip, method = False, 'weights (weak, kept)'
    elif first and min(distance_on, distance_off) <= NEAR_ZERO_FRACTION * period_seconds:
        flip, method = near_flip, 'near-zero'
    elif ratio >= 1.5:
        flip, method = False, 'weights'
    elif ratio <= 1 / 1.5:
        flip, method = True, 'weights'
    else:
        flip, method = near_flip, 'near-zero tie-break'
    bass_ratio = None
    if method != 'weights (strong)':
        bass_ratio = bass_parity(events, section, anchor_seconds + (period_seconds / 2 if flip else 0.0))
        if bass_ratio is not None and bass_ratio >= BASS_PARITY_RATIO:
            flip, method = not flip, method + ' + bass override'
    section['parity'] = dict(evidence, ratio=round(ratio, 2), flipped=bool(flip), method=method,
                         bass_off_on=None if bass_ratio is None else round(bass_ratio, 2))
    if flip:
        section['anchor_s'] = anchor_seconds + period_seconds / 2
    return section


REFINE_SIGMA_MS, REFINE_LIMIT_MS, REFINE_MIN_SNARES = 1.5, 25.0, 40
REFINE_MAX_SHIFT_MS = 8.0


def residual_mode(residuals_ms, weights, limit_ms=REFINE_LIMIT_MS, sigma_ms=REFINE_SIGMA_MS):
    """Weighted kernel-density mode of residuals (ms) within +-lim; (nan, n) when fewer than 8."""
    within_limit = np.abs(residuals_ms) < limit_ms
    near_residuals, weights = residuals_ms[within_limit], weights[within_limit]
    if len(near_residuals) < 8:
        return float('nan'), int(len(near_residuals))
    candidate_offsets = np.arange(-limit_ms, limit_ms, 0.25)
    density = (weights[None, :] * np.exp(-0.5 * ((candidate_offsets[:, None] - near_residuals[None, :]) / sigma_ms) ** 2)).sum(1)
    return float(candidate_offsets[np.argmax(density)]), int(len(near_residuals))


def refine_anchor(events, section):
    """Shift the anchor onto the most supported residual peak. The user's grids follow the snare/clap onsets while kick onsets sit
    at a per-track offset, so the snare residual density (vs the current eighth lattice) gives the candidates and kick weight
    at the same position confirms them. Sections with too few snares keep the lattice fit."""
    event_times, event_weights, event_classes = events['t'], events['w'], events['c']
    low_threshold_times = events.get('tl', event_times)
    in_section = (event_times >= section['start_s']) & (event_times <= section['end_s'])
    event_times, event_weights, event_classes, low_threshold_times = event_times[in_section], event_weights[in_section], event_classes[in_section], low_threshold_times[in_section]
    half = section['period_s'] / 2
    residuals_ms = (low_threshold_times - (section['anchor_s'] + np.round((low_threshold_times - section['anchor_s']) / half) * half)) * 1000.0
    is_near = np.abs(residuals_ms) < REFINE_LIMIT_MS
    is_snare_near = is_near & (event_classes == 1)
    if is_snare_near.sum() < REFINE_MIN_SNARES:
        section['refine'] = dict(shift_ms=0.0, method='kept lattice fit', n_snares=int(is_snare_near.sum()))
        return section
    candidate_offsets = np.arange(-REFINE_LIMIT_MS, REFINE_LIMIT_MS, 0.25)
    density = (event_weights[is_snare_near][None, :] * np.exp(-0.5 * ((candidate_offsets[:, None] - residuals_ms[is_snare_near][None, :]) / REFINE_SIGMA_MS) ** 2)).sum(1)
    peak_indices = [index for index in range(1, len(candidate_offsets) - 1) if density[index] >= density[index - 1] and density[index] > density[index + 1] and density[index] >= 0.4 * density.max()]
    best_shift_ms, best_score = None, -1.0
    for index in peak_indices or [int(np.argmax(density))]:
        in_window = is_near & (np.abs(residuals_ms - candidate_offsets[index]) <= 2.5)
        score = event_weights[in_window & (event_classes == 1)].sum() + event_weights[in_window & (event_classes == 0)].sum()
        if score > best_score:
            best_shift_ms, best_score = float(np.average(residuals_ms[in_window & (event_classes == 1)], weights=event_weights[in_window & (event_classes == 1)])) if (in_window & (event_classes == 1)).any() else float(candidate_offsets[index]), score
    if abs(best_shift_ms) > REFINE_MAX_SHIFT_MS:
        section['refine'] = dict(shift_ms=0.0, method='kept lattice fit', rejected_shift_ms=round(best_shift_ms, 2), n_snares=int(is_snare_near.sum()))
        return section
    section['anchor_s'] += best_shift_ms / 1000.0
    section['refine'] = dict(shift_ms=round(best_shift_ms, 2), method='snare peak + kick support', n_snares=int(is_snare_near.sum()), n_candidates=len(peak_indices))
    return section


def assess(events, sections, duration_seconds):
    """Reasons to look at a track by hand before its grid is written. Returns (level, reasons), level in ok / note / check."""
    if not sections:
        return 'check', ['no section found']
    reasons, needs_check = [], False
    if sections and sections[0].get('forced_rb'):
        reasons.append('no tempo section found, grid forced at the Rekordbox tempo'); needs_check = True
    first_section = sections[0]
    parity = first_section.get('parity') or {}
    on_beat_score, offbeat_score = parity.get('on_kick', 0) + parity.get('on_snare', 0), parity.get('off_kick', 0) + parity.get('off_snare', 0)
    ratio = (on_beat_score + 1e-9) / (offbeat_score + 1e-9)
    if parity.get('method') == 'near-zero' and ((ratio < 1 / 2.5 and not parity.get('flipped')) or (ratio > 2.5 and parity.get('flipped'))):
        reasons.append('parity: start-position rule overrode strong kick/snare weights')
    if parity.get('method') == 'near-zero tie-break':
        reasons.append('parity: weights inconclusive, decided by start position')
    covered = sum(section['end_s'] - section['start_s'] for section in sections)
    if len(sections) > 1:
        reasons.append(f'{len(sections)} tempo sections')
    if covered < 0.6 * duration_seconds:
        reasons.append(f'grid covers only {100 * covered / duration_seconds:.0f}% of the track')
        needs_check = True
    if first_section['start_s'] > 20:
        reasons.append(f'first section starts at {first_section["start_s"]:.0f} s')
    downbeat_info = first_section.get('downbeat')
    if downbeat_info and downbeat_info['applied']:
        reasons.append(f"beat 1 moved {downbeat_info['q']} beat(s) after the first grid beat (snares on the backbeat)")
    refine_info = first_section.get('refine') or {}
    if 'rejected_shift_ms' in refine_info:
        reasons.append(f'snare refinement wanted {refine_info["rejected_shift_ms"]:+.1f} ms, rejected as too large, anchor from lattice fit')
        needs_check = True
    elif refine_info.get('method') == 'kept lattice fit':
        reasons.append('few snares, anchor from lattice fit only')
    return ('check' if needs_check else ('note' if reasons else 'ok')), reasons


def same_grid(first_grid, second_grid, tolerance_milliseconds=6.0):
    if round(first_grid['bpm']) != round(second_grid['bpm']) or abs(first_grid['bpm'] - second_grid['bpm']) > 0.25:
        return False
    period_seconds = 0.5 * (first_grid['period_s'] + second_grid['period_s'])
    distance_seconds = (first_grid['anchor_s'] - second_grid['anchor_s']) % (period_seconds / 2)
    distance_seconds = min(distance_seconds, period_seconds / 2 - distance_seconds)
    return 1000 * distance_seconds < tolerance_milliseconds


def drop_overlapping(kept_sections, min_overlap_fraction=0.4, bpm_tolerance=0.5):
    """Remove a section that overlaps a longer section of (nearly) the same tempo by >= frac of its own length."""
    remaining = []
    for index, (raw_section, fit) in enumerate(kept_sections):
        duration_seconds = fit['end_s'] - fit['start_s']
        is_overlapped = False
        for other_index, (other_section, other_fit) in enumerate(kept_sections):
            if other_index == index or abs(fit['bpm'] - other_fit['bpm']) > bpm_tolerance:
                continue
            overlap_seconds = min(fit['end_s'], other_fit['end_s']) - max(fit['start_s'], other_fit['start_s'])
            if overlap_seconds >= min_overlap_fraction * duration_seconds and (other_fit['end_s'] - other_fit['start_s']) > duration_seconds:
                is_overlapped = True
        if not is_overlapped:
            remaining.append((raw_section, fit))
    return remaining


NEAR_ZERO_FRACTION = 0.10
STRONG_RATIO = 12.0
KEEP_RATIO = 1.5
EXACT_ZERO_FRACTION = 0.03   # one lattice puts a beat within 3% of a period of t=0 while the other sits ~half a beat away
EXACT_ZERO_RATIO = 3.0   # only overrides weak/moderate kick+snare evidence
HARMONIC_RATIOS = (4 / 3, 5 / 4, 3 / 2, 5 / 3, 6 / 5, 7 / 4, 7 / 5)


def drop_harmonic(kept_sections, max_length_seconds=75.0, tolerance=0.004):
    """Remove short sections whose tempo is a simple polyrhythmic ratio of a longer section (4/3, 5/4, ...)."""
    remaining = []
    for index, (raw_section, fit) in enumerate(kept_sections):
        duration_seconds = fit['end_s'] - fit['start_s']
        is_harmonic = False
        for other_index, (other_section, other_fit) in enumerate(kept_sections):
            if other_index == index or (other_fit['end_s'] - other_fit['start_s']) <= duration_seconds or duration_seconds > max_length_seconds:
                continue
            tempo_ratio = fit['bpm'] / other_fit['bpm']
            if any(abs(tempo_ratio / harmonic_ratio - 1) < tolerance or abs(tempo_ratio * harmonic_ratio - 1) < tolerance for harmonic_ratio in HARMONIC_RATIOS):
                is_harmonic = True
        if not is_harmonic:
            remaining.append((raw_section, fit))
    return remaining


def fix_ratio(events, kept_sections, max_rms_milliseconds=5.0):
    """A later section locked on 3/2 (or 2/3) of its real tempo: refit at the ratio that lies closer to the previous section's tempo."""
    fixed = []
    for index, (raw_section, fit) in enumerate(kept_sections):
        if index > 0:
            previous_bpm = fixed[-1][1]['bpm']
            best = None
            for ratio in (2 / 3, 3 / 2):
                candidate_bpm = fit['bpm'] * ratio
                if abs(np.log(candidate_bpm / previous_bpm)) < abs(np.log(fit['bpm'] / previous_bpm)) - 0.02 and 60 <= candidate_bpm <= 200:
                    refit = fit_section(events, dict(raw_section, bpms=[candidate_bpm], start=fit['start_s'], end=fit['end_s']))
                    if refit and refit['rms_ms'] <= max_rms_milliseconds and refit['n_inl'] >= max(15, 0.7 * fit['n_inl']) and (best is None or refit['rms_ms'] < best['rms_ms']):
                        best = refit
            if best:
                fit = best
        fixed.append((raw_section, fit))
    return fixed


def clean_sections(events, raw_sections, fits, max_rms_milliseconds=5.0, min_inlier_count=15):
    """Drop badly fitted sections and merge neighbours that continue the same grid."""
    kept_sections = [(raw_section, fit) for raw_section, fit in zip(raw_sections, fits) if fit and fit['rms_ms'] <= max_rms_milliseconds and fit['n_inl'] >= min_inlier_count]
    kept_sections = drop_overlapping(kept_sections)
    kept_sections = drop_harmonic(kept_sections)
    kept_sections = fix_ratio(events, kept_sections)
    if not kept_sections:
        kept_sections = [(raw_section, fit) for raw_section, fit in zip(raw_sections, fits) if fit]
    merged = []
    for raw_section, fit in kept_sections:
        if merged and same_grid(merged[-1][1], fit):
            previous_section, previous_fit = merged[-1]
            union_section = dict(start=min(previous_section['start'], raw_section['start']), end=max(previous_section['end'], raw_section['end']), bpms=previous_section['bpms'] + raw_section['bpms'],
                     blocks=sorted(previous_section['blocks'] + raw_section['blocks'], key=lambda block: block['start']), broke_at=raw_section.get('broke_at'))
            union_fit = fit_section(events, union_section)
            merged[-1] = (union_section, union_fit if union_fit else previous_fit)
        else:
            merged.append((raw_section, fit))
    return [fit for _, fit in merged]


def probe_header(path):
    """Container facts that decide the offset between decoded time and Rekordbox time."""
    import subprocess
    try:
        probe_data = tempo_detection.probe_json(path)
        audio_stream = next(stream for stream in probe_data['streams'] if stream.get('codec_type') == 'audio')
    except Exception as error:
        return {'error': repr(error)}
    tags = {tag_name.lower(): tag_value for tag_name, tag_value in (probe_data.get('format', {}).get('tags') or {}).items()}
    tags.update({tag_name.lower(): tag_value for tag_name, tag_value in (audio_stream.get('tags') or {}).items()})
    return dict(codec=audio_stream.get('codec_name'), start_time=float(audio_stream.get('start_time') or 0.0), sr=int(audio_stream.get('sample_rate') or 0),
                encoder=tags.get('encoder'), itunsmpb=tags.get('itunsmpb'))


MP3_DELAY_MS = 1105 / 44100 * 1000.0


def iter_tags(data):
    import struct
    offset = struct.unpack_from(">I", data, 4)[0]
    while offset + 12 <= len(data):
        tag_type = bytes(data[offset:offset + 4]).decode("ascii", "replace")
        header_length, tag_length = struct.unpack_from(">II", data, offset + 4)
        if tag_length < 12:
            break
        yield tag_type, offset, header_length, tag_length
        offset += tag_length


def read_detail_waveform(data):
    """(kind, heights 0..31) from the PWV3 / PWV5 detail waveform (150 entries per second) of an ANLZ .EXT file."""
    import struct
    for wanted_tag in ("PWV3", "PWV5"):
        for tag_type, offset, header_length, tag_length in iter_tags(data):
            if tag_type != wanted_tag:
                continue
            entry_size, entry_count = struct.unpack_from(">II", data, offset + 12)
            raw_bytes = np.frombuffer(data, dtype=np.uint8, count=entry_count * entry_size, offset=offset + header_length)
            if wanted_tag == "PWV3":
                return wanted_tag, (raw_bytes & 0x1F).astype(float)
            byte_pairs = raw_bytes.reshape(-1, 2)
            word = (byte_pairs[:, 0].astype(np.uint16) << 8) | byte_pairs[:, 1]
            return wanted_tag, ((word >> 2) & 0x1F).astype(float)
    return None


def measure_waveform_offset(heights, samples, sample_rate, min_shift_ms=-100.0, max_shift_ms=150.0, step_ms=0.25):
    """Shift (ms) = Rekordbox waveform time minus decoded time, by correlating Rekordbox's detail waveform with the
    peak envelope of the decoded audio. Returns (shift_ms, corr, runner_up_corr)."""
    from scipy.ndimage import maximum_filter1d
    window_samples = int(round(sample_rate / 150.0))
    peak_envelope = maximum_filter1d(np.abs(samples).astype(np.float32), size=window_samples, mode="constant")
    root_envelope = np.sqrt(peak_envelope)
    height_count = len(heights)
    sample_positions = np.arange(height_count) * sample_rate / 150.0
    normalized_heights = (heights - heights.mean()) / (heights.std() + 1e-9)
    shifts = np.arange(min_shift_ms, max_shift_ms + step_ms, step_ms)
    correlations = np.full(len(shifts), -1.0)
    envelope_length = len(peak_envelope)
    stride = sample_rate // 150 if sample_rate % 150 == 0 else 0
    for shift_index, shift_ms in enumerate(shifts):
        if stride:
            # sr is a multiple of 150: the sampled positions are an arithmetic progression, so a strided view replaces the gather
            # (a rounding tie may move a position by one sample)
            start_offset = int(np.round(window_samples // 2 - shift_ms * sample_rate / 1000.0))
            first_height_index = max(0, -(start_offset // stride))
            last_height_index = min(height_count - 1, (envelope_length - 1 - start_offset) // stride)
            if last_height_index - first_height_index + 1 < 200:
                continue
            envelope_values = root_envelope[start_offset + stride * first_height_index:start_offset + stride * last_height_index + 1:stride]
            height_values = normalized_heights[first_height_index:last_height_index + 1]
        else:
            positions = np.round(sample_positions - shift_ms * sample_rate / 1000.0 + window_samples // 2).astype(np.intp)
            is_valid = (positions >= 0) & (positions < envelope_length)
            if is_valid.sum() < 200:
                continue
            envelope_values, height_values = root_envelope[positions[is_valid]], normalized_heights[is_valid]
        envelope_values = envelope_values - envelope_values.mean()
        correlations[shift_index] = float(np.mean(envelope_values * height_values)) / (envelope_values.std() + 1e-9)
    best_index = int(np.argmax(correlations))
    best_shift_ms = shifts[best_index]
    if 0 < best_index < len(correlations) - 1:
        previous_correlation, peak_correlation, next_correlation = correlations[best_index - 1], correlations[best_index], correlations[best_index + 1]
        curvature = previous_correlation - 2 * peak_correlation + next_correlation
        if curvature < 0:
            best_shift_ms = shifts[best_index] + 0.5 * step_ms * (previous_correlation - next_correlation) / curvature
    is_far = np.abs(shifts - shifts[best_index]) > 15
    runner_up_correlation = float(correlations[is_far].max()) if is_far.any() else float("nan")
    return float(best_shift_ms), float(correlations[best_index]), runner_up_correlation


def waveform_offset_ms(ext_path, audio_path):
    """Measured Rekordbox-waveform offset of one track, or None if the .EXT has no detail waveform."""
    waveform = read_detail_waveform(open(ext_path, "rb").read())
    if waveform is None:
        return None
    samples, sample_rate = tempo_detection.load_audio(audio_path)
    offset_ms, correlation, runner_up = measure_waveform_offset(waveform[1], samples, sample_rate)
    return dict(ms=offset_ms, corr=correlation, runner_up=runner_up, kind=waveform[0])


def container_offset_ms(header, waveform_offset=None):
    """Rekordbox grid time (PQTZ) minus decoded (ffmpeg) time -> (offset_ms, status).
    Measured: PQTZ offset = waveform offset (snapped to a multiple of the 1105-sample mp3 delay, 25.057 ms)
    plus the AAC priming (ffprobe start_time, 47.891 ms) for m4a. The waveform offset is 0 for every m4a and
    for mp3 without Xing/Info tag, 25.06 ms for most mp3 with the tag, and 50.1 ms for some (not predictable
    from headers). Without a reliable waveform measurement: m4a priming, wav 0, mp3 unresolved."""
    codec = (header or {}).get('codec')
    priming_ms = 1000 * header.get('start_time', 0.0) if codec == 'aac' else 0.0
    if codec == 'aac':
        return priming_ms, 'm4a priming'
    is_reliable = bool(waveform_offset) and waveform_offset['corr'] >= 0.7 and waveform_offset['corr'] - waveform_offset['runner_up'] >= 0.03
    if is_reliable:
        mp3_delay_count = round(waveform_offset['ms'] / MP3_DELAY_MS)
        if abs(waveform_offset['ms'] - mp3_delay_count * MP3_DELAY_MS) <= 6.0:
            return priming_ms + mp3_delay_count * MP3_DELAY_MS, ('m4a priming + ' if priming_ms else '') + f'waveform x{mp3_delay_count}'
        return priming_ms + waveform_offset['ms'], ('m4a priming + ' if priming_ms else '') + 'waveform (unsnapped)'
    if codec == 'aac':
        return priming_ms, 'm4a priming'
    if codec and codec.startswith('pcm'):
        return 0.0, 'wav'
    return 0.0, 'unresolved'


def section_start_s(events, section, previous_start, max_misses=4, tolerance=0.008):
    """Earliest time the section's lattice is still supported by events, scanning backwards from its fitted start (limit: prev_start)."""
    period_seconds, anchor_seconds = section['period_s'], section['anchor_s']
    event_times = events['t']
    beat_index = int(np.ceil((section['start_s'] - anchor_seconds) / period_seconds))
    best, miss_count = anchor_seconds + beat_index * period_seconds, 0
    while miss_count < max_misses and anchor_seconds + (beat_index - 1) * period_seconds > previous_start:
        beat_index -= 1
        beat_time = anchor_seconds + beat_index * period_seconds
        nearby_events = event_times[(event_times > beat_time - period_seconds / 2 - tolerance) & (event_times < beat_time + period_seconds / 2 + tolerance)]
        if len(nearby_events) == 0:
            continue
        residuals = ((nearby_events - beat_time + period_seconds / 4) % (period_seconds / 2)) - period_seconds / 4
        if np.any(np.abs(residuals) < tolerance):
            best, miss_count = beat_time, 0
        else:
            miss_count += 1
    return best - period_seconds / 2


BASS_RATE = 100.0


def add_bass(events, path):
    """Attach the 100 Hz 90-250 Hz band envelope (float16) used by the downbeat vote. Returns True when it had to be computed."""
    if 'bass' in events:
        return False
    from lib import audio_features
    samples, sample_rate = tempo_detection.load_audio(path)
    envelopes, hop_samples = audio_features.band_envelope_array(samples, sample_rate, int(BASS_RATE), bands=(1,))
    events['bass'] = envelopes[1].astype(np.float16)
    events['bass_rate'] = float(sample_rate) / hop_samples
    return True


def downbeat_scores(events, section, first_beat_time):
    """Score for each phase q in 0..3 = beat index (from the first grid beat at decoded time t0) of beat 1.
    Template match of the on-beat kick/snare weight histograms (backbeat: kick 1, snare 2+4; half-time: kick 1, snare 3) plus the
    vote of 90-250 Hz onsets per beat position. Both parts are normalised to sum 1 per histogram."""
    period_seconds = section['period_s']
    event_times, event_weights, event_classes = events['t'], events['w'], events['c']
    in_section = (event_times >= section['start_s']) & (event_times <= section['end_s'])
    beat_numbers = (event_times[in_section] - first_beat_time) / period_seconds
    event_weights, event_classes = event_weights[in_section], event_classes[in_section]
    is_near_beat = np.abs(beat_numbers - np.round(beat_numbers)) < 0.12
    beat_positions = np.round(beat_numbers).astype(int) % 4
    kick_histogram = np.array([event_weights[is_near_beat & (event_classes == 0) & (beat_positions == candidate_position)].sum() for candidate_position in range(4)])
    snare_histogram = np.array([event_weights[is_near_beat & (event_classes == 1) & (beat_positions == candidate_position)].sum() for candidate_position in range(4)])
    kick_histogram, snare_histogram = kick_histogram / (kick_histogram.sum() + 1e-9), snare_histogram / (snare_histogram.sum() + 1e-9)
    bass_votes = np.zeros(4)
    bass = events.get('bass')
    if bass is not None:
        rate = events.get('bass_rate', BASS_RATE)
        bass_log = np.log(bass.astype(np.float32) + 1e-4)
        flux = np.concatenate([np.zeros(3, np.float32), np.maximum(bass_log[3:] - bass_log[:-3], 0)])
        beat_indices = np.arange(0, int((section['end_s'] - first_beat_time) / period_seconds) + 1)
        for beat_index in beat_indices:
            first_frame, last_frame = int((first_beat_time + beat_index * period_seconds - 0.02) * rate), int((first_beat_time + beat_index * period_seconds + 0.04) * rate) + 1
            if first_frame >= 0 and last_frame < len(flux) and first_beat_time + beat_index * period_seconds >= section['start_s']:
                bass_votes[beat_index % 4] += flux[first_frame:last_frame].max()
        bass_votes /= bass_votes.sum() + 1e-9
    scores = np.zeros(4)
    for candidate_position in range(4):
        kick_shifted = np.array([kick_histogram[(candidate_position + offset) % 4] for offset in range(4)])
        snare_shifted = np.array([snare_histogram[(candidate_position + offset) % 4] for offset in range(4)])
        kick_score = kick_shifted @ [1, .1, .5, .1]
        scores[candidate_position] = max(kick_score + snare_shifted @ [0, 1, 0, 1], kick_score + snare_shifted @ [0, 0, 1.6, 0]) + bass_votes[candidate_position]
    return scores


def pick_downbeat_rule(events, section, first_beat_time):
    """(q, margin, info): phase of beat 1 relative to the first grid beat. Default 0 (about 92% of tracks start on beat 1).
    Only one case is trusted: a clean backbeat (snare weight >= 90% on one parity of the first-beat frame, both positions of that
    parity >= 25%, >= 15 snares) with the snares on the even positions means beat 1 is odd beats away; 1 or 3 is chosen by the full
    score. The full score on its own (kick template + bass votes) flipped 10 of 92 correct tracks on the 100-track run, so it is only
    reported (info.best/margin), never applied otherwise."""
    if events is None or 'bass' not in events:
        return 0, 0.0, None
    scores = downbeat_scores(events, section, first_beat_time)
    best_position = int(np.argmax(scores))
    margin = float(scores[best_position] - scores[0])
    period_seconds = section['period_s']
    event_times, event_weights, event_classes = events['t'], events['w'], events['c']
    is_snare_in_section = (event_times >= section['start_s']) & (event_times <= section['end_s']) & (event_classes == 1)
    beat_numbers = (event_times[is_snare_in_section] - first_beat_time) / period_seconds
    is_near_beat = np.abs(beat_numbers - np.round(beat_numbers)) < 0.12
    beat_positions = np.round(beat_numbers).astype(int) % 4
    snare_histogram = np.array([event_weights[is_snare_in_section][is_near_beat & (beat_positions == position)].sum() for position in range(4)])
    snare_total = snare_histogram.sum()
    backbeat_odd = bool(snare_total >= 15 and (snare_histogram[0] + snare_histogram[2]) / snare_total >= 0.9 and min(snare_histogram[0], snare_histogram[2]) / snare_total >= 0.25)
    picked_position = (1 if scores[1] >= scores[3] else 3) if backbeat_odd else 0
    # v7: the full score must independently agree (argmax == pick, margin >= DOWNBEAT_MIN_MARGIN). On the 200-track holdout the snare-only
    # rule moved 4 correct tracks (All the Sikes, Ballista, Mist Remina Remix, Pl9 Euro Remix) that all had best=0 / margin 0.0.
    if picked_position and not (best_position == picked_position and margin >= DOWNBEAT_MIN_MARGIN):
        picked_position = 0
    info = dict(best=best_position, margin=round(margin, 3), scores=[round(float(value), 3) for value in scores], applied=bool(picked_position), q=picked_position,
                snare_phase_share=[round(float(value / (snare_total + 1e-9)), 2) for value in snare_histogram])
    return picked_position, margin, info


def build_grid(sections, model_ms, duration_seconds, events=None, start_snap_ms=3.0):
    """Final grid in Rekordbox time: list of (beat_in_bar, tempo_x100, time_ms), integer ms.
    The first section's constant grid is extended back to the file start (a first beat up to start_snap_ms before 0 is snapped to 0);
    later sections start where their lattice stops being supported by events (midpoint of the gap without events).
    The first beat is numbered 1 unless pick_downbeat finds beat 1 later (shift); beats cycle 1-4."""
    if not sections:
        return []
    sections = sorted(sections, key=lambda section: section['start_s'])
    cuts = []
    for section_index in range(1, len(sections)):
        previous_section, next_section = sections[section_index - 1], sections[section_index]
        cuts.append(section_start_s(events, next_section, previous_section['start_s']) if events is not None else 0.5 * (previous_section['end_s'] + next_section['start_s']))
    bounds = [None] + cuts + [duration_seconds]
    beats = []
    for section_index, section in enumerate(sections):
        period_seconds, anchor_seconds = section['period_s'], section['anchor_s']
        section_end = bounds[section_index + 1]
        if section_index == 0:
            first_beat_index = int(np.ceil((-model_ms / 1000.0 - anchor_seconds) / period_seconds - start_snap_ms / 1000.0 / period_seconds))
        else:
            first_beat_index = int(np.ceil((bounds[section_index] - anchor_seconds) / period_seconds - 1e-9))
        last_beat_index = int(np.floor((section_end - anchor_seconds) / period_seconds + 1e-9))
        tempo = int(round(6000.0 / period_seconds))
        shift = 0
        if section_index == 0:
            shift, _, info = pick_downbeat(events, section, anchor_seconds + first_beat_index * period_seconds)
            section['downbeat'] = info
        for beat_number, beat_index in enumerate(range(first_beat_index, last_beat_index + 1)):
            beats.append(((beat_number - shift) % 4 + 1, tempo, max(0, int(round(1000 * (anchor_seconds + beat_index * period_seconds) + model_ms)))))
    return beats


def dry_compare(built, stored):
    """Compare the built grid with the stored Rekordbox grid (both lists of (beat_in_bar, tempo_x100, time_ms)).
    Each stored beat is matched to the nearest built beat within half a beat."""
    if not built or not stored:
        return dict(ok=False, reason='empty grid')
    built_times = np.array([built_beat[2] for built_beat in built], float)
    diffs, numbering_shifts = [], []
    unmatched = 0
    for stored_beat_in_bar, tempo, stored_time_ms in stored:
        nearest_index = int(np.argmin(np.abs(built_times - stored_time_ms)))
        half = 3000000.0 / max(built[nearest_index][1], 1)
        if abs(built_times[nearest_index] - stored_time_ms) > half:
            unmatched += 1
            continue
        diffs.append(built_times[nearest_index] - stored_time_ms)
        numbering_shifts.append((built[nearest_index][0] - stored_beat_in_bar) % 4)
    if not diffs:
        return dict(ok=False, reason='no matches')
    differences = np.array(diffs)
    shifts = np.bincount(numbering_shifts, minlength=4)
    most_common_shift = int(np.argmax(shifts))
    return dict(ok=True, n_stored=len(stored), n_built=len(built), unmatched=unmatched,
                diff_median_ms=float(np.median(differences)), diff_p95_abs_ms=float(np.percentile(np.abs(differences), 95)),
                diff_max_abs_ms=float(np.abs(differences).max()), numbering_shift=most_common_shift,
                numbering_match_frac=float(shifts[most_common_shift] / len(numbering_shifts)),
                first_built_ms=built[0][2], first_stored_ms=stored[0][2])


def analyze(path, cache=True):
    events = get_events(path, cache)
    blocks = block_scans(events)
    sections = find_sections(blocks)
    fits = [fit_section(events, section) for section in sections]
    fits = clean_sections(events, sections, fits)
    analysis = dict(file=os.path.basename(path), duration_s=round(events['dur'], 1), n_kicks=events['nk'], n_snares=events['ns'], sections=[])
    for fit in fits:
        analysis['sections'].append(refine_anchor(events, fix_parity(events, fit, first=not analysis['sections'])))
    return analysis


if __name__ == '__main__':
    for audio_path in sys.argv[1:]:
        analysis = analyze(audio_path)
        print(f"{analysis['file'][:40]:40s} dur {analysis['duration_s']:6.1f} k{analysis['n_kicks']}/s{analysis['n_snares']}")
        for section in analysis['sections']:
            drift_values = [drift for _, _, drift in section['drift']]
            print(f"   {section['start_s']:6.1f}-{section['end_s']:6.1f} bpm {section['bpm']:9.4f} rms {section['rms_ms']:5.2f} inl {section['n_inl']:3d} "
                  f"drift[{min(drift_values):+.1f},{max(drift_values):+.1f}]ms break@{section['broke_at']}" if drift_values else
                  f"   {section['start_s']:6.1f}-{section['end_s']:6.1f} bpm {section['bpm']:9.4f} rms {section['rms_ms']:5.2f} inl {section['n_inl']:3d} nodrift break@{section['broke_at']}")


DOWNBEAT_MODEL = None


def load_downbeat_model(path=None):
    global DOWNBEAT_MODEL
    from lib.beatgrid import rankers as ranker
    path = path or str(paths.MODELS_BEATGRID / 'downbeat_model.json')
    DOWNBEAT_MODEL = ranker.LRModel.load(path)
    return DOWNBEAT_MODEL


def pick_downbeat(events, section, first_beat_time, model=None):
    """(q, margin, info): beat 1 phase from the learned model (ranker.downbeat_candidates, one logistic score per q). Beat 1 stays on
    the first grid beat unless another phase beats it by model.margin; falls back to the rule version without events or model."""
    if events is None or 'bass' not in events:
        return 0, 0.0, None
    try:
        from lib.beatgrid import rankers as ranker
        model = model or DOWNBEAT_MODEL or load_downbeat_model()
        candidate_features = ranker.downbeat_candidates(events, section, first_beat_time)
    except Exception:
        return pick_downbeat_rule(events, section, first_beat_time)
    if candidate_features is None:
        return 0, 0.0, None
    probabilities = model.proba(candidate_features)
    best_position = int(np.argmax(probabilities))
    margin = float(probabilities[best_position] - probabilities[0])
    picked_position = best_position if (best_position != 0 and margin > model.margin) else 0
    info = dict(best=best_position, margin=round(margin, 3), scores=[round(float(value), 3) for value in probabilities], applied=bool(picked_position), q=picked_position)
    try:
        from lib.beatgrid import advanced_fine_adjust as advanced
        advanced_result = advanced.downbeat_proba(events, section, first_beat_time, candidate_features)
    except Exception:
        advanced_result = None
    if advanced_result is not None:   # advanced second pass (bar-folded per-beat energies attached by track_job.build_grid_advanced)
        advanced_probabilities, advanced_required_margin = advanced_result
        advanced_best_position = int(np.argmax(advanced_probabilities))
        advanced_margin = float(advanced_probabilities[advanced_best_position] - advanced_probabilities[0])
        advanced_picked_position = advanced_best_position if (advanced_best_position != 0 and advanced_margin > advanced_required_margin) else 0
        info = dict(best=advanced_best_position, margin=round(advanced_margin, 3), scores=[round(float(value), 3) for value in advanced_probabilities], applied=bool(advanced_picked_position), q=advanced_picked_position, adv=True,
                    basic=dict(q=picked_position, scores=info['scores']))
        return advanced_picked_position, advanced_margin, info
    return picked_position, margin, info


FINE_MODEL = None
FINE_DEADZONE_MS, FINE_CAP_MS = 1.0, 15.0


def load_fine_model(path=None):
    global FINE_MODEL
    from lib.beatgrid import rankers as ranker
    path = path or str(paths.MODELS_BEATGRID / 'fine_adjust_model.json')
    FINE_MODEL = ranker.GB.load(path)
    return FINE_MODEL


def fine_adjust(events, section, model=None):
    """Learned residual correction of the final anchor (ms), from the folded novelty and event residual structure.
    Shifts below FINE_DEADZONE_MS are ignored, larger ones are capped at FINE_CAP_MS."""
    try:
        from lib.beatgrid import rankers as ranker
        model = model or FINE_MODEL or load_fine_model()
        feature_vector = ranker.fine_features(events, section)
        shift_ms = float(model.raw(feature_vector[None, :])[0])
    except Exception:
        return section
    section['fine_raw_ms'] = round(shift_ms, 3)
    if abs(shift_ms) < FINE_DEADZONE_MS:
        shift_ms = 0.0
    shift_ms = max(-FINE_CAP_MS, min(FINE_CAP_MS, shift_ms))
    section['fine_shift_ms'] = round(shift_ms, 2)
    section['anchor_s'] += shift_ms / 1000.0
    return section


def forced_rekordbox_section(events, rekordbox_bpm):
    """Fallback when no section was found: fit the whole track at the Rekordbox tempo (None when the fit is poor)."""
    if not rekordbox_bpm or rekordbox_bpm < 60:
        return None
    fit = fit_section(events, dict(bpms=[float(round(rekordbox_bpm))], start=0.0, end=float(events['dur']), blocks=[dict(res=None)]))
    if fit is None or fit['rms_ms'] > REKORDBOX_RATIO_MAX_RMS or fit['n_inl'] < 15:
        return None
    fit['forced_rb'] = True
    return fit


REKORDBOX_NEAR_MAX_BPM = 0.8


def fix_rekordbox_near(events, fits, rekordbox_bpm):
    """A section whose free-fit tempo is within RB_NEAR_MAX_BPM of a whole-number Rekordbox tempo but not on it is a drifting fit of that tempo:
    refit with the tempo fixed (phase scan on the eighth lattice, then fit_fixed), kept when the rms stays acceptable."""
    if not rekordbox_bpm or rekordbox_bpm < 60 or abs(rekordbox_bpm - round(rekordbox_bpm)) > 0.005:
        return fits
    bpm = float(round(rekordbox_bpm))
    event_times, event_weights = events['t'], events['w']
    fixed_fits = []
    for fit in fits:
        if fit is not None and 0.004 < abs(fit['bpm'] - bpm) < REKORDBOX_NEAR_MAX_BPM:
            in_section = (event_times >= fit['start_s']) & (event_times <= fit['end_s'])
            half = 30.0 / bpm
            candidate_phases = np.arange(0.0, half, 0.002)
            section_times, section_weights = event_times[in_section], event_weights[in_section]
            phase_scores = [(section_weights * np.exp(-0.5 * ((((section_times - candidate_phase + half / 2) % half) - half / 2) / 0.006) ** 2)).sum() for candidate_phase in candidate_phases]
            fixed_fit = fit_fixed(section_times, section_weights, float(candidate_phases[int(np.argmax(phase_scores))]), bpm)
            if fixed_fit is not None and fixed_fit[1] <= max(REKORDBOX_RATIO_MAX_RMS, fit['rms_ms'] + 2.0) and fixed_fit[2] >= 0.7 * fit['n_inl']:
                refit = dict(fit, bpm=bpm, anchor_s=fixed_fit[0], period_s=60.0 / bpm, rms_ms=round(fixed_fit[1], 2), n_inl=fixed_fit[2], snapped=True, rb_near_fix=round(fit['bpm'], 4))
                fit = refit
        fixed_fits.append(fit)
    return fixed_fits
