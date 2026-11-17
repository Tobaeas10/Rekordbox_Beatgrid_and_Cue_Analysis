#!/usr/bin/env python3
"""beatgrid.py - find kicks/snares in a song, then BPM + constant beat grid.

STATUS (see HANDOFF.md): step 2 (event detection over whole file) implemented here.
Steps 3-7 (BPM/grid fit, constant-BPM flag, output) are added below as they get done.

Usage: python beatgrid.py song.(mp3|wav|m4a) [--plot] [--json out.json]
"""
import sys
import json
import argparse
import numpy as np
try:
    import librosa
except ImportError:
    librosa = None
from scipy import fft as scipy_fft
from scipy.signal import (butter, sosfiltfilt, fftconvolve, oaconvolve, find_peaks,
                          hilbert, resample_poly)

# ----------------------------------------------------------------------------
# Event detection
# ----------------------------------------------------------------------------
KICK_DECIMATION_FACTOR = 20            # decimate kick band 20x (44.1k -> 2205 Hz); plenty for <300 Hz


def detect_kicks(samples, sample_rate, min_score=0.6, onset_fraction=0.25):
    """Pitch-drop matched filter + instantaneous-frequency verification
    (logic from kick_sweep.py, vectorised over the whole file at low rate).
    Returns list of (time_s, score)."""
    kick_band = sosfiltfilt(butter(4, [25, 300], btype="band", fs=sample_rate, output="sos"), samples)
    decimation_factor = max(1, int(round(sample_rate / 2205)))
    kick_band = resample_poly(kick_band, 1, decimation_factor)
    decimated_rate = sample_rate / decimation_factor
    template_length = int(0.12 * decimated_rate)
    template_times = np.arange(template_length) / decimated_rate
    sample_count = len(kick_band)
    score = np.zeros(sample_count)
    fft_length = scipy_fft.next_fast_len(sample_count + template_length - 1)
    decay_envelope = np.exp(-template_times / 0.12)
    kick_band_float32 = kick_band.astype(np.float32)
    decay_energy = scipy_fft.irfft(scipy_fft.rfft(kick_band ** 2, fft_length) * scipy_fft.rfft((decay_envelope ** 2)[::-1], fft_length), fft_length)[template_length - 1:template_length - 1 + sample_count]
    normalizer = np.sqrt((decay_envelope ** 2).sum() * decay_energy + 1e-9)
    for start_frequency in (140, 170, 200, 230):
        for end_frequency in (45, 52):
            for time_constant in (0.015, 0.025, 0.04):
                frequency_curve = end_frequency + (start_frequency - end_frequency) * np.exp(-template_times / time_constant)
                chirp_template = decay_envelope * np.exp(2j * np.pi * np.cumsum(frequency_curve) / decimated_rate)
                correlation = np.abs(oaconvolve(kick_band_float32, np.conj(chirp_template)[::-1].astype(np.complex64), mode="full")[template_length - 1:template_length - 1 + sample_count])
                score = np.maximum(score, correlation / normalizer)
    peak_indices, _ = find_peaks(score, height=min_score, distance=int(0.12 * decimated_rate))

    # verification via instantaneous frequency (chunk-free: analytic signal of the
    # decimated signal is cheap)
    analytic_signal = hilbert(kick_band)
    amplitude = np.abs(analytic_signal)
    instantaneous_frequency = np.gradient(np.unwrap(np.angle(analytic_signal))) * decimated_rate / (2 * np.pi)
    smoothing_length = max(1, int(0.006 * decimated_rate))
    smoothed_frequency = np.convolve(instantaneous_frequency, np.ones(smoothing_length) / smoothing_length, "same")

    def f_at(sample_index):
        half_window = max(1, int(0.005 * decimated_rate))
        window_start, window_end = max(0, sample_index - half_window), min(sample_count, sample_index + half_window)
        return np.average(smoothed_frequency[window_start:window_end], weights=amplitude[window_start:window_end] + 1e-12)

    # onset refinement on the 80-300 Hz envelope (full-rate would be better; 0.45 ms res here)
    high_band_envelope = np.abs(hilbert(sosfiltfilt(butter(4, [80, 300], btype="band", fs=decimated_rate, output="sos"), kick_band)))
    high_band_envelope = np.convolve(high_band_envelope, np.ones(max(1, int(.002 * decimated_rate))) / max(1, int(.002 * decimated_rate)), "same")

    onsets = []
    for peak_index in peak_indices:
        if peak_index + int(.16 * decimated_rate) >= sample_count or peak_index - int(.06 * decimated_rate) < 0:
            continue
        frequencies = [f_at(peak_index + int(delay_seconds * decimated_rate)) for delay_seconds in (0.003, .025, .05, .1, .15)]
        has_kick_shape = (100 <= frequencies[0] <= 240 and frequencies[2] <= 75 and 25 <= frequencies[4] <= 70)
        if not has_kick_shape:
            continue
        search_start = peak_index - int(.02 * decimated_rate)
        search_end = peak_index + int(.02 * decimated_rate)
        floor = np.percentile(high_band_envelope[peak_index - int(.05 * decimated_rate):peak_index - int(.02 * decimated_rate)], 90)
        plateau = np.max(high_band_envelope[peak_index:peak_index + int(.05 * decimated_rate)])
        onset_index = search_start + int(np.argmax(high_band_envelope[search_start:search_end] > floor + onset_fraction * (plateau - floor)))
        onsets.append((onset_index / decimated_rate, float(score[peak_index])))
    return onsets


def detect_snares(samples, sample_rate, ratio_min=3.0, rel_level=0.35, reference_percentile=97, onset_fraction=0.25, low_threshold_fraction=None):
    """Broadband (2-12 kHz) energy jump detector with onset refinement.
    The loudness gate is relative to the track's own loud hits (reference_percentile percentile
    of the 'after' level at candidate peaks), not a single window max.
    Returns list of (time_s, strength 0..~1); with low_threshold_fraction a third element holds the onset time at that lower threshold fraction."""
    snare_band = sosfiltfilt(butter(4, [2000, 12000], btype="band", fs=sample_rate, output="sos"), samples)
    rectified = np.abs(snare_band)
    del snare_band
    sample_count = len(samples)
    cumulative_rectified = np.concatenate([[0], np.cumsum(rectified)])
    del rectified

    def smooth(milliseconds):
        # moving average of rect over k samples, window [i-h, i-h+k) clipped at the ends (slicing for the interior, same values as the gather form)
        window_length = max(1, int(sample_rate * milliseconds / 1000))
        half_length = window_length // 2
        smoothed = np.empty(sample_count)
        interior_start, interior_end = half_length, sample_count - window_length + half_length + 1
        if interior_end > interior_start:
            smoothed[interior_start:interior_end] = (cumulative_rectified[window_length:] - cumulative_rectified[:-window_length])[:interior_end - interior_start] / window_length
        else:
            interior_start = interior_end = min(max(interior_start, 0), sample_count)
        for gap_start, gap_end in ((0, interior_start), (interior_end, sample_count)):
            if gap_end > gap_start:
                indices = np.arange(gap_start, gap_end)
                window_start = np.clip(indices - half_length, 0, sample_count); window_end = np.clip(indices - half_length + window_length, 0, sample_count)
                smoothed[gap_start:gap_end] = (cumulative_rectified[window_end] - cumulative_rectified[window_start]) / np.maximum(window_end - window_start, 1)
        return smoothed

    smoothed_envelope = smooth(1.0)
    fine = smooth(0.5)
    window_samples = int(0.03 * sample_rate)
    cumulative_envelope = np.concatenate([[0], np.cumsum(smoothed_envelope)])
    after = np.zeros(sample_count); before = np.zeros(sample_count)
    after[window_samples:sample_count - window_samples] = (cumulative_envelope[2 * window_samples:sample_count] - cumulative_envelope[window_samples:sample_count - window_samples]) / window_samples
    before[window_samples:sample_count - window_samples] = (cumulative_envelope[window_samples:sample_count - window_samples] - cumulative_envelope[:sample_count - 2 * window_samples]) / window_samples
    ratio = after / (before + 1e-9)

    # candidate peaks of the ratio (>= ratio_min), spaced >= 120 ms
    peak_indices, _ = find_peaks(ratio * (ratio >= ratio_min), height=ratio_min, distance=int(0.12 * sample_rate))
    if len(peak_indices) == 0:
        return []
    peak_levels = after[peak_indices]
    reference_level = np.percentile(peak_levels, reference_percentile)
    peak_indices = peak_indices[peak_levels >= rel_level * reference_level]

    onsets = []
    for peak_index in peak_indices:
        if peak_index - int(.04 * sample_rate) < 0 or peak_index + int(.03 * sample_rate) >= sample_count:
            continue
        floor = np.percentile(fine[peak_index - int(.04 * sample_rate):peak_index - int(.015 * sample_rate)], 90)
        plateau = np.median(fine[peak_index + int(.005 * sample_rate):peak_index + int(.025 * sample_rate)])
        threshold = floor + onset_fraction * (plateau - floor)
        search_start = peak_index - int(.015 * sample_rate)
        onset_index = search_start + int(np.argmax(fine[search_start:peak_index + int(.01 * sample_rate)] > threshold))
        if low_threshold_fraction:
            low_threshold_index = search_start + int(np.argmax(fine[search_start:peak_index + int(.01 * sample_rate)] > floor + low_threshold_fraction * (plateau - floor)))
            onsets.append((onset_index / sample_rate, float(min(after[peak_index] / reference_level, 1.5)), low_threshold_index / sample_rate))
        else:
            onsets.append((onset_index / sample_rate, float(min(after[peak_index] / reference_level, 1.5))))
    return onsets


_AUDIO_MEMO = {}


def load_audio(path, sample_rate=None):
    """Decode once per process: the last decoded file is kept (read-only), later calls for the same file and rate reuse it."""
    import os
    try:
        key = (str(path), sample_rate, os.path.getmtime(path))
    except OSError:
        return _load_audio(path, sample_rate)
    memo_entry = _AUDIO_MEMO.get(key)
    if memo_entry is None:
        _AUDIO_MEMO.clear()
        samples, rate = _load_audio(path, sample_rate)
        samples.setflags(write=False)
        memo_entry = _AUDIO_MEMO[key] = (samples, rate)
    return memo_entry


_PROBE_MEMO = {}


def probe_json(path):
    """ffprobe streams+format as a dict, one call per file and process (decode, header facts and offset model share it)."""
    import os
    import subprocess
    import json
    from lib import ffmpeg_tools
    try:
        file_status = os.stat(path)
        key = (str(path), file_status.st_mtime, file_status.st_size)
    except OSError:
        key = None
    if key is not None and key in _PROBE_MEMO:
        return _PROBE_MEMO[key]
    probe_result = json.loads(subprocess.run([ffmpeg_tools.executable("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                                  capture_output=True, text=True).stdout)
    if key is not None:
        _PROBE_MEMO.clear()
        _PROBE_MEMO[key] = probe_result
    return probe_result


def _load_audio(path, sample_rate=None):
    """Load mono float32 through ffmpeg (fixed decoder, so header delay handling is the same for every container).
    librosa is only a fallback."""
    import subprocess
    from lib import ffmpeg_tools
    if ffmpeg_tools.is_available("ffmpeg"):
        try:
            rate = sample_rate
            if rate is None:
                rate = int(next(stream for stream in probe_json(path)["streams"] if stream.get("codec_type") == "audio")["sample_rate"])
            raw = subprocess.run([ffmpeg_tools.executable("ffmpeg"), "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                                  "-ar", str(rate), "-"], capture_output=True, check=True).stdout
            return np.frombuffer(raw, dtype=np.float32).copy(), rate
        except Exception:
            pass
    if librosa is None:
        raise RuntimeError("could not decode %s (ffmpeg failed or missing, librosa not installed)" % path)
    return librosa.load(path, sr=sample_rate, mono=True)


# ----------------------------------------------------------------------------
# BPM / grid fitting
# ----------------------------------------------------------------------------
from scipy.ndimage import gaussian_filter1d

BINS_PER_BAR = 256                       # bins per bar (4 beats) when folding events
# bar-position templates in EIGHTH-note slots 0..7 (slot 0 = downbeat)
TEMPLATES = {
    "dubstep": dict(K=(0, 4),       S=(4,)),
    "dnb":     dict(K=(0, 5),       S=(2, 6)),
    "ukg_2step": dict(K=(0,),       S=(2, 6)),
    "ukg_4otf":  dict(K=(0, 2, 4, 6), S=(2, 6)),
}
MISSING_KICK_PENALTY, MISSING_SNARE_PENALTY = 0.5, 0.2      # penalty per unit weight of an event that misses the template
SIGMA_MILLISECONDS = 12.0                # slop allowed around a slot when scoring


def make_events(kicks, snares, kick_bias_s):
    """-> arrays times, weights, classes (0=kick,1=snare), low_threshold_times. Kick times are corrected by kick_bias_s
    (kick onset detector fires later than the snare one for simultaneous hits).
    low_threshold_times = times, except snares detected with low_threshold_fraction, which carry their low-threshold onset time."""
    times = [kick[0] - kick_bias_s for kick in kicks] + [snare[0] for snare in snares]
    low_threshold_times = list(times[:len(kicks)]) + [snare[2] if len(snare) > 2 else snare[0] for snare in snares]
    weights = [1.0 * min(kick[1], 1.0) for kick in kicks] + [0.6 * min(snare[1], 1.0) for snare in snares]
    classes = [0] * len(kicks) + [1] * len(snares)
    order = np.argsort(times)
    return np.array(times)[order], np.array(weights)[order], np.array(classes)[order], np.array(low_threshold_times)[order]


def estimate_kick_bias(kicks, snares, default=0.004, tolerance=0.012):
    """Median (kick - snare) over near-coincident pairs; kicks are detected a few ms late."""
    snare_times = np.array([snare[0] for snare in snares]) if len(snares) else np.array([])
    differences = []
    for kick in kicks:
        if len(snare_times):
            nearest_index = np.argmin(np.abs(snare_times - kick[0]))
            if abs(snare_times[nearest_index] - kick[0]) < tolerance:
                differences.append(kick[0] - snare_times[nearest_index])
    if len(differences) >= 3:
        return float(np.median(differences)), len(differences)
    return default, len(differences)


_SCAN_CACHE = {}


def scan_bpm(event_times, event_weights, event_classes, bpm_grid):
    """For each bpm fold events into a 4-beat bar, score every template and every
    phase. Returns array (len(bpm_grid), n_templates) of best score and the
    argmax phase bin. Higher is better. Vectorised over the bpm grid."""
    names = list(TEMPLATES)
    bpm_grid = np.asarray(bpm_grid, float)
    bpm_count = len(bpm_grid)
    is_kick = event_classes == 0
    kick_weight_total, snare_weight_total = event_weights[is_kick].sum(), event_weights[~is_kick].sum()
    key = bpm_grid.tobytes()
    cached = _SCAN_CACHE.get(key)
    if cached is None:
        built_bar_seconds = 4 * 60.0 / bpm_grid
        sigma_bins = SIGMA_MILLISECONDS / 1000.0 / built_bar_seconds * BINS_PER_BAR
        omega = 2 * np.pi * np.arange(BINS_PER_BAR // 2 + 1) / BINS_PER_BAR
        built_transfer = np.exp(-0.5 * (omega[None, :] * sigma_bins[:, None]) ** 2) * (np.sqrt(2 * np.pi) * sigma_bins)[:, None]
        frequency_bins = np.arange(BINS_PER_BAR // 2 + 1)
        phase_templates = {}
        for template_name in names:
            slots = TEMPLATES[template_name]
            phase_templates[template_name] = ((1 + MISSING_KICK_PENALTY) * sum(np.exp(2j * np.pi * frequency_bins * (slot * (BINS_PER_BAR // 8)) / BINS_PER_BAR) for slot in slots["K"]),
                      (1 + MISSING_SNARE_PENALTY) * sum(np.exp(2j * np.pi * frequency_bins * (slot * (BINS_PER_BAR // 8)) / BINS_PER_BAR) for slot in slots["S"]))
        if len(_SCAN_CACHE) >= 6:
            _SCAN_CACHE.clear()
        cached = _SCAN_CACHE[key] = (built_bar_seconds, built_transfer, (np.arange(bpm_count) * BINS_PER_BAR)[:, None], phase_templates)
    bar_seconds, transfer, rows, phase_templates = cached

    def spec(selection):
        # smoothed fold histogram, kept in the frequency domain (rows = bpm candidates)
        if not selection.any():
            return np.zeros((bpm_count, BINS_PER_BAR // 2 + 1), complex)
        bar_position = (event_times[selection][None, :] % bar_seconds[:, None]) / bar_seconds[:, None]
        flat_bins = rows + (bar_position * BINS_PER_BAR).astype(int) % BINS_PER_BAR
        histogram = np.bincount(flat_bins.ravel(), weights=np.tile(event_weights[selection], bpm_count), minlength=bpm_count * BINS_PER_BAR).reshape(bpm_count, BINS_PER_BAR)
        return np.fft.rfft(histogram, axis=1) * transfer

    kick_spectrum, snare_spectrum = spec(is_kick), spec(~is_kick)
    best_scores = np.full((bpm_count, len(names)), -1e9)
    best_phase_bins = np.zeros((bpm_count, len(names)), int)
    buffer = np.empty_like(kick_spectrum)
    for template_index, template_name in enumerate(names):
        # np.roll(H, -sl*NB/8) == multiply the spectrum by exp(+i*omega*sl*NB/8); all slots of a template and both event classes share one inverse FFT
        kick_template_spectrum, snare_template_spectrum = phase_templates[template_name]
        np.multiply(kick_spectrum, kick_template_spectrum[None, :], out=buffer)
        buffer += snare_spectrum * snare_template_spectrum[None, :]
        combined = np.fft.irfft(buffer, n=BINS_PER_BAR, axis=1)
        scores = combined - (MISSING_KICK_PENALTY * kick_weight_total + MISSING_SNARE_PENALTY * snare_weight_total)
        best_phase = np.argmax(np.round(scores, 6), axis=1)
        best_scores[:, template_index] = scores[np.arange(bpm_count), best_phase]
        best_phase_bins[:, template_index] = best_phase
    return names, best_scores, best_phase_bins


def refine_grid(event_times, event_weights, event_classes, bpm, bar_start_seconds, template_name, tolerance=0.02, iteration_count=4):
    """Given rough bpm and bar-start time, assign events to template slots and
    do weighted least squares  t = t0 + k*(T/2)  (k integer eighth-index).
    Returns dict(bpm, t0, rms_ms, n_inliers, idx, resid)."""
    template_slots = TEMPLATES[template_name]
    half = 60.0 / bpm / 2
    first_beat_time = bar_start_seconds
    for _ in range(iteration_count):
        eighth_indices = np.round((event_times - first_beat_time) / half).astype(int)
        slots = eighth_indices % 8
        slot_is_allowed = np.array([(slots[i] in (template_slots["K"] if event_classes[i] == 0 else template_slots["S"])) for i in range(len(event_times))])
        residuals = event_times - (first_beat_time + eighth_indices * half)
        inlier_mask = slot_is_allowed & (np.abs(residuals) < tolerance)
        if inlier_mask.sum() < 4:
            return None
        design_matrix = np.stack([np.ones(inlier_mask.sum()), eighth_indices[inlier_mask]], 1)
        sqrt_weights = np.sqrt(event_weights[inlier_mask])
        solution, *_ = np.linalg.lstsq(design_matrix * sqrt_weights[:, None], event_times[inlier_mask] * sqrt_weights, rcond=None)
        first_beat_time, half = solution
        tolerance = max(0.008, tolerance * 0.7)
    eighth_indices = np.round((event_times - first_beat_time) / half).astype(int)
    residuals = event_times - (first_beat_time + eighth_indices * half)
    slots = eighth_indices % 8
    slot_is_allowed = np.array([(slots[i] in (template_slots["K"] if event_classes[i] == 0 else template_slots["S"])) for i in range(len(event_times))])
    inlier_mask = slot_is_allowed & (np.abs(residuals) < tolerance * 1.5)
    return dict(bpm=60.0 / (2 * half), t0=first_beat_time, half=half, rms_ms=1000 * float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))),
                n_inl=int(inlier_mask.sum()), inl=inlier_mask, k=eighth_indices, res=residuals)


# ----------------------------------------------------------------------------
# Block-wise analysis: confident blocks -> first constant-tempo section -> grid
# ----------------------------------------------------------------------------
BLOCK_SECONDS, HOP_SECONDS = 30.0, 15.0
BPM_LOWER_LIMIT, BPM_UPPER_LIMIT = 95.0, 190.0


def lattice_fit(event_times, event_weights, beat_anchor_seconds, bpm, tolerance=0.020, iteration_count=5):
    """Template-free refit: events within tolerance of the eighth-note lattice anchored at
    beat_anchor_seconds (parity of k preserved: even k = beat). Weighted LS on t = anchor_seconds + k*half.
    Returns dict(bpm, a (= the anchor in seconds), rms_ms, n_inl, w_inl) or None."""
    half = 60.0 / bpm / 2
    anchor_seconds = beat_anchor_seconds
    for _ in range(iteration_count):
        eighth_indices = np.round((event_times - anchor_seconds) / half)
        residuals = event_times - (anchor_seconds + eighth_indices * half)
        inlier_mask = np.abs(residuals) < tolerance
        if inlier_mask.sum() < 5:
            return None
        design_matrix = np.stack([np.ones(inlier_mask.sum()), eighth_indices[inlier_mask]], 1)
        sqrt_weights = np.sqrt(event_weights[inlier_mask])
        solution, *_ = np.linalg.lstsq(design_matrix * sqrt_weights[:, None], event_times[inlier_mask] * sqrt_weights, rcond=None)
        anchor_seconds, half = solution
        tolerance = max(0.007, tolerance * 0.75)
    eighth_indices = np.round((event_times - anchor_seconds) / half)
    residuals = event_times - (anchor_seconds + eighth_indices * half)
    inlier_mask = np.abs(residuals) < tolerance * 1.3
    return dict(bpm=60.0 / (2 * half), a=anchor_seconds, rms_ms=1000 * float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))),
                n_inl=int(inlier_mask.sum()), w_inl=float(event_weights[inlier_mask].sum()))


def scan_block(event_times, event_weights, event_classes, block_start_seconds, grid):
    """Scan one block; times are made relative to t_start (keeps phase errors small).
    Returns dict(bpm, template, score, conf, beat0 (abs time of a template bar start)) or None."""
    if len(event_times) < 10:
        return None
    names, best_scores, best_phase_bins = scan_bpm(event_times - block_start_seconds, event_weights, event_classes, grid)
    flat_index = int(np.argmax(best_scores))
    bpm_index, template_index = divmod(flat_index, len(names))
    best_score = float(best_scores[bpm_index, template_index])
    weight_total = float(event_weights.sum())
    bar_seconds = 4 * 60.0 / grid[bpm_index]
    bar0 = block_start_seconds + best_phase_bins[bpm_index, template_index] / BINS_PER_BAR * bar_seconds
    # z-score of the peak against all bpm candidates (how much it stands out)
    best_per_bpm = best_scores.max(1)
    z_score = (best_score - np.median(best_per_bpm)) / (1.4826 * np.median(np.abs(best_per_bpm - np.median(best_per_bpm))) + 1e-9)
    return dict(bpm=float(grid[bpm_index]), template=names[template_index], score=best_score, frac=best_score / (weight_total + 1e-9), z=float(z_score),
                bar0=float(bar0), n=len(event_times))


def grow_section(event_times, event_weights, event_classes, evidence_mask, start, bpm, beat0, tolerance=0.015, window_size=10, max_good_in_bad=2):
    """Walk through evidence events (kicks if plentiful, else kicks+strong snares) after `start`.
    An event is 'good' if it lies within tol of the eighth-note lattice of the current grid.
    The grid is refit every 12 good events. The section ends when <= max_good_in_bad of the last
    `win` events are good (tempo/phase break). Returns (end_time, grid, broke_at or None)."""
    evidence_indices = np.where(evidence_mask & (event_times >= start))[0]
    current_fit = dict(bpm=bpm, beat0=beat0)
    unused_good_list = []
    good_flags = []
    last_good_time = start
    good_count = 0
    for position, event_index in enumerate(evidence_indices):
        half = 60.0 / current_fit["bpm"] / 2
        lattice_index = np.round((event_times[event_index] - current_fit["beat0"]) / half)
        is_good = abs(event_times[event_index] - (current_fit["beat0"] + lattice_index * half)) < tolerance
        good_flags.append(is_good)
        if is_good:
            last_good_time = event_times[event_index]; good_count += 1
            if good_count % 12 == 0:
                in_range = (event_times >= start) & (event_times <= event_times[event_index])
                refit = lattice_fit(event_times[in_range], event_weights[in_range], current_fit["beat0"], current_fit["bpm"], tolerance=0.012)
                if refit:
                    current_fit = dict(bpm=refit["bpm"], beat0=refit["a"])
        if len(good_flags) >= window_size and sum(good_flags[-window_size:]) <= max_good_in_bad:
            window_start_index = evidence_indices[position - window_size + 1]
            good_times = [event_times[evidence_indices[flag_index]] for flag_index, flag in enumerate(good_flags) if flag and evidence_indices[flag_index] < window_start_index]
            end = max(good_times) if good_times else start
            return end, current_fit, float(event_times[window_start_index])
    return last_good_time, current_fit, None


def final_fit(event_times, event_weights, start, end, grid):
    in_range = (event_times >= start) & (event_times <= end + 0.05)
    fit = lattice_fit(event_times[in_range], event_weights[in_range], grid["beat0"], grid["bpm"], tolerance=0.012)
    return fit



def fold_scores(relative_event_times, event_weights, event_classes, bpm):
    """Per-template score vector over all NB phase shifts at ONE bpm (same maths as scan_bpm)."""
    names = list(TEMPLATES)
    is_kick = event_classes == 0
    kick_weight_total, snare_weight_total = event_weights[is_kick].sum(), event_weights[~is_kick].sum()
    bar_seconds = 4 * 60.0 / bpm
    bar_position = (relative_event_times % bar_seconds) / bar_seconds
    sigma_bins = SIGMA_MILLISECONDS / 1000.0 / bar_seconds * BINS_PER_BAR
    smoothed_histogram = lambda selection: gaussian_filter1d(np.bincount((bar_position[selection] * BINS_PER_BAR).astype(int) % BINS_PER_BAR, weights=event_weights[selection], minlength=BINS_PER_BAR).astype(float), sigma_bins, mode="wrap") * np.sqrt(2 * np.pi) * sigma_bins
    kick_histogram, snare_histogram = smoothed_histogram(is_kick), smoothed_histogram(~is_kick)
    out = []
    for template_name in names:
        template_slots = TEMPLATES[template_name]
        kick_alignment = sum(np.roll(kick_histogram, -slot * (BINS_PER_BAR // 8)) for slot in template_slots["K"])
        snare_alignment = sum(np.roll(snare_histogram, -slot * (BINS_PER_BAR // 8)) for slot in template_slots["S"])
        out.append((kick_alignment - MISSING_KICK_PENALTY * (kick_weight_total - kick_alignment)) + (snare_alignment - MISSING_SNARE_PENALTY * (snare_weight_total - snare_alignment)))
    return names, np.array(out)


def block_lock(event_times, event_weights, event_classes, block_start, block_end, bpm, beat0):
    """Score of the given grid inside block [st,en) vs. the block's own best possible score.
    Returns (pred_score, best_score_at_this_bpm_any_phase, W_total, n)."""
    in_block = (event_times >= block_start) & (event_times < block_end)
    event_count = int(in_block.sum())
    if event_count < 8:
        return None
    names, scores = fold_scores(event_times[in_block] - block_start, event_weights[in_block], event_classes[in_block], bpm)
    bar_seconds = 4 * 60.0 / bpm
    beat_seconds = 60.0 / bpm
    predicted_score = -1e9
    for beat_index in range(4):
        start_slot = int(round(((beat0 + beat_index * beat_seconds - block_start) % bar_seconds) / bar_seconds * BINS_PER_BAR)) % BINS_PER_BAR
        for slot_shift in (-1, 0, 1):
            predicted_score = max(predicted_score, scores[:, (start_slot + slot_shift) % BINS_PER_BAR].max())
    return float(predicted_score), float(scores.max()), float(event_weights[in_block].sum()), event_count


def refine_window(event_times, event_weights, event_classes, start, end, bpm, span=0.15, step=0.002):
    """Fine bpm scan on all events in [start,end) (times relative to start). Returns (bpm, bar0_abs, template)."""
    in_window = (event_times >= start) & (event_times < end)
    bpm_candidates = np.arange(bpm - span, bpm + span + 1e-9, step)
    names, best_scores, best_phase_bins = scan_bpm(event_times[in_window] - start, event_weights[in_window], event_classes[in_window], bpm_candidates)
    bpm_index, template_index = divmod(int(np.argmax(best_scores)), len(names))
    bar_seconds = 4 * 60.0 / bpm_candidates[bpm_index]
    return float(bpm_candidates[bpm_index]), float(start + best_phase_bins[bpm_index, template_index] / BINS_PER_BAR * bar_seconds), names[template_index]


def analyze(path, verbose=False, block_s=BLOCK_SECONDS, hop_s=HOP_SECONDS, max_sections=4):
    samples, sample_rate = load_audio(path)
    duration_seconds = len(samples) / sample_rate
    kicks = detect_kicks(samples, sample_rate)
    snares = detect_snares(samples, sample_rate)
    kick_bias, pair_count = estimate_kick_bias(kicks, snares)
    event_times, event_weights, event_classes, _ = make_events(kicks, snares, kick_bias)
    bpm_candidates = np.arange(BPM_LOWER_LIMIT, BPM_UPPER_LIMIT, 0.03)

    # evidence for lattice tracking: kicks when plentiful (high precision), else kicks + strong snares
    strong_snare = (event_classes == 1) & (event_weights >= 0.6 * 0.8)
    evidence_mask = (event_classes == 0) if len(kicks) >= 30 else ((event_classes == 0) | strong_snare)

    # 1) per-block scans -> seeds
    blocks = []
    block_start = 0.0
    while block_start < duration_seconds - 8:
        in_block = (event_times >= block_start) & (event_times < block_start + block_s)
        blocks.append(dict(start=block_start, end=min(block_start + block_s, duration_seconds), res=scan_block(event_times[in_block], event_weights[in_block], event_classes[in_block], block_start, bpm_candidates)))
        block_start += hop_s
    for block in blocks:
        result = block["res"]
        block["confident"] = bool(result and result["n"] >= 12 and result["frac"] >= 0.25 and result["z"] >= 6)

    # 2) sections: seed from a confident block, grow block by block, refining bpm over the whole window
    sections = []
    min_start = 0.0
    while len(sections) < max_sections:
        seed = next((block for block in blocks if block["confident"] and block["start"] >= min_start), None)
        if seed is None:
            break
        result = seed["res"]
        section_start, section_end = seed["start"], seed["end"]
        section_bpm, section_anchor, template_name = refine_window(event_times, event_weights, event_classes, section_start, section_end, result["bpm"], span=0.1)
        break_time = None
        for block in blocks:
            if block["start"] <= seed["start"]:
                continue
            block_result = block["res"]
            if block_result is None or block_result["n"] < 10 or block_result["frac"] < 0.04:
                continue                         # no drum evidence in this block
            lock = block_lock(event_times, event_weights, event_classes, block["start"], block["end"], section_bpm, section_anchor)
            if lock is None:
                continue
            predicted_score = lock[0]
            if predicted_score >= 0.6 * block_result["score"] and predicted_score > 0:
                section_end = block["end"]
                section_bpm, section_anchor, template_name = refine_window(event_times, event_weights, event_classes, section_start, section_end, section_bpm, span=0.06, step=0.001)
            elif predicted_score < 0.3 * block_result["score"] or predicted_score <= 0:
                break_time = block["start"]
                break
            # in-between: ambiguous, ignore
        fit = final_fit(event_times, event_weights, section_start, section_end, dict(bpm=section_bpm, beat0=section_anchor))
        if fit is None:
            min_start = seed["start"] + hop_s
            continue
        beat_period = 60.0 / fit["bpm"]
        sections.append(dict(start_s=round(section_start, 2), end_s=round(float(section_end), 2), bpm=round(fit["bpm"], 4),
                             grid_anchor_s=round(float(fit["a"]), 5), beat_period_s=round(beat_period, 6),
                             fit_rms_ms=round(fit["rms_ms"], 2), n_inliers=fit["n_inl"],
                             break_detected_at_s=None if break_time is None else round(break_time, 2),
                             template=template_name, _a=fit["a"], _T=beat_period))
        if break_time is None:
            break
        min_start = float(section_end) + 10.0
    if not sections:
        return dict(file=path, duration_s=duration_seconds, error="no confident drum section found",
                    n_kicks=len(kicks), n_snares=len(snares))

    first_section = sections[0]
    flags = []
    if first_section["break_detected_at_s"] is not None:
        flags.append("bpm_maybe_not_constant")
    first_beat_index = int(np.ceil((first_section["start_s"] - first_section["_a"]) / first_section["_T"]))
    out = dict(
        file=path, duration_s=round(duration_seconds, 2), bpm=first_section["bpm"], grid_anchor_s=first_section["grid_anchor_s"], beat_period_s=first_section["beat_period_s"],
        first_section=dict(start_s=first_section["start_s"], end_s=first_section["end_s"]), fit_rms_ms=first_section["fit_rms_ms"], n_inliers=first_section["n_inliers"],
        flags=flags, template_guess=first_section["template"], kick_bias_ms=round(kick_bias * 1000, 2), kick_snare_pairs=pair_count,
        n_kicks=len(kicks), n_snares=len(snares),
        beats_first_section_s=[round(float(first_section["_a"] + beat_index * first_section["_T"]), 4) for beat_index in range(first_beat_index, first_beat_index + 16)],
        sections=[{key: value for key, value in section.items() if not key.startswith("_")} for section in sections],
    )
    if verbose:
        out["blocks"] = [dict(start=block["start"], conf=block["confident"],
                              **({key: (round(value, 2) if isinstance(value, float) else value) for key, value in block["res"].items() if key != "bar0"} if block["res"] else {}))
                         for block in blocks]
    out["_events"] = (event_times, event_weights, event_classes)
    return out


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("file")
    argument_parser.add_argument("--json")
    argument_parser.add_argument("--verbose", action="store_true")
    arguments = argument_parser.parse_args()
    analysis = analyze(arguments.file, verbose=arguments.verbose)
    analysis.pop("_events", None)
    print(json.dumps(analysis, indent=1))
    if arguments.json:
        json.dump(analysis, open(arguments.json, "w"), indent=1)
