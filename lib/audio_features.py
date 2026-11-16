"""Audio feature extraction: per-8th band energies (band energies), per-beat energies / raw-audio features (beat features), 100 Hz band envelopes (band envelopes)."""
import copy
import time

from scipy.signal import butter, sosfiltfilt, sosfilt
import numpy as np
from lib.beatgrid import tempo_detection


# ---------------------------------------------------------------- band energies (10 bands per 8th note, input of the cue models)
BAND_EDGES_HZ = [30, 60, 100, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000]


def _power_spectrogram(samples, sample_rate, window_size, hop_seconds):
    """Power spectrogram, float32 FFT, hop in seconds. Returns (frames x bins, frame start times)."""
    import scipy.fft as scipy_fft
    from numpy.lib.stride_tricks import sliding_window_view
    hop_samples = max(1, int(round(sample_rate * hop_seconds)))
    padded_samples = np.concatenate([np.zeros(window_size // 2, np.float32), samples.astype(np.float32), np.zeros(window_size, np.float32)])
    frames = sliding_window_view(padded_samples, window_size)[::hop_samples]
    window = np.hanning(window_size).astype(np.float32)
    power = np.empty((len(frames), window_size // 2 + 1), np.float32)
    for chunk_start in range(0, len(frames), 4000):
        spectrum = scipy_fft.rfft(frames[chunk_start:chunk_start + 4000] * window, axis=1)
        power[chunk_start:chunk_start + 4000] = spectrum.real ** 2 + spectrum.imag ** 2
    frame_count = len(samples) // hop_samples + 1
    return power[:frame_count], np.arange(min(frame_count, len(power))) * hop_samples / sample_rate


def compute_band_energies(samples, sample_rate, first_beat_time, period_seconds, beat_count):
    from scipy.signal import resample_poly
    hop_seconds = 0.02
    slot_count = beat_count * 2; slot_seconds = period_seconds / 2.0
    levels = np.zeros((slot_count, 10)); fluxes = np.zeros((slot_count, 10)); centroids = np.zeros(slot_count); rolloffs = np.zeros(slot_count)
    samples = samples.astype(np.float32)
    low_samples = resample_poly(samples, 1, 8).astype(np.float32); low_sample_rate = sample_rate / 8.0     # low bands (<500 Hz) from 8x downsampled audio
    for band_samples, band_sample_rate, window_size, bands in ((low_samples, low_sample_rate, 1024, range(0, 4)), (samples, sample_rate, 1024, range(4, 10))):
        power, frame_times = _power_spectrogram(band_samples, band_sample_rate, window_size, hop_seconds); power = power[:len(frame_times)]
        frequencies = np.fft.rfftfreq(window_size, 1 / band_sample_rate)
        slot_index = np.floor((frame_times - first_beat_time) / slot_seconds).astype(int)
        in_range = (slot_index >= 0) & (slot_index < slot_count); slot_of_frame = slot_index[in_range]
        frames_per_slot = np.maximum(np.bincount(slot_of_frame, minlength=slot_count), 1)
        for band in bands:
            frequency_mask = (frequencies >= BAND_EDGES_HZ[band]) & (frequencies < BAND_EDGES_HZ[band + 1])
            level_db = 10 * np.log10(power[:, frequency_mask].sum(1) + 1e-10)
            spectral_flux = np.maximum(np.diff(level_db, prepend=level_db[0]), 0)
            levels[:, band] = np.bincount(slot_of_frame, level_db[in_range], slot_count) / frames_per_slot
            fluxes[:, band] = np.bincount(slot_of_frame, spectral_flux[in_range], slot_count)
        if band_sample_rate == sample_rate:
            frequency_mask = (frequencies >= 30) & (frequencies <= 16000); masked_power = power[:, frequency_mask]; kilohertz = frequencies[frequency_mask] / 1000.0
            total_power = masked_power.sum(1) + 1e-12
            centroid = (masked_power * kilohertz[None, :]).sum(1) / total_power
            rolloff = kilohertz[np.minimum((np.cumsum(masked_power, 1) < 0.85 * total_power[:, None]).sum(1), len(kilohertz) - 1)]
            centroids = np.bincount(slot_of_frame, centroid[in_range], slot_count) / frames_per_slot
            rolloffs = np.bincount(slot_of_frame, rolloff[in_range], slot_count) / frames_per_slot
    return np.concatenate([levels, fluxes, centroids[:, None], rolloffs[:, None]], 1).reshape(beat_count, 2, 22).astype(np.float16)


# ---------------------------------------------------------------- beat energies and raw-audio research features
BEAT_ENERGY_BANDS = ((30, 100), (100, 500), (500, 3000), (3000, 10000))
HALF_WINDOW_MILLISECONDS = 30
MAX_HALF_BEAT_WINDOWS = 300


def band_envelopes_1ms(samples, sample_rate):
    block_size = max(1, sample_rate // 1000)
    band_envelopes = []
    for low_hz, high_hz in BEAT_ENERGY_BANDS:
        filter_sections = butter(4, [low_hz, min(high_hz, sample_rate / 2 * 0.95)], btype="bandpass", fs=sample_rate, output="sos")
        filtered = np.abs(sosfiltfilt(filter_sections, samples)).astype(np.float32)
        frame_count = len(filtered) // block_size
        band_envelopes.append(filtered[:frame_count * block_size].reshape(frame_count, block_size).mean(1))
    return np.stack(band_envelopes), block_size / sample_rate


def windows(events, section, anchor_seconds, samples, sample_rate):
    """The W / Wk / Wpar / Wraw windows around the half-beat lattice of one section (anchor a, decoded audio y at rate sr).
    Returns (dict, E, dt, n, c) where E is the 1 ms band envelope (needed for BE)."""
    period_seconds = section["period_s"]
    half = period_seconds / 2
    features = {}
    envelopes, frame_seconds = band_envelopes_1ms(samples, sample_rate)
    frame_count = envelopes.shape[1]
    # half-beat windows
    first_half_beat = int(np.ceil((max(section["start_s"], 0.05) - anchor_seconds) / half))
    last_half_beat = int(np.floor((min(section["end_s"], events["dur"] - 0.05) - anchor_seconds) / half))
    half_beat_indices = np.arange(first_half_beat, last_half_beat + 1)
    centers = anchor_seconds + half_beat_indices * half
    offsets_seconds = np.arange(-HALF_WINDOW_MILLISECONDS, HALF_WINDOW_MILLISECONDS + 1) * 0.001
    frame_indices = np.rint((centers[:, None] + offsets_seconds[None, :]) / frame_seconds).astype(int)
    in_range = (frame_indices[:, 0] >= 0) & (frame_indices[:, -1] < frame_count)
    half_beat_indices, centers, frame_indices = half_beat_indices[in_range], centers[in_range], frame_indices[in_range]
    band_windows = np.stack([envelopes[band_index][frame_indices] for band_index in range(len(BEAT_ENERGY_BANDS))], 1)  # K,4,61
    if len(half_beat_indices) > MAX_HALF_BEAT_WINDOWS:
        loudness = band_windows[:, 2].max(1)
        kept = np.sort(np.argsort(-loudness)[:MAX_HALF_BEAT_WINDOWS])
        half_beat_indices, centers, band_windows = half_beat_indices[kept], centers[kept], band_windows[kept]
    scale = np.median(band_windows.max(2), 0)[None, :, None] + 1e-9
    features["W"] = (band_windows / scale).astype(np.float16)
    features["Wk"] = half_beat_indices.astype(np.int32)
    features["Wpar"] = (half_beat_indices % 2).astype(np.int8)
    # raw full band at 0.25 ms
    step = max(1, int(round(0.00025 * sample_rate)))
    radius_samples = int(0.015 * sample_rate)
    raw_rows = []
    for center_time in centers:
        sample_index = int(round(center_time * sample_rate))
        segment = samples[sample_index - radius_samples:sample_index + radius_samples] if sample_index - radius_samples >= 0 and sample_index + radius_samples <= len(samples) else None
        if segment is None or len(segment) < 2 * radius_samples:
            raw_rows.append(np.zeros(2 * radius_samples // step, np.float32))
        else:
            raw_rows.append(np.abs(segment[:(len(segment) // step) * step]).reshape(-1, step).mean(1))
    raw_windows = np.array(raw_rows, np.float32)
    features["Wraw"] = (raw_windows / (np.median(raw_windows.max(1)) + 1e-9)).astype(np.float16)
    return features, envelopes, frame_seconds, frame_count, centers


def beat_energies(envelopes, frame_seconds, anchor_seconds, period_seconds, duration_seconds):
    """Per-beat band energies [pre, onset, sustain] over the whole track from the first lattice beat >= 0 on.
    Returns (BE float16 (nb, 4, 3), t0 = time of beat 0 in decoded seconds)."""
    frame_count = envelopes.shape[1]
    leading_beat_count = int(np.ceil(-anchor_seconds / period_seconds))
    first_beat_time = anchor_seconds + leading_beat_count * period_seconds
    beat_count = int((duration_seconds - first_beat_time) / period_seconds)
    beat_times = first_beat_time + np.arange(beat_count) * period_seconds
    cumulative_sums = np.concatenate([np.zeros((envelopes.shape[0], 1)), np.cumsum(envelopes.astype(np.float64), 1)], 1)

    def mean_window(start_fraction, end_fraction):
        start_frames = np.clip(np.rint((beat_times + start_fraction * period_seconds) / frame_seconds).astype(int), 0, frame_count)
        end_frames = np.clip(np.rint((beat_times + end_fraction * period_seconds) / frame_seconds).astype(int), 0, frame_count)
        window_lengths = np.maximum(end_frames - start_frames, 1)
        return (cumulative_sums[:, end_frames] - cumulative_sums[:, start_frames]) / window_lengths  # 4, nb
    energies = np.stack([mean_window(-0.45, -0.05), mean_window(-0.03, 0.25), mean_window(0.25, 0.7)], 2).transpose(1, 0, 2)  # nb,4,3
    energies = energies / (np.median(energies.reshape(len(energies), 4, 3).max(2), 0)[None, :, None] + 1e-9)
    return energies.astype(np.float16), first_beat_time


def compute_beat_features(events, sections, audio, beats, model_ms):
    from lib.beatgrid import tempo_detection
    from lib.beatgrid import grid_pipeline
    section = sections[0]
    period_seconds = section["period_s"]
    half = period_seconds / 2
    anchor_seconds = section["anchor_s"] - (section.get("fine_shift_ms") or 0.0) / 1000.0
    features = dict(info_anchor_pre_s=anchor_seconds, info_period_s=period_seconds, info_start_s=section["start_s"], info_end_s=section["end_s"],
               info_fine_raw_ms=float(section.get("fine_raw_ms", 0.0)), info_fit_rms_ms=float(section.get("rms_ms", section.get("fit_rms_ms", 0.0))), info_dur_s=float(events["dur"]),
               info_n_sections=len(sections), info_model_ms=float(model_ms))
    shifted_sections = copy.deepcopy(sections)
    shifted_sections[0]["anchor_s"] = anchor_seconds
    shifted_sections[0]["fine_shift_ms"] = 0.0
    built = grid_pipeline.build_grid(shifted_sections, model_ms, events["dur"], events)
    if built:
        dry_compare_result = grid_pipeline.dry_compare(built, beats)
        for key, value in dry_compare_result.items():
            if isinstance(value, (int, float, bool)):
                features["label_" + key] = float(value)
        features["label_q_applied"] = float((shifted_sections[0].get("downbeat") or {}).get("q", 0))
        features["label_first_built_ms"] = float(built[0][2])
    samples, sample_rate = tempo_detection.load_audio(audio)
    features["info_sr"] = float(sample_rate)
    window_features, envelopes, frame_seconds, frame_count, centers = windows(events, section, anchor_seconds, samples, sample_rate)
    features.update(window_features)
    beat_energy_table, first_beat_time = beat_energies(envelopes, frame_seconds, anchor_seconds, period_seconds, events["dur"])
    features["info_lattice_t0_s"] = first_beat_time
    features["BE"] = beat_energy_table.astype(np.float16)
    return features


# ---------------------------------------------------------------- 100 Hz band envelopes
ENVELOPE_BANDS = ((30, 90), (90, 250), (250, 2000), (2000, 12000))


def band_envelope_array(samples, sample_rate, rate=100, bands=None):
    """-> (float32 array (len(ENVELOPE_BANDS), n_frames), hop_samples). Frame k covers samples [k*hop_samples, (k+1)*hop_samples), value = band RMS.
    bands: optional indices of the bands to compute (others stay zero)."""
    hop_samples = max(1, int(round(sample_rate / rate)))
    frame_count = len(samples) // hop_samples
    band_envelope_table = np.zeros((len(ENVELOPE_BANDS), frame_count), np.float32)
    samples = samples.astype(np.float32)
    for band_index, (low_hz, high_hz) in enumerate(ENVELOPE_BANDS):
        if bands is not None and band_index not in bands:
            continue
        filter_sections = butter(2, [low_hz, high_hz], btype='bandpass', fs=sample_rate, output='sos')
        filtered = sosfilt(filter_sections, samples)[: frame_count * hop_samples]
        band_envelope_table[band_index] = np.sqrt((filtered.reshape(frame_count, hop_samples) ** 2).mean(1))
    return band_envelope_table, hop_samples


def compute_band_envelopes(path):
    samples, sample_rate = tempo_detection.load_audio(path)
    envelopes, hop_samples = band_envelope_array(samples, sample_rate)
    return dict(env=envelopes.astype(np.float16), hop=np.int32(hop_samples), sr=np.int32(sample_rate), dur=np.float32(len(samples) / sample_rate))
