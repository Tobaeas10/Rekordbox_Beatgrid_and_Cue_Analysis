"""Sweep-onset kick detector (research): matched filter for downward kick sweeps starting up to 800 Hz and ending at 42-60 Hz,
decimated to 4.41 kHz; onset = start of the sweep (the vertical line in the spectrogram). usage: python kick_sweep_onset_probe.py [track ...] (needs WORK/sweep_probe_truth.json and the audio in debug/audio/)"""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys, json, pickle, numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, oaconvolve, find_peaks
from scipy import fft as sp_fft
from lib.beatgrid import tempo_detection

SWEEP_START_FREQUENCIES = (250, 400, 600, 800); SWEEP_END_FREQUENCIES = (42, 50, 60); SWEEP_DECAY_SECONDS = (0.006, 0.012, 0.02, 0.03)


def detect_sweeps(samples, sample_rate, min_score=0.5, start_frequencies=SWEEP_START_FREQUENCIES, end_frequencies=SWEEP_END_FREQUENCIES, decay_constants=SWEEP_DECAY_SECONDS, duration_seconds=0.10):
    filtered = sosfiltfilt(butter(4, [38, 900], btype='band', fs=sample_rate, output='sos'), samples)
    decimation_factor = max(1, int(round(sample_rate / 4410))); filtered = resample_poly(filtered, 1, decimation_factor); decimated_rate = sample_rate / decimation_factor
    template_length = int(duration_seconds * decimated_rate); template_times = np.arange(template_length) / decimated_rate; sample_count = len(filtered)
    amplitude_envelope = np.exp(-template_times / 0.08); fft_length = sp_fft.next_fast_len(sample_count + template_length - 1)
    energy = sp_fft.irfft(sp_fft.rfft(filtered ** 2, fft_length) * sp_fft.rfft((amplitude_envelope ** 2)[::-1], fft_length), fft_length)[template_length - 1:template_length - 1 + sample_count]
    normalisation = np.sqrt((amplitude_envelope ** 2).sum() * energy + 1e-9); filtered_float32 = filtered.astype(np.float32)
    best_score = np.zeros(sample_count); best_start_frequency = np.zeros(sample_count)
    for start_frequency in start_frequencies:
        for end_frequency in end_frequencies:
            for decay_constant in decay_constants:
                frequency_sweep = end_frequency + (start_frequency - end_frequency) * np.exp(-template_times / decay_constant)
                template = (amplitude_envelope * np.exp(2j * np.pi * np.cumsum(frequency_sweep) / decimated_rate)).astype(np.complex64)
                correlation = np.abs(oaconvolve(filtered_float32, np.conj(template)[::-1], mode='full')[template_length - 1:template_length - 1 + sample_count]) / normalisation
                improved = correlation > best_score; best_score = np.where(improved, correlation, best_score); best_start_frequency = np.where(improved, start_frequency, best_start_frequency)
    peak_indices, _ = find_peaks(best_score, height=min_score, distance=int(0.06 * decimated_rate))
    return [(p / decimated_rate, float(best_score[p]), float(best_start_frequency[p])) for p in peak_indices]


def lattice_report(events, truth_seconds, period_seconds, start_seconds, end_seconds, tolerance_ms=6.0, name=''):
    events = [event for event in events if start_seconds <= event[0] <= end_seconds]
    onset_times = np.array([event[0] for event in events]); scores = np.array([event[1] for event in events])
    offsets_ms = ((onset_times - truth_seconds + period_seconds / 2) % period_seconds - period_seconds / 2) * 1000
    on_lattice = np.abs(offsets_ms) <= tolerance_ms
    # mode of offsets (score weighted, 1 ms bins, 3 ms smoothing)
    histogram = np.histogram(offsets_ms, bins=np.arange(-period_seconds * 500, period_seconds * 500 + 1, 1.0), weights=scores)[0]
    smoothed_histogram = np.convolve(histogram, np.ones(3), 'same'); best_offset_ms = np.argmax(smoothed_histogram) - period_seconds * 500 + 0.5
    remaining_histogram = smoothed_histogram.copy(); top_modes = []
    for _ in range(3):
        peak_index = int(np.argmax(remaining_histogram)); top_modes.append((round(peak_index - period_seconds * 500 + 0.5), round(float(remaining_histogram[peak_index] / (smoothed_histogram.sum() + 1e-9)), 2))); remaining_histogram[max(0, peak_index - 15):peak_index + 16] = 0
    return dict(n=len(onset_times), on=round(float(on_lattice.mean()), 2), modes=top_modes)


if __name__ == '__main__':
    truth_by_track = json.load(open(str(paths.WORK / 'sweep_probe_truth.json'))); sweeps_by_track = {}
    for track_name in (sys.argv[1:] or truth_by_track):
        track_truth = truth_by_track[track_name]; extension = 'mp3' if track_name == 'lowtek' else 'm4a'
        samples, sample_rate = tempo_detection.load_audio(f'{paths.DEBUG}/audio/{track_name}.{extension}')
        events = detect_sweeps(samples, sample_rate, min_score=float(__import__('os').environ.get('MS', 0.5))); period_seconds = 60 / track_truth['bpm']
        cached_events = pickle.load(open(track_truth['cache'] + track_truth['id'] + '.ff2.pkl', 'rb'))
        cached_kick_events = [(event_time, event_weight) for event_time, event_weight, event_class in zip(cached_events['t'], cached_events['w'], cached_events['c']) if event_class == 0]
        print(f"{track_name:10s} truth-vs-our {round((track_truth['truth_s'] - track_truth['anchor_s']) * 1000)} ms | sweep {lattice_report(events, track_truth['truth_s'], period_seconds, track_truth['start'], track_truth['end'])} | old kick {lattice_report(cached_kick_events, track_truth['truth_s'], period_seconds, track_truth['start'], track_truth['end'])}", flush=True)
        sweeps_by_track[track_name] = events
    pickle.dump(sweeps_by_track, open(str(paths.WORK / 'sweep_probe_events.pkl'), 'wb'))
