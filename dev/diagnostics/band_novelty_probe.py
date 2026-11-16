import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import numpy as np, scipy.signal as scipy_signal, sys, os, pickle
from lib.beatgrid.tempo_detection import load_audio

BANDS = {'sub': (35, 110), 'lo': (110, 400), 'mid': (400, 2500), 'hi': (2500, 7000), 'air': (7000, 16000)}

def band_envelope_array(samples, sample_rate, low_hz, high_hz, smooth_ms=1.5):
    sos = scipy_signal.butter(4, [low_hz, min(high_hz, sample_rate / 2 - 100)], btype='band', fs=sample_rate, output='sos')
    filtered = scipy_signal.sosfiltfilt(sos, samples)
    envelope = np.abs(scipy_signal.hilbert(filtered))
    window_samples = max(1, int(sample_rate * smooth_ms / 1000))
    envelope = np.convolve(envelope, np.ones(window_samples) / window_samples, mode='same')
    return envelope

def novelty(samples, sample_rate, hop_ms=1.0):
    hop_samples = int(round(sample_rate * hop_ms / 1000))
    novelty_by_band = {}
    for band_name, (low_hz, high_hz) in BANDS.items():
        envelope = band_envelope_array(samples, sample_rate, low_hz, high_hz)
        envelope = envelope[: len(envelope) // hop_samples * hop_samples].reshape(-1, hop_samples).mean(1)
        log_envelope = np.log(envelope + 1e-5 * envelope.max())
        # slow baseline (30 ms) vs fast (2 ms): rise = attack
        smoothed = np.convolve(log_envelope, np.ones(3) / 3, mode='same')
        baseline = np.concatenate([np.full(25, smoothed[0]), smoothed[:-25]])
        novelty_values = np.maximum(smoothed - baseline, 0)
        novelty_by_band[band_name] = novelty_values.astype(np.float32)
    return novelty_by_band, hop_ms / 1000

if __name__ == '__main__':
    audio_path = sys.argv[1]
    samples, sample_rate = load_audio('../' + audio_path)
    novelty_by_band, hop_seconds = novelty(samples, sample_rate)
    pickle.dump((novelty_by_band, hop_seconds, len(samples) / sample_rate), open(audio_path + '.nov', 'wb'))
    print(audio_path, sample_rate, len(samples) / sample_rate, {band_name: float(values.mean()) for band_name, values in novelty_by_band.items()})
