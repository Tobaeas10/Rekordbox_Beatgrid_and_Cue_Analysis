import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import pickle, numpy as np, sys
novelty_path, bpm = sys.argv[1], float(sys.argv[2])
novelty_by_band, hop_seconds, duration_seconds = pickle.load(open(novelty_path + '.nov', 'rb'))
period_seconds = 60 / bpm
frame_count = len(next(iter(novelty_by_band.values())))
frame_times = np.arange(frame_count) * hop_seconds
bin_count = int(round(period_seconds / hop_seconds / 2))  # 2ms bins... use 1ms
bin_count = int(round(period_seconds / hop_seconds))
phase_bins = ((frame_times % period_seconds) / period_seconds * bin_count).astype(int) % bin_count
for band_name, values in novelty_by_band.items():
    histogram = np.bincount(phase_bins, weights=values, minlength=bin_count)
    histogram = np.convolve(np.r_[histogram[-3:], histogram, histogram[:3]], np.ones(3) / 3, mode='same')[3:-3]
    top = np.argsort(histogram)[::-1]
    peak_bins = []
    for bin_index in top:
        if all(min(abs(bin_index - chosen_bin), bin_count - abs(bin_index - chosen_bin)) > 15 for chosen_bin in peak_bins): peak_bins.append(bin_index)
        if len(peak_bins) == 6: break
    print(band_name, [(int(peak_bin * hop_seconds * 1000), round(float(histogram[peak_bin] / histogram.mean()), 1)) for peak_bin in peak_bins])
