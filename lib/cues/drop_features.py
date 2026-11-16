"""feats6.py: PRODUCTION feature pipeline (cand6). track_features(t) -> (cands, nb, F) for one track dict t (needs t['BE'], t['BX'], t['pl'], t['bpm']).
Identical to the per-track loop of cand6.py (kept as the reference/training script). Used for inference with drop_start_model.pkl + drop_end_model.pkl
and the cue_predictor.decode() decoder (probabilities -> D, e1, H, e2)."""
import numpy as np


# ---- per-beat features (beatfeats.py)
"""beatfeats.py (wip37): beat-resolved features per candidate bar, from BE (per beat, 4 bands x [pre,onset,sustain])."""
def _cosine_similarity(first_vectors, second_vectors):
    return (first_vectors * second_vectors).sum(1) / (np.linalg.norm(first_vectors, axis=1) * np.linalg.norm(second_vectors, axis=1) + 1e-6)
def beat_resolved_features(track):
    beat_energy_table = track['BE']; bar_count = len(beat_energy_table) // 4; beat_count = bar_count * 4
    log_levels = np.log1p(beat_energy_table[:beat_count].reshape(beat_count, 12))
    beat_normalized = (log_levels - np.median(log_levels, 0)) / (np.percentile(log_levels, 90, 0) - np.percentile(log_levels, 10, 0) + 1e-3)   # (nk,12) col=band*3+win
    bar_index = np.arange(bar_count + 1)
    def make_window_mean(values):
        cumulative_sums = np.cumsum(np.vstack([np.zeros((1, values.shape[1])), values]), 0)
        def window_mean(start_indices, end_indices):
            start_indices = np.clip(start_indices, 0, len(values)); end_indices = np.clip(end_indices, 0, len(values)); return (cumulative_sums[end_indices] - cumulative_sums[start_indices]) / np.maximum(end_indices - start_indices, 1)[:, None]
        return window_mean
    feature_blocks = []
    # 1. beat-level novelty (w=2,4 beats) around candidate: beats 4i-4..4i+4, all cols and low cols (bands 0,1)
    beat_index = np.arange(beat_count + 1)
    for cols in (slice(0, 12), slice(0, 6)):
        window_mean = make_window_mean(beat_normalized[:, cols])
        for window_length in (2, 4):
            novelty = np.linalg.norm(window_mean(beat_index, beat_index + window_length) - window_mean(beat_index - window_length, beat_index), axis=1)
            local_novelty = np.stack([novelty[np.clip(4 * bar_index + beat_offset, 0, beat_count)] for beat_offset in range(-4, 5)], 1)
            feature_blocks += [local_novelty, local_novelty.argmax(1)[:, None] - 4.0, local_novelty.max(1)[:, None]]
    # 2. pattern (beat-resolved bar vectors) novelty and self-similarity
    bar_patterns = beat_normalized.reshape(bar_count, 48); pattern_window_mean = make_window_mean(bar_patterns); bar_indices = bar_index
    for window_length in (1, 2, 4, 8):
        mean_before, mean_after = pattern_window_mean(bar_indices - window_length, bar_indices), pattern_window_mean(bar_indices, bar_indices + window_length)
        feature_blocks += [np.linalg.norm(mean_after - mean_before, axis=1)[:, None], (1 - _cosine_similarity(mean_before, mean_after))[:, None]]
    band_patterns = beat_normalized.reshape(bar_count, 4, 4, 3)
    for band in range(4):
        single_band_pattern = band_patterns[:, :, band, :].reshape(bar_count, 12); single_band_window_mean = make_window_mean(single_band_pattern)
        for window_length in (2, 4, 8):
            feature_blocks += [np.linalg.norm(single_band_window_mean(bar_indices, bar_indices + window_length) - single_band_window_mean(bar_indices - window_length, bar_indices), axis=1)[:, None]]
    for lag in (1, 2, 4, 8, 16):
        feature_blocks += [(1 - _cosine_similarity(bar_patterns[np.clip(bar_indices, 0, bar_count - 1)], bar_patterns[np.clip(bar_indices - lag, 0, bar_count - 1)]))[:, None],
              (1 - _cosine_similarity(bar_patterns[np.clip(bar_indices, 0, bar_count - 1)], bar_patterns[np.clip(bar_indices + lag, 0, bar_count - 1)]))[:, None]]
    # 3. onset density per band (beat flags) next/prev 1,2,4 bars
    for band in range(4):
        active_mask = (beat_normalized[:, band * 3 + 1] > 0.3).astype(np.float64); onset_window_mean = make_window_mean(active_mask[:, None])
        sustain_mask = (beat_normalized[:, band * 3 + 2] > 0.3).astype(np.float64); sustain_window_mean = make_window_mean(sustain_mask[:, None])
        for window_length in (1, 2, 4):
            next_onset_density, previous_onset_density = onset_window_mean(4 * bar_indices, 4 * (bar_indices + window_length)), onset_window_mean(4 * (bar_indices - window_length), 4 * bar_indices)
            next_sustain_density, previous_sustain_density = sustain_window_mean(4 * bar_indices, 4 * (bar_indices + window_length)), sustain_window_mean(4 * (bar_indices - window_length), 4 * bar_indices)
            feature_blocks += [next_onset_density, previous_onset_density, next_onset_density - previous_onset_density, next_sustain_density - previous_sustain_density]
    # 4. raw beat-level low/low-mid values for the 12 beats of bars i-1, i, i+1 (cols 0..5)
    raw_beat_values = np.stack([beat_normalized[np.clip(4 * bar_indices + beat_offset, 0, beat_count - 1), :6] for beat_offset in range(-4, 8)], 1).reshape(bar_count + 1, -1)
    feature_blocks.append(raw_beat_values)
    return np.concatenate(feature_blocks, 1).astype(np.float32)


# ---- per-8th-note band features (bandfeats.py)
"""bandfeats.py (wip37): per-candidate-bar features from BX (extract_band_features.py output, (nbeats,2,22) per 8th: 10 band dB levels, 10 band flux, centroid kHz, rolloff kHz).
Bands: 0-3 low (30-500), 4-6 mid (0.5-4k), 7 4-6k, 8 6-8k (hihat), 9 8-12k (top / low-pass filter).
Targets: after-drop (low end stays, mid/high drops), hihat flux (fakeout vs none), low-pass in fakeouts (8-12k vs 2-4k, rolloff), plus 8th-note resolution step placement."""
def eighth_note_band_features(track):
    band_energy_table = track['BX'].astype(np.float64); bar_count = len(band_energy_table) // 4; slot_count = bar_count * 8
    slot_table = band_energy_table[:bar_count * 4].reshape(slot_count, 22)                     # per 8th slot
    levels = slot_table[:, :10]; fluxes = slot_table[:, 10:20]; shape = slot_table[:, 20:22]
    level_z = (levels - np.median(levels, 0)) / (np.percentile(levels, 90, 0) - np.percentile(levels, 10, 0) + 1e-3)
    log_flux = np.log1p(np.maximum(fluxes, 0)); flux_z = (log_flux - np.median(log_flux, 0)) / (np.percentile(log_flux, 90, 0) - np.percentile(log_flux, 10, 0) + 1e-3)
    shape_z = (shape - np.median(shape, 0)) / (np.percentile(shape, 90, 0) - np.percentile(shape, 10, 0) + 1e-3)
    normalized_table = np.concatenate([level_z, flux_z, shape_z], 1)                 # (ns,22) normalised
    bar_table = normalized_table.reshape(bar_count, 8, 22).mean(1)                   # bar table
    column_group_mean = lambda table, first_column, end_column: table[:, first_column:end_column].mean(1)
    low, mid, high, top, hihat_flux = column_group_mean(level_z, 0, 4), column_group_mean(level_z, 4, 7), column_group_mean(level_z, 7, 9), level_z[:, 9], flux_z[:, 8]
    mid_flux, low_flux = column_group_mean(flux_z, 4, 7), column_group_mean(flux_z, 0, 4)
    series = np.stack([low, mid, high, top, hihat_flux, mid_flux, low_flux, mid - low, high - low, top - high, shape_z[:, 1], shape_z[:, 0]], 1)   # (ns,12)
    bar_series = series.reshape(bar_count, 8, 12).mean(1)               # bar series
    bar_index = np.arange(bar_count + 1)
    def make_window_mean(values):
        cumulative_sums = np.cumsum(np.vstack([np.zeros((1, values.shape[1])), values]), 0)
        def window_mean(start_indices, end_indices):
            start_indices = np.clip(start_indices, 0, len(values)); end_indices = np.clip(end_indices, 0, len(values)); return (cumulative_sums[end_indices] - cumulative_sums[start_indices]) / np.maximum(end_indices - start_indices, 1)[:, None]
        return window_mean
    bar_table_mean, bar_series_mean = make_window_mean(bar_table), make_window_mean(bar_series)
    feature_blocks = []
    for window_bars in (1, 2, 4, 8, 16):                           # all 22 dims after-before
        feature_blocks.append(bar_table_mean(bar_index, bar_index + window_bars) - bar_table_mean(bar_index - window_bars, bar_index))
    for window_bars in (4, 8, 16):                                  # series: after vs previous-16 and vs previous-w
        feature_blocks += [bar_series_mean(bar_index, bar_index + window_bars) - bar_series_mean(bar_index - 16, bar_index), bar_series_mean(bar_index, bar_index + window_bars) - bar_series_mean(bar_index - window_bars, bar_index)]
    for window_bars in (4, 8, 16):                                  # absolute series level next w bars
        feature_blocks.append(bar_series_mean(bar_index, bar_index + window_bars))
    feature_blocks.append(bar_series_mean(bar_index - 16, bar_index))
    for bar_offset in (-2, -1, 1, 2):                              # bar table at neighbours
        feature_blocks.append(bar_table[np.clip(bar_index + bar_offset, 0, bar_count - 1)])
    feature_blocks.append(bar_table[np.clip(bar_index, 0, bar_count - 1)])
    # 8th resolution around the boundary: low / mid / hi levels + hat flux at slots -8..+7 (2 bars)
    slot_series = series[:, [0, 1, 2, 4]]
    feature_blocks.append(np.stack([slot_series[np.clip(8 * bar_index + slot_offset, 0, slot_count - 1)] for slot_offset in range(-8, 8)], 1).reshape(bar_count + 1, -1))
    # 8th novelty (w=4 slots) at the 9 slots around boundary for low vs hi series, argmax offset
    slot_novelty_mean = make_window_mean(series[:, [0, 1, 2, 4]]); slot_index = np.arange(slot_count + 1)
    for cols in ([0], [1, 2, 3]):
        novelty = np.abs(slot_novelty_mean(slot_index, slot_index + 4)[:, cols] - slot_novelty_mean(slot_index - 4, slot_index)[:, cols]).sum(1)
        local_novelty = np.stack([novelty[np.clip(8 * bar_index + slot_offset, 0, slot_count)] for slot_offset in range(-4, 5)], 1)
        feature_blocks += [local_novelty, local_novelty.argmax(1)[:, None] - 4.0, local_novelty.max(1)[:, None]]
    return np.concatenate(feature_blocks, 1).astype(np.float32)


# ---- per-bar features (from bars.py)
def drop_ends(track):
    hot_cues=track['hot'];memory_cues=track['mem'];first_drop_start=hot_cues[5];second_drop_start=hot_cues.get(9)
    def end(start_bar,limit_bar):
        for memory_bar in memory_cues:
            offset_bars=memory_bar-start_bar
            if offset_bars>=16-0.05 and abs(offset_bars/8-round(offset_bars/8))<0.02 and (limit_bar is None or memory_bar<=limit_bar+0.05): return memory_bar
        return None
    first_drop_end=end(first_drop_start,second_drop_start-8 if second_drop_start is not None else None)
    second_drop_end=end(second_drop_start,None) if second_drop_start is not None else None
    return first_drop_start,first_drop_end,second_drop_start,second_drop_end


def bar_level_features(track):
    beat_energy_table=track['BE'];phase_offset=track['phase']
    bar_count=(len(beat_energy_table)-phase_offset)//4
    bar_cube=beat_energy_table[phase_offset:phase_offset+bar_count*4].reshape(bar_count,4,4,3)  # bar,beat,band,win
    bar_mean=bar_cube.mean(1).reshape(bar_count,12)
    bar_std=bar_cube.std(1).reshape(bar_count,12)
    first_beat_deviation=bar_cube[:,0].reshape(bar_count,12)-bar_mean
    base=np.concatenate([bar_mean,bar_std,first_beat_deviation],1)
    log_levels=np.log1p(bar_mean)
    def shift_rows(values,shift):
        shifted=np.zeros_like(values);
        if shift>0: shifted[shift:]=values[:-shift]; shifted[:shift]=values[:1]
        else: shifted[:shift]=values[-shift:]; shifted[shift:]=values[-1:]
        return shifted
    def window_average(values,first_offset,last_offset): # mean of the bars from (bar + first_offset) to (bar + last_offset)
        cumulative_sums=np.cumsum(np.vstack([np.zeros((1,values.shape[1])),values]),0);row_count=len(values);row_index=np.arange(row_count)
        window_start=np.clip(row_index+first_offset,0,row_count);window_end=np.clip(row_index+last_offset+1,0,row_count);window_end=np.maximum(window_end,window_start+1);window_start=np.minimum(window_start,row_count-1);window_end=np.minimum(window_end,row_count)
        return (cumulative_sums[window_end]-cumulative_sums[window_start])/np.maximum(window_end-window_start,1)[:,None]
    feature_blocks=[base,log_levels-window_average(log_levels,-4,-1),window_average(log_levels,0,3)-window_average(log_levels,-4,-1),window_average(log_levels,0,7)-window_average(log_levels,-8,-1),window_average(log_levels,0,3)-window_average(log_levels,4,7)]
    feature_blocks+= [window_average(log_levels,-8,-1),window_average(log_levels,0,7),shift_rows(log_levels,1),shift_rows(log_levels,-1)]
    bar_position=np.arange(bar_count)[:,None]
    feature_blocks+=[bar_position,bar_count-bar_position,bar_position/bar_count,(bar_position%8==0)*1.0,(bar_position%16==0)*1.0,np.full((bar_count,1),track['bpm'])]
    return np.concatenate(feature_blocks,1).astype(np.float32),bar_count


# ---- track features (feats6.py)
GENRE_PLAYLIST_NAMES = ['Riddim','Tearout','DnB','2-Step','UKG (4x4 only)','140','Jungle','Liquid Riddim','Dubstep']
CANDIDATE_BAR_OFFSETS = [-8, -4, -1, 0, 1, 4, 8]
BAR_FEATURE_COLUMNS = list(range(0, 12)) + list(range(36, 96)) + list(range(120, 132))

def normalized_bar_features(track):
    beat_energy_table = track['BE']; bar_count = len(beat_energy_table) // 4
    bar_cube = beat_energy_table[:bar_count*4].reshape(bar_count, 4, 4, 3); bar_mean = bar_cube.mean(1).reshape(bar_count, 12)
    log_levels = np.log1p(bar_mean)
    median_levels = np.median(log_levels, 0); level_range = np.percentile(log_levels, 90, 0) - np.percentile(log_levels, 10, 0) + 1e-3
    normalized_levels = (log_levels - median_levels) / level_range
    level_quantiles = np.percentile(log_levels, [20, 50, 80], 0)
    level_masks = np.stack([(log_levels > level_quantiles[2]), (log_levels < level_quantiles[0])], 0).astype(np.float32)
    feature_blocks = [normalized_levels, level_masks[0], level_masks[1]]
    cumulative_sums = np.cumsum(np.vstack([np.zeros((1, 12)), normalized_levels]), 0)
    def window_mean(start_indices, end_indices):
        start_indices = np.clip(start_indices, 0, bar_count); end_indices = np.clip(end_indices, 0, bar_count); window_is_valid = end_indices > start_indices
        return np.where(window_is_valid[:, None], (cumulative_sums[np.maximum(end_indices, start_indices+1).clip(0, bar_count)] - cumulative_sums[start_indices]) / np.maximum(end_indices - start_indices, 1)[:, None], 0)
    bar_index = np.arange(bar_count)
    for window_bars in (2, 4, 8, 16):
        before_mean = window_mean(bar_index-window_bars, bar_index); after_mean = window_mean(bar_index, bar_index+window_bars)
        feature_blocks += [after_mean-before_mean, np.linalg.norm(after_mean-before_mean, axis=1, keepdims=True)]
    high_level_share = level_masks[0][:, [0, 1]].mean(1, keepdims=True); cumulative_high = np.cumsum(np.vstack([np.zeros((1, 1)), high_level_share]), 0)
    for window_bars in (4, 8, 16):
        feature_blocks += [(cumulative_high[np.clip(bar_index+window_bars, 0, bar_count)] - cumulative_high[bar_index]) / window_bars, (cumulative_high[bar_index] - cumulative_high[np.clip(bar_index-window_bars, 0, bar_count)]) / window_bars]
    onset_levels = bar_cube[:, :, :, 1].reshape(bar_count, 16); feature_blocks.append(onset_levels.mean(1, keepdims=True) / (onset_levels.mean() + 1e-6))
    # END features (new): level fall after bar vs level inside previous 16 bars (low/mid band cols 0..5 of Z)
    low_normalized = normalized_levels[:, :6].mean(1, keepdims=True); cumulative_low = np.cumsum(np.vstack([np.zeros((1, 1)), low_normalized]), 0)
    def low_window_mean(start_indices, end_indices):
        start_indices = np.clip(start_indices, 0, bar_count); end_indices = np.clip(end_indices, 0, bar_count); window_lengths = np.maximum(end_indices - start_indices, 1)[:, None]
        return (cumulative_low[end_indices] - cumulative_low[start_indices]) / window_lengths
    for window_bars in (4, 8, 16):
        feature_blocks += [low_window_mean(bar_index, bar_index+window_bars) - low_window_mean(bar_index-16, bar_index), low_window_mean(bar_index, bar_index+window_bars) - low_window_mean(bar_index-window_bars, bar_index)]
    return np.concatenate(feature_blocks, 1).astype(np.float32)


def phase_alignment_features(track):
    beat_energy_table = track['BE']; bar_count = len(beat_energy_table) // 4
    bar_cube = beat_energy_table[:bar_count*4].reshape(bar_count, 4, 4, 3); log_levels = np.log1p(bar_cube.mean(1).reshape(bar_count, 12))
    normalized_levels = (log_levels - np.median(log_levels, 0)) / (np.percentile(log_levels, 90, 0) - np.percentile(log_levels, 10, 0) + 1e-3)
    cumulative_sums = np.cumsum(np.vstack([np.zeros((1, 12)), normalized_levels]), 0)
    def window_mean(start_indices, end_indices):
        start_indices = np.clip(start_indices, 0, bar_count); end_indices = np.clip(end_indices, 0, bar_count); return (cumulative_sums[end_indices] - cumulative_sums[start_indices]) / np.maximum(end_indices - start_indices, 1)[:, None]
    bar_index = np.arange(bar_count + 1); feature_columns = []
    for window_bars in (4, 8):
        novelty = np.linalg.norm(window_mean(bar_index, bar_index + window_bars) - window_mean(bar_index - window_bars, bar_index), axis=1); novelty[:2] = 0
        total_novelty = novelty.sum() + 1e-6
        for period in (8, 4):
            phase_share = np.array([novelty[phase_index::period].sum() for phase_index in range(period)]) / total_novelty
            phase_slot = bar_index % period; feature_columns += [phase_share[phase_slot], phase_share[phase_slot] - phase_share.max(), (phase_share[phase_slot] >= phase_share.max() - 1e-9).astype(float), (phase_share[None, :] > phase_share[phase_slot][:, None]).sum(1)]
        local_novelty = np.stack([novelty[np.clip(bar_index + bar_offset, 0, bar_count)] for bar_offset in range(-3, 4)], 1)
        feature_columns += [novelty / (local_novelty.max(1) + 1e-6), local_novelty.argmax(1) - 3.0, novelty / (novelty.max() + 1e-6)]
    low_active = (normalized_levels[:, :6].mean(1) > 0.0); first_active_bar = int(np.argmax(low_active)) if low_active.any() else 0
    bars_since_first_active = bar_index - first_active_bar; feature_columns += [bars_since_first_active % 8 == 0, bars_since_first_active % 4 == 0, bars_since_first_active % 16 == 0, np.full(bar_count + 1, first_active_bar % 8), np.full(bar_count + 1, first_active_bar)]
    return np.stack(feature_columns, 1).astype(np.float32)



def track_features(track):
    bar_features, bar_count = bar_level_features(track); normalized_features = normalized_bar_features(track); bar_features = np.concatenate([bar_features[:, BAR_FEATURE_COLUMNS], normalized_features], 1)
    candidates = np.arange(0, bar_count + 1)
    features = np.concatenate([bar_features[np.clip(candidates + offset, 0, bar_count - 1)] for offset in CANDIDATE_BAR_OFFSETS], 1)
    genre_flags = np.array([genre in track['pl'] for genre in GENRE_PLAYLIST_NAMES], np.float32)
    features = np.concatenate([features, beat_resolved_features(track), eighth_note_band_features(track), np.tile(genre_flags, (len(candidates), 1)), phase_alignment_features(track),
                        np.stack([candidates % 4 == 0, candidates % 8 == 0, candidates % 16 == 0, bar_count - candidates, candidates, candidates / bar_count, np.full(len(candidates), track['bpm'])], 1)], 1).astype(np.float32)
    return candidates, bar_count, features
