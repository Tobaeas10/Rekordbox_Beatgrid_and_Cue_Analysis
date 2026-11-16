"""Fakeout-end predictor (fk_model: HGB over per-beat features of the first 104 beats of a drop). predict_fakeout_end(t, drop_bar) -> (bar or None, prob)."""
import numpy as np

from lib import paths

"""fkfeats.py: fakeout-end candidate features. fakeout_candidate_features(t, S) -> (cand_beats, F): candidates are beat offsets b=2..MAX_BEATS_AFTER_DROP after drop start S (bar).
Series at beat resolution (BX groups level/flux/centroid/rolloff + BE 4 bands x3 windows), 8th resolution for the BX groups near the candidate."""
MAX_BEATS_AFTER_DROP=104
GENRE_PLAYLIST_NAMES=['Riddim','Tearout','DnB','2-Step','UKG (4x4 only)','140','Jungle','Liquid Riddim','Dubstep']
BAND_GROUPS=[[0,1],[2,3],[4,5],[6,7],[8],[9]]
def series(track):
    band_energy_table=track['BX'].astype(np.float32); beat_energy_table=track['BE']; beat_count=min(len(band_energy_table),len(beat_energy_table))
    group_levels=np.stack([band_energy_table[:beat_count,:,band_group].mean((1,2)) for band_group in BAND_GROUPS],1)
    group_flux=np.stack([band_energy_table[:beat_count,:,[10+i for i in band_group]].mean((1,2)) for band_group in BAND_GROUPS],1)
    centroid_rolloff=band_energy_table[:beat_count,:,20:22].mean(1)
    log_beat_energy=np.log1p(beat_energy_table[:beat_count].reshape(beat_count,12))
    beat_series=np.concatenate([group_levels,group_flux,centroid_rolloff,log_beat_energy],1)
    eighth_slots=band_energy_table[:beat_count].reshape(beat_count*2,22)
    eighth_series=np.concatenate([np.stack([eighth_slots[:,band_group].mean(1) for band_group in BAND_GROUPS],1),np.stack([eighth_slots[:,[10+i for i in band_group]].mean(1) for band_group in BAND_GROUPS],1),eighth_slots[:,20:22]],1)
    return beat_series,eighth_series
def cumulative_sum_table(values): return np.vstack([np.zeros((1,values.shape[1]),np.float64),np.cumsum(values,0,dtype=np.float64)])
def fakeout_candidate_features(track,drop_start_bar):
    beat_series,eighth_series=series(track); beat_count=len(beat_series); cumulative_series=cumulative_sum_table(beat_series); cumulative_eighth=cumulative_sum_table(eighth_series)
    start_beat=int(round(drop_start_bar*4)); beat_offsets=np.arange(1,MAX_BEATS_AFTER_DROP+1); candidate_beats=start_beat+beat_offsets
    valid=candidate_beats+8<=beat_count; beat_offsets=beat_offsets[valid]; candidate_beats=candidate_beats[valid]
    def window_mean(start_indices,end_indices,cumulative_table=cumulative_series): 
        start_indices=np.clip(start_indices,0,beat_count if cumulative_table is cumulative_series else 2*beat_count); end_indices=np.clip(end_indices,0,beat_count if cumulative_table is cumulative_series else 2*beat_count); return (cumulative_table[end_indices]-cumulative_table[start_indices])/np.maximum(end_indices-start_indices,1)[:,None]
    feature_blocks=[]
    for window_length in (1,2,4,8): feature_blocks.append(window_mean(candidate_beats,candidate_beats+window_length)-window_mean(candidate_beats-window_length,candidate_beats))
    for window_length in (2,4): feature_blocks.append(window_mean(2*candidate_beats,2*candidate_beats+window_length,cumulative_eighth)-window_mean(2*candidate_beats-window_length,2*candidate_beats,cumulative_eighth))
    pre_drop_mean=window_mean(np.full_like(candidate_beats,start_beat),candidate_beats); 
    for window_length in (16,32): feature_blocks.append(window_mean(candidate_beats,candidate_beats+window_length)-pre_drop_mean)
    feature_blocks.append(window_mean(candidate_beats,candidate_beats+32)-window_mean(candidate_beats+32,candidate_beats+64))
    first_beats_mean=window_mean(np.full_like(candidate_beats,start_beat),np.full_like(candidate_beats,start_beat+4)); reference_mean=window_mean(np.full_like(candidate_beats,start_beat+4),np.full_like(candidate_beats,start_beat+32)); feature_blocks.append(first_beats_mean-reference_mean)
    feature_blocks.append(window_mean(np.full_like(candidate_beats,start_beat),np.full_like(candidate_beats,start_beat+16))-window_mean(np.full_like(candidate_beats,start_beat+32),np.full_like(candidate_beats,start_beat+64)))
    feature_blocks.append(window_mean(np.full_like(candidate_beats,start_beat+32),np.full_like(candidate_beats,start_beat+64))-window_mean(np.full_like(candidate_beats,start_beat-32),np.full_like(candidate_beats,start_beat)))
    # neighbour-beat step features (lets the model learn systematic lag of the user's mc vs the audio step)
    for beat_shift in (-3,-2,-1,1,2,3):
        feature_blocks.append(window_mean(candidate_beats+beat_shift,candidate_beats+beat_shift+2)-window_mean(candidate_beats+beat_shift-2,candidate_beats+beat_shift))
    # one-beat step norm profile over +-4 beats (where is the biggest change?)
    step_norm_profile=np.stack([np.linalg.norm(window_mean(candidate_beats+beat_shift,candidate_beats+beat_shift+1)[:,:12]-window_mean(candidate_beats+beat_shift-1,candidate_beats+beat_shift)[:,:12],axis=1) for beat_shift in range(-4,5)],1)
    feature_blocks.append(step_norm_profile);feature_blocks.append(step_norm_profile.argmax(1)[:,None]-4.0)
    # matched-length changepoint, normalised step norms with within-window ranking, pre-region spread
    series_scale=beat_series.std(0)+1e-3; normalized_series=beat_series/series_scale; cumulative_normalized=cumulative_sum_table(normalized_series); cumulative_squares=cumulative_sum_table(normalized_series**2)
    def normalized_window_mean(start_indices,end_indices): start_indices=np.clip(start_indices,0,beat_count); end_indices=np.clip(end_indices,0,beat_count); return (cumulative_normalized[end_indices]-cumulative_normalized[start_indices])/np.maximum(end_indices-start_indices,1)[:,None]
    matched_offsets=np.clip(beat_offsets,2,32); feature_blocks.append(window_mean(candidate_beats,candidate_beats+matched_offsets)-pre_drop_mean)
    start_beats=np.full_like(candidate_beats,start_beat)
    for window_length in (1,2,4,8):
        step=normalized_window_mean(candidate_beats,candidate_beats+window_length)-normalized_window_mean(candidate_beats-window_length,candidate_beats); step_norm=np.linalg.norm(step,axis=1); low_step_norm=np.linalg.norm(step[:,:6],axis=1)
        step_rank=(step_norm[:,None]<step_norm[None,:]).sum(1)/len(step_norm)
        feature_blocks.append(np.stack([step_norm,step_norm/(step_norm.max()+1e-6),step_rank,low_step_norm,low_step_norm/(low_step_norm.max()+1e-6)],1))
    mean_square=(cumulative_squares[np.clip(candidate_beats,0,beat_count)]-cumulative_squares[start_beat])/np.maximum(candidate_beats-start_beat,1)[:,None]; mean_before=normalized_window_mean(start_beats,candidate_beats); feature_blocks.append(np.sqrt(np.maximum(mean_square-mean_before**2,0))[:,:12])
    feature_blocks.append(normalized_window_mean(candidate_beats,candidate_beats+matched_offsets)-normalized_window_mean(start_beats,candidate_beats))
    # pattern-presence flags per beat (kick, snare, hat = flux; sub = level), track-adaptive threshold
    band_energy_float=track['BX'][:beat_count].astype(np.float32)
    def flag_above_threshold(values):
        lower_level,upper_level=np.percentile(values,[25,95]); return (values>lower_level+0.4*(upper_level-lower_level)).astype(np.float64)
    kick_flag=flag_above_threshold(band_energy_float[:,:,[10,11]].mean(2).max(1)); snare_flag=flag_above_threshold(band_energy_float[:,:,[14,15,16]].mean(2).max(1)); hat_flag=flag_above_threshold(band_energy_float[:,:,[18,19]].mean(2).max(1)); sub_flag=flag_above_threshold(band_energy_float[:,:,[0,1]].mean(2).mean(1))
    flag_table=np.stack([kick_flag,snare_flag,hat_flag,sub_flag,kick_flag*snare_flag*hat_flag],1); cumulative_flags=cumulative_sum_table(flag_table)
    def flag_window_mean(start_indices,end_indices): start_indices=np.clip(start_indices,0,beat_count); end_indices=np.clip(end_indices,0,beat_count); return (cumulative_flags[end_indices]-cumulative_flags[start_indices])/np.maximum(end_indices-start_indices,1)[:,None]
    flag_features=[]
    for window_start_offset,window_end_offset in ((0,4),(0,8),(0,16),(-4,0),(-8,0),(-16,0)): flag_features.append(flag_window_mean(candidate_beats+window_start_offset,candidate_beats+window_end_offset))
    flag_features.append(flag_window_mean(candidate_beats,candidate_beats+16)-flag_window_mean(candidate_beats-16,candidate_beats)); flag_features.append(flag_window_mean(candidate_beats,candidate_beats+4)-flag_window_mean(candidate_beats-4,candidate_beats))
    # per-bar coverage: fraction of the 4 bars after/before with >=1 flag, and flag on beat 1 of bars after
    def bar_coverage(offset_beats):
        coverage=np.zeros((len(candidate_beats),5))
        for bar_number in range(4):
            bar_start_beats=candidate_beats+offset_beats+4*bar_number; segment=np.stack([(flag_table[np.clip(bar_start_beats+j,0,beat_count-1)]) for j in range(4)],0); coverage+= (segment.max(0))
        return coverage/4
    flag_features.append(bar_coverage(0));flag_features.append(bar_coverage(-16));flag_features.append(bar_coverage(0)-bar_coverage(-16))
    flag_features.append(flag_table[np.clip(candidate_beats,0,beat_count-1)]); flag_features.append(np.stack([flag_table[np.clip(candidate_beats+4*bar_number,0,beat_count-1)] for bar_number in range(1,4)],0).mean(0))
    # run lengths (in 4-beat bars aligned at the candidate) of consecutive bars with >=1 / >=2 flags, forward and backward (cap 16)
    def flag_run_lengths(direction,minimum_flag_count):
        run_lengths=np.zeros((len(candidate_beats),5)); alive=np.ones((len(candidate_beats),5),bool)
        for bar_number in range(16):
            bar_start_beats=candidate_beats+4*bar_number if direction>0 else candidate_beats-4*(bar_number+1)
            flag_count=np.zeros((len(candidate_beats),5))
            for j in range(4): flag_count+=flag_table[np.clip(bar_start_beats+j,0,beat_count-1)]
            alive&=(flag_count>=minimum_flag_count); run_lengths+=alive
        return run_lengths
    flag_features+= [flag_run_lengths(1,1),flag_run_lengths(-1,1),flag_run_lengths(1,2),flag_run_lengths(-1,2)]
    feature_blocks+=flag_features
    bar_offsets=beat_offsets/4.0
    position_features=[beat_offsets,bar_offsets,beat_offsets%4,beat_offsets%2]+[bar_offsets-2**power for power in range(0,5)]
    nearest_power_distance=np.array([min(abs(candidate_bar_offset-2**power) for power in range(0,5)) for candidate_bar_offset in bar_offsets]); position_features+=[nearest_power_distance]
    position_features=np.stack(position_features,1)
    genre_flags=np.tile(np.array([genre in track['pl'] for genre in GENRE_PLAYLIST_NAMES],np.float32),(len(beat_offsets),1))
    feature_blocks+= [position_features,genre_flags,np.tile([track['bpm'],drop_start_bar%8,drop_start_bar%4],(len(beat_offsets),1))]
    return beat_offsets,np.concatenate(feature_blocks,1).astype(np.float32)

import pickle
_FAKEOUT_MODEL=None
def predict_fakeout_end(track,drop_start_bar,is_second_drop=False,threshold=0.4):
    global _FAKEOUT_MODEL
    if _FAKEOUT_MODEL is None: _FAKEOUT_MODEL=pickle.load(open(paths.MODELS_CUES / 'fakeout_end_model.pkl','rb'))
    beat_offsets,features=fakeout_candidate_features(track,drop_start_bar)
    if len(beat_offsets)<20: return None,0.0
    probabilities=_FAKEOUT_MODEL.predict_proba(np.concatenate([features,np.full((len(beat_offsets),1),1 if is_second_drop else 0)],1))[:,1]
    best_index=int(np.argmax(probabilities));return (drop_start_bar+beat_offsets[best_index]/4.0 if probabilities[best_index]>threshold else None),float(probabilities[best_index])
