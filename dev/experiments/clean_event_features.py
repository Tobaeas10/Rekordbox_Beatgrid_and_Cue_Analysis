import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import numpy as np
WINDOW_RADIUS=25
def _window_statistics(windows):
    window_count=len(windows)
    if window_count<6: return [-99.0]*5+[float(window_count)]
    peak_heights=windows.max(1); peak_offsets=windows.argmax(1)-WINDOW_RADIUS
    pre_beat_level=windows[:,:WINDOW_RADIUS//2].mean(1)
    cleanliness=peak_heights/(pre_beat_level+0.05*peak_heights+1e-9)
    is_loud=peak_heights>=np.median(peak_heights)
    score=np.where(is_loud,cleanliness,-1)
    top_count=max(6,int(0.3*window_count)); selected=np.argsort(-score)[:top_count]
    selected_windows=windows[selected]; selected_heights=peak_heights[selected]
    quarter_peak_offsets=np.array([np.argmax(window>=0.25*peak_height) for window,peak_height in zip(selected_windows,selected_heights)])-WINDOW_RADIUS
    half_peak_offsets=np.array([np.argmax(window>=0.5*peak_height) for window,peak_height in zip(selected_windows,selected_heights)])-WINDOW_RADIUS
    selected_peak_offsets=peak_offsets[selected]
    median_offset=np.median(selected_peak_offsets)
    median_absolute_deviation=np.median(np.abs(selected_peak_offsets-median_offset))
    # agreement of halves in time order
    time_order=np.sort(selected); first_half,second_half=time_order[:len(time_order)//2],time_order[len(time_order)//2:]
    half_difference=np.median(peak_offsets[first_half])-np.median(peak_offsets[second_half])
    return [float(median_offset),float(np.median(quarter_peak_offsets)),float(np.median(half_peak_offsets)),float(median_absolute_deviation),float(half_difference),float(window_count)]
def extra_window_features(events,section):
    period_seconds,anchor_seconds=section['period_s'],section['anchor_s']; half=period_seconds/2; frame_seconds=events['nov_dt']
    half_beat_indices=np.arange(int(np.ceil((section['start_s']-anchor_seconds)/half)),int(np.floor((section['end_s']-anchor_seconds)/half))+1)
    features=[]
    for novelty_key in ('nov_mid','nov_low'):
        novelty=events[novelty_key]; frame_count=len(novelty)
        window_start_frames=np.rint((anchor_seconds+half_beat_indices*half-WINDOW_RADIUS/1000.0)/frame_seconds).astype(int)
        in_range=(window_start_frames>=0)&(window_start_frames+2*WINDOW_RADIUS+1<=frame_count)
        frame_indices=window_start_frames[in_range][:,None]+np.arange(2*WINDOW_RADIUS+1)[None,:]
        windows=novelty[frame_indices] if in_range.any() else np.zeros((0,2*WINDOW_RADIUS+1)); valid_half_beat_indices=half_beat_indices[in_range]
        if len(windows)==0:
            features+=[-99.0]*18; continue
        parity=valid_half_beat_indices%2
        mean_peak_by_parity=[windows[parity==parity_value].max(1).mean() if (parity==parity_value).any() else 0 for parity_value in (0,1)]
        strong_parity=int(np.argmax(mean_peak_by_parity))
        for selection_mask in (parity==strong_parity,parity!=strong_parity,np.ones(len(windows),bool)):
            features+=_window_statistics(windows[selection_mask])
    return np.array(features,float)
