"""Candidate dataset for phase selection. One row per candidate peak per track; label = within TOL_MS of the stored-grid phase."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json,pickle,sys,numpy as np,pandas as pd
CALIBRATION_MS=4.0       # mid-band fold peak is this much later than the visible attack
TOLERANCE_MS=8.0
event_cache_folder=str(paths.EVENT_CACHES / 'old961') + '/'
def wrap(value,period): return (value+period/2)%period-period/2

def fold_novelty(novelty,frame_seconds,period_seconds,anchor_seconds,start_seconds,end_seconds,bin_count):
    length=len(novelty); frame_times=np.arange(length)*frame_seconds; mask=(frame_times>=start_seconds)&(frame_times<=end_seconds)
    phase_bins=(((frame_times[mask]-anchor_seconds)%period_seconds)/period_seconds*bin_count).astype(int)%bin_count
    return np.bincount(phase_bins,weights=novelty[mask].astype(np.float64),minlength=bin_count)

def smooth_circular(histogram,window):
    half_window=window//2; padded=np.concatenate([histogram[-half_window:],histogram,histogram[:half_window]]); return np.convolve(padded,np.ones(window)/window,'valid')

def find_peaks(histogram,threshold):
    length=len(histogram); out=[]
    for bin_index in range(length):
        level=histogram[bin_index]
        if level>=threshold and level>=histogram[bin_index-1] and level>histogram[(bin_index+1)%length]: out.append(bin_index)
    return out

def local_maximum(histogram,bin_index,half_width):
    length=len(histogram); indices=[(bin_index+offset)%length for offset in range(-half_width,half_width+1)]; return float(histogram[indices].max())

def event_share(events,section_mask,period_seconds,anchor_seconds,event_class,center_ms,tolerance_ms):
    times=events['t'][section_mask]; classes=events['c'][section_mask]; weights=events['w'][section_mask]
    is_class=classes==event_class
    if not is_class.any(): return 0.0,0
    residuals_ms=wrap(times[is_class]-anchor_seconds-center_ms/1000.0,period_seconds)*1000
    within_tolerance=np.abs(residuals_ms)<=tolerance_ms
    return float(weights[is_class][within_tolerance].sum()/weights[is_class].sum()),int(within_tolerance.sum())

def build(report_path,rows_path,fs_keep=None,max_candidates=8):
    report=json.load(open(report_path)); records=[]
    scope={row['id']:row for row in json.load(open(rows_path))}
    for report_row in report:
        if report_row['id'] not in scope or not report_row['sections']: continue
        section=max(report_row['sections'],key=lambda section:section['end_s']-section['start_s'])
        if 'offset_mod_beat_ms' not in section: continue
        period_seconds=60.0/section['bpm']; anchor_seconds=section['anchor_ms']/1000.0
        true_offset_ms=section['err_vs_model_ms']      # truth minus ours, ms
        events=pickle.load(open(event_cache_folder+report_row['id']+'.ff2.pkl','rb'))
        if 'nov_low' not in events: continue
        bin_count=int(round(period_seconds*1000))
        mid_fold=smooth_circular(fold_novelty(events['nov_mid'],events['nov_dt'],period_seconds,anchor_seconds,section['start_s'],section['end_s'],bin_count),3)
        low_fold=smooth_circular(fold_novelty(events['nov_low'],events['nov_dt'],period_seconds,anchor_seconds,section['start_s'],section['end_s'],bin_count),3)
        if mid_fold.max()<=0 or low_fold.max()<=0: continue
        mid_peak=mid_fold.max(); low_peak=low_fold.max()
        mid_normalized,low_normalized=mid_fold/mid_peak,low_fold/low_peak
        candidate_bins=set(find_peaks(mid_normalized,0.25))|set(find_peaks(low_normalized,0.45))|{0}
        # sort by combined height, keep top maxc (ours always)
        combined_height=lambda bin_index:mid_normalized[bin_index]+0.5*low_normalized[bin_index]
        sorted_candidates=sorted(candidate_bins,key=lambda bin_index:-combined_height(bin_index))
        keep=[bin_index for bin_index in sorted_candidates if bin_index!=0][:max_candidates-1]+[0]
        # merge near duplicates (<5 bins)
        kept_bins=[]
        for bin_index in sorted(keep,key=lambda bin_index:(bin_index!=0,-combined_height(bin_index))):
            if all(min(abs(bin_index-other_bin),bin_count-abs(bin_index-other_bin))>=5 for other_bin in kept_bins): kept_bins.append(bin_index)
        section_mask=(events['t']>=section['start_s'])&(events['t']<=section['end_s'])
        kick_mask=events['c'][section_mask]==0
        ours_kick_share=event_share(events,section_mask,period_seconds,anchor_seconds,0,0,4); ours_snare_share=event_share(events,section_mask,period_seconds,anchor_seconds,1,0,4)
        mid_median=float(np.median(mid_normalized)); low_median=float(np.median(low_normalized))
        rank_order=sorted(kept_bins,key=lambda bin_index:-mid_normalized[bin_index])
        for bin_index in kept_bins:
            relative_position_ms=wrap(bin_index/bin_count*period_seconds,period_seconds)*1000      # candidate phase relative to ours, ms
            calibrated_position_ms=relative_position_ms-CALIBRATION_MS               # calibrated visible-attack position
            opposite_bin=(bin_index+bin_count//2)%bin_count
            kick_share,kick_count=event_share(events,section_mask,period_seconds,anchor_seconds,0,calibrated_position_ms,4); snare_share,snare_count=event_share(events,section_mask,period_seconds,anchor_seconds,1,calibrated_position_ms,4)
            kick_share_shifted,_=event_share(events,section_mask,period_seconds,anchor_seconds,0,calibrated_position_ms+period_seconds*500,4); snare_share_shifted,_=event_share(events,section_mask,period_seconds,anchor_seconds,1,calibrated_position_ms+period_seconds*500,4)
            records.append(dict(id=report_row['id'],pos=calibrated_position_ms,is_ours=int(abs(calibrated_position_ms)<=6),dist=abs(calibrated_position_ms),
                 fm=local_maximum(mid_normalized,bin_index,2),fl=local_maximum(low_normalized,bin_index,3),fm_half=local_maximum(mid_normalized,opposite_bin,3),fl_half=local_maximum(low_normalized,opposite_bin,3),
                 fm_rank=rank_order.index(bin_index),fm_sharp=float(mid_normalized[bin_index]/(np.mean(mid_normalized[[(bin_index+offset)%bin_count for offset in range(-25,26,5) if abs(offset)>=10]])+1e-9)),
                 fl_sharp=float(low_normalized[bin_index]/(np.mean(low_normalized[[(bin_index+offset)%bin_count for offset in range(-25,26,5) if abs(offset)>=10]])+1e-9)),
                 kshare=kick_share,sshare=snare_share,kn=np.log1p(kick_count),sn=np.log1p(snare_count),kshare_half=kick_share_shifted,sshare_half=snare_share_shifted,
                 ours_k=ours_kick_share[0],ours_s=ours_snare_share[0],ours_kn=np.log1p(ours_kick_share[1]),ours_sn=np.log1p(ours_snare_share[1]),
                 T=period_seconds*1000,rms=section['fit_rms_ms'],lnk=np.log1p(report_row['n_kicks']),lns=np.log1p(report_row['n_snares']),
                 mid_strength=float(mid_peak/(np.median(mid_fold)+1e-9)),low_strength=float(low_peak/(np.median(low_fold)+1e-9)),
                 label=int(abs(calibrated_position_ms-true_offset_ms)<=TOLERANCE_MS),dpos=calibrated_position_ms-true_offset_ms,tru=true_offset_ms,cur_err=-true_offset_ms,
                 ncand=len(kept_bins)))
    return pd.DataFrame(records)
if __name__=='__main__':
    candidates=build(str(paths.EVAL_REPORTS / 'eval_report.json'),str(paths.WORK / 'scope.json'))
    candidates.to_pickle(str(paths.WORK / 'phase_cand.pkl')); print(candidates.shape, candidates.id.nunique())
    tracks_with_candidate_label=candidates.groupby('id').label.max(); print('tracks with a positive candidate:',tracks_with_candidate_label.mean(),'  tracks:',len(tracks_with_candidate_label))
    ours_positive_per_track=candidates[candidates.is_ours==1].groupby('id').label.max(); print('ours positive:',ours_positive_per_track.mean())
