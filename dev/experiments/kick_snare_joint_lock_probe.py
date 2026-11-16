"""Joint kick+snare lock per candidate: bars where a kick hits at beat q AND a snare hits at beat q+2 (6 ms), and spacing jitter."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,pickle,json,glob,numpy as np,pandas as pd
from dev.experiments.periodicity_candidate_dataset import REPORT_AND_CACHE_SETS
from lib.beatgrid import rankers
candidate_table=pd.read_pickle(str(paths.WORK / 'periodicity_candidates.pkl'));track_info=pd.read_pickle(str(paths.WORK / 'periodicity_candidates_info.pkl')).set_index('id')
report_and_cache_by_id={}
for report_path,cache_folder in REPORT_AND_CACHE_SETS:
    for report_row in json.load(open(report_path)): report_and_cache_by_id.setdefault(report_row['id'],(report_row,cache_folder))
def joint_lock_fractions(events,anchor_seconds,period_seconds,start_seconds,end_seconds):
    in_section=(events['t']>=start_seconds)&(events['t']<=end_seconds);event_times,event_classes=events['t'][in_section],events['c'][in_section]
    lock_fractions=[]
    for snare_lag_beats in (2,1):  # snare lag beats after kick
        kick_beats=np.round((event_times[event_classes==0]-anchor_seconds)/period_seconds);kick_errors_ms=(event_times[event_classes==0]-anchor_seconds-kick_beats*period_seconds)*1000;snare_beats=np.round((event_times[event_classes==1]-anchor_seconds)/period_seconds);snare_errors_ms=(event_times[event_classes==1]-anchor_seconds-snare_beats*period_seconds)*1000
        kick_beat_set={int(beat) for beat,error_ms in zip(kick_beats,kick_errors_ms) if abs(error_ms)<=6};snare_beat_set={int(beat) for beat,error_ms in zip(snare_beats,snare_errors_ms) if abs(error_ms)<=6}
        beat_count=max((end_seconds-start_seconds)/period_seconds,8);joint_beats=[beat for beat in kick_beat_set if (beat+snare_lag_beats) in snare_beat_set and beat%4 in (0,1,2,3)]
        bars_by_position={}
        for beat in joint_beats: bars_by_position.setdefault(beat%4,set()).add(beat//4)
        lock_fractions.append(max((len(bar_set) for bar_set in bars_by_position.values()),default=0)/(beat_count/4))
    return lock_fractions
results=[];track_ids=list(track_info.index)
for track_id in track_ids:
    report_row,cache_folder=report_and_cache_by_id[track_id];section=report_row['sections'][0];events=pickle.load(open(cache_folder+track_id+'.ff2.pkl','rb'));period_seconds=60/section['bpm'];anchor_seconds=section['anchor_ms']/1000
    candidates=candidate_table[candidate_table.id==track_id].reset_index(drop=True)
    lock_matrix=np.array([joint_lock_fractions(events,anchor_seconds+position_ms/1000,period_seconds,section['start_s'],section['end_s']) for position_ms in candidates.pos])
    results.append((track_id,lock_matrix,candidates.label.values))
gross_track_ids={track_id for track_id in track_ids if abs(track_info.tru[track_id])>20}
for lag_index,lag_name in((0,'lag2 (kick 1 -> snare 3)'),(1,'lag1')):
    normal_rate=np.mean([result[1][:,lag_index].argmax()==0 or result[1][:,lag_index].max()==result[1][0,lag_index] for result in results if result[0] not in gross_track_ids])
    gr=[(result[1][:,lag_index].argmax()==np.argmax(result[2])) for result in results if result[0] in gross_track_ids and result[2].max()==1]
    print(lag_name,'ours argmax on normal',round(normal_rate,3),'| truth argmax on gross',sum(gr),'/',len(gr))
