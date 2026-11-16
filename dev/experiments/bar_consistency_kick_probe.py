import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json,pickle,numpy as np,sys
report_rows=json.load(open(str(paths.EVAL_REPORTS / 'eval_report_-_all_songs_wip19.json')))
def consistency_mode_offset(event_times,anchor_seconds,period_seconds,start_seconds,end_seconds,tolerance=0.006):
    in_section=(event_times>=start_seconds)&(event_times<end_seconds); event_times=event_times[in_section]
    if len(event_times)<20: return None
    bar_seconds=4*period_seconds
    beats_from_anchor=((event_times-anchor_seconds)/period_seconds)            # beats from anchor
    bar_index=np.floor(beats_from_anchor/4).astype(int); beat_in_bar=np.floor(beats_from_anchor%4+1e-9).astype(int)%4; fraction_to_nearest_beat=((beats_from_anchor%1)+0.5)%1-0.5  # offset to nearest beat in beats
    nearest_beat_in_bar=np.round(beats_from_anchor%4).astype(int)%4
    nearest_bar_index=np.floor((beats_from_anchor-nearest_beat_in_bar+0.5*0)/4+0.5).astype(int)
    candidate_offset_fractions=np.arange(-0.5,0.5,0.004)
    bar_count=max(1,len(np.unique(nearest_bar_index)))
    best=(-1,0,0)
    offset_to_nearest_beat=((beats_from_anchor-np.round(beats_from_anchor))*period_seconds)        # seconds offset to nearest beat
    rounded_bar_index=np.floor(np.round(beats_from_anchor)/4).astype(int); position_in_bar=(np.round(beats_from_anchor).astype(int))%4
    for candidate_offset_seconds in candidate_offset_fractions*period_seconds:
        near_offset=np.abs(offset_to_nearest_beat-candidate_offset_seconds)<tolerance
        if not near_offset.any(): continue
        for beat_position in range(4):
            near_offset_at_position=near_offset&(position_in_bar==beat_position)
            bar_count_at_position=len(np.unique(rounded_bar_index[near_offset_at_position]))
            if bar_count_at_position>best[0]: best=(bar_count_at_position,candidate_offset_seconds*1000,beat_position)
    return best[1],best[0]/(end_seconds-start_seconds)*bar_seconds
results=[]
for report_row in report_rows:
    if not report_row['sections'] or not (report_row.get('dry_run') or {}).get('ok'): continue
    section=report_row['sections'][0]
    if section['bpm']>=160 or len(report_row['sections'])>1: continue
    try: events=pickle.load(open(f"{paths.EVENT_CACHES}/all_songs/{report_row['id']}.ff2.pkl",'rb'))
    except: continue
    period_seconds=60/section['bpm']; anchor_seconds=section['anchor_ms']/1000
    event_times,event_classes=events['t'],events['c']
    mode_result=consistency_mode_offset(event_times[event_classes==0],anchor_seconds,period_seconds,section['start_s'],section['end_s'])
    if mode_result is None: continue
    results.append((report_row['title'][:22],report_row['dry_run']['diff_median_ms'],mode_result[0],mode_result[1]))
json.dump(results,open(str(paths.WORK / 'bar_consistency_kick_probe.json'),'w'))
bad_results=[result for result in results if abs(result[1])>5]; good_results=[result for result in results if abs(result[1])<=5]
agrees_with_truth=lambda result: abs(result[2]-(-result[1]))<=20
print('n',len(results),'bad',len(bad_results),'| kick-consistency mode within 20ms of truth: bad',sum(map(agrees_with_truth,bad_results)),'good',sum(map(agrees_with_truth,good_results)),'of',len(good_results))
print('good tracks where mode is >20ms from truth (would be wrong rough align):',len(good_results)-sum(map(agrees_with_truth,good_results)))
for result in bad_results[:30]: print(result[0],result[1],round(result[2],1),round(result[3],2))
