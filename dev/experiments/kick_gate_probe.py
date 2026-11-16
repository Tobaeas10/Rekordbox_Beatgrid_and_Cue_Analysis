import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json,pickle,numpy as np,re,sys
report_rows=json.load(open(str(paths.EVAL_REPORTS / 'eval_report_-_all_songs_wip19.json')))
def kick_offset_mode(event_times,event_weights,anchor_seconds,period_seconds,start_seconds,end_seconds,sigma_seconds=0.004):
    in_section=(event_times>=start_seconds)&(event_times<end_seconds)
    beat_residuals=((event_times[in_section]-anchor_seconds)/period_seconds+0.5)%1-0.5   # in beats
    section_weights=event_weights[in_section]
    candidate_offset_fractions=np.arange(-0.5,0.5,0.0025)
    density=(section_weights[None,:]*np.exp(-0.5*((candidate_offset_fractions[:,None]-beat_residuals[None,:])/(sigma_seconds/period_seconds))**2)).sum(1)
    best_index=int(np.argmax(density)); zero_index=int(np.argmin(np.abs(candidate_offset_fractions)))
    density_near_zero=density[max(0,zero_index-6):zero_index+7].max()
    return candidate_offset_fractions[best_index]*period_seconds*1000,density[best_index],density_near_zero,len(beat_residuals),section_weights.sum()
results=[]
for report_row in report_rows:
    if not report_row['sections'] or not (report_row.get('dry_run') or {}).get('ok'): continue
    section=report_row['sections'][0]
    if section['bpm']>=160: continue
    try: events=pickle.load(open(f"{paths.EVENT_CACHES}/all_songs/{report_row['id']}.ff2.pkl",'rb'))
    except: continue
    period_seconds=60/section['bpm']; anchor_seconds=section['anchor_ms']/1000
    event_times,event_weights,event_classes=events['t'],events['w'],events['c']
    is_kick=event_classes==0
    mode_offset_ms,mode_density,zero_density,kick_count,total_weight=kick_offset_mode(event_times[is_kick],event_weights[is_kick],anchor_seconds,period_seconds,section['start_s'],section['end_s'])
    results.append(dict(id=report_row['id'],title=report_row['title'][:24],med=report_row['dry_run']['diff_median_ms'],k_off=mode_offset_ms,ratio=mode_density/(zero_density+1e-9),nk=kick_count,share=mode_density/(total_weight+1e-9)))
json.dump(results,open(str(paths.WORK / 'kick_gate_probe.json'),'w'))
import collections
triggered=[result for result in results if abs(result['k_off'])>30 and result['ratio']>3]
print(len(results),'trigger',len(triggered))
for result in triggered: print(result['title'],'med',result['med'],'k_off',round(result['k_off'],1),'ratio',round(result['ratio'],1),'nk',result['nk'])
