import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,json,pickle,numpy as np
event_cache_folder=str(paths.EVENT_CACHES / 'old961') + '/'
eval_report=json.load(open(str(paths.EVAL_REPORTS / 'eval_report.json')))
rows=[]
for report_row in eval_report:
    events=pickle.load(open(event_cache_folder+f"{report_row['id']}.ff2.pkl",'rb'))
    section=report_row['sections'][0]; period_ms=60000/section['bpm']
    median_difference_ms=report_row['dry_run']['diff_median_ms']
    wrong_parity = abs(abs(median_difference_ms)-period_ms/2)<30
    # final built anchor; true lattice = built if ok, else built+T/2
    model_offset_ms=report_row['offset_model']['ms']
    anchor_seconds=section['anchor_ms']/1000+(0 if not wrong_parity else period_ms/2000)
    event_times,event_weights,event_classes=events['t'],events['w'],events['c']
    in_section=(event_times>=section['start_s'])&(event_times<section['end_s'])
    def hist(anchor_candidate_seconds):
        beat_phase=((event_times[in_section]-anchor_candidate_seconds)/(period_ms/1000))
        beat_index=np.round(beat_phase); residual_ms=(beat_phase-beat_index)*period_ms  # ms residual
        on_beat=np.abs(residual_ms)<12
        parity=(beat_index.astype(int))%2
        weights_by_class={}
        for class_name,class_id in (('K',0),('S',1)):
            for parity_value in (0,1):
                weights_by_class[f'{class_name}{parity_value}']=float(event_weights[in_section][on_beat&(event_classes[in_section]==class_id)&(parity==parity_value)].sum())
        return weights_by_class
    histograms_true=hist(anchor_seconds); histograms_flipped=hist(anchor_seconds+period_ms/2000)
    rows.append((report_row['artist'][:16]+' - '+report_row['title'][:20],wrong_parity,section['parity']['method'],section['parity']['ratio'],histograms_true,histograms_flipped,abs(median_difference_ms)<20))
json.dump([(row[0],row[1],row[2],row[3],row[4],row[5]) for row in rows],open(str(paths.WORK / 'lattice_parity_study.json'),'w'))
for row in rows:
    if row[1]:
        print(row[0],row[2],row[3]); print('  true ',{k:round(v) for k,v in row[4].items()}); print('  false',{k:round(v) for k,v in row[5].items()})
good_rows=[row for row in rows if not row[1] and row[5]]
print(len(good_rows))
import statistics
def tot(histogram): return histogram['K0']+histogram['K1']+histogram['S0']+histogram['S1']
on_off_ratios=[tot(row[4])/(tot(row[5])+1e-9) for row in good_rows]
print('good ratio on/off true: min',min(on_off_ratios),'p5',np.percentile(on_off_ratios,5),'median',np.median(on_off_ratios))
