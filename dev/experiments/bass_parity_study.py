import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json,pickle,numpy as np
event_cache_folder=str(paths.EVENT_CACHES / 'old961') + '/'
eval_report=json.load(open(str(paths.EVAL_REPORTS / 'eval_report.json')))
results=[]
for report_row in eval_report:
    events=pickle.load(open(event_cache_folder+f"{report_row['id']}.ff2.pkl",'rb'))
    section=report_row['sections'][0]; period_seconds=60.0/section['bpm']
    median_difference_ms=report_row['dry_run']['diff_median_ms']
    wrong_parity=abs(abs(median_difference_ms)-period_seconds*500)<30
    built_anchor_seconds=section['anchor_ms']/1000
    true_anchor_seconds=built_anchor_seconds+(period_seconds/2 if wrong_parity else 0)
    bass_envelope=events['bass'].astype(float); bass_rate=events['bass_rate']
    bass_envelope=np.log1p(bass_envelope/ (np.median(bass_envelope)+1e-9))
    onset_strength=np.maximum(np.diff(bass_envelope,prepend=bass_envelope[0]),0)
    onset_strength=np.convolve(onset_strength,np.ones(3)/3,'same')
    frame_times=np.arange(len(onset_strength))/bass_rate
    in_section=(frame_times>=section['start_s'])&(frame_times<section['end_s'])
    def score(anchor_seconds,shift_seconds):
        phase=((frame_times[in_section]-anchor_seconds-shift_seconds)/period_seconds)%1.0
        near_beat=(phase<0.06)|(phase>0.94)
        return onset_strength[in_section][near_beat].sum()
    scores=[]
    for shift_seconds in (0,):
        true_score=score(true_anchor_seconds,0)+0; false_score=score(true_anchor_seconds+period_seconds/2,0)
    results.append((report_row['artist'][:14]+'-'+report_row['title'][:16],wrong_parity,true_score/(false_score+1e-9)))
true_over_false_ratios=np.array([result_row[2] for result_row in results if not result_row[1]])
print('good: n',len(true_over_false_ratios),'min',true_over_false_ratios.min(),'p5',np.percentile(true_over_false_ratios,5),'p25',np.percentile(true_over_false_ratios,25),'median',np.median(true_over_false_ratios),'<1:',(true_over_false_ratios<1).sum(),'<1.1',(true_over_false_ratios<1.1).sum())
for result_row in results:
    if result_row[1]: print('BAD',result_row)
for result_row in sorted([result_row for result_row in results if not result_row[1]],key=lambda result_row:result_row[2])[:10]: print(result_row)
