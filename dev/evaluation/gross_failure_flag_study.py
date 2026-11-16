import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys
import numpy as np,collections
from dev.model_training.beatgrid.genre_map import REPORT_SETS
from dev.model_training.beatgrid.track_metrics import ACAPELLA_PATTERN, TEMPO_CHANGE_PATTERN, failure_category
rows=[]
for set_name,reports in REPORT_SETS.items():
    for report_row in reports:
        track_name=report_row['artist']+' - '+report_row['title']
        if ACAPELLA_PATTERN.search(track_name) or TEMPO_CHANGE_PATTERN.search(track_name): continue
        rows.append(report_row)
seen=set(); unique_rows=[]
for report_row in rows:
    if report_row['id'] in seen: continue
    seen.add(report_row['id']); unique_rows.append(report_row)
print(len(unique_rows)); print(unique_rows[0]['assess'])
def row_features(report_row):
    assessment=report_row.get('assess') or {}; level=assessment.get('level') if isinstance(assessment,dict) else None
    sections=report_row['sections']; 
    if not sections: return dict(lvl='nogrid',rms=99,n=0,why=[])
    first_section=sections[0]
    return dict(lvl=level,rms=first_section['fit_rms_ms'],n=len(sections),why=(assessment.get('why') or assessment.get('reasons') or []) if isinstance(assessment,dict) else [])
features=[row_features(report_row) for report_row in unique_rows]; categories=[failure_category(report_row) for report_row in unique_rows]
dry_runs=[(report_row.get('dry_run') or {}) for report_row in unique_rows]
gross=[ (category!='OK' and not (category=='TIMING' and abs(dry_run.get('diff_median_ms',0))<=15 )) and 'DOWNBEAT' not in category.replace('DOWNBEAT','') or category in('NOGRID','COUNT') for category,dry_run in zip(categories,dry_runs)]
gross=np.array([ (category in('NOGRID',) or 'COUNT' in category or ('SPIKE' in category) or ('TIMING' in category and abs(dry_run.get('diff_median_ms',0))>15)) for category,dry_run in zip(categories,dry_runs)])
print('gross-type failures',gross.sum(),'of',len(unique_rows),collections.Counter(categories))
fit_rms_values=np.array([feature['rms'] for feature in features]); levels=np.array([feature['lvl'] for feature in features],dtype=object)
for rule_name,flagged in (('level==check',levels=='check'),('rms>4',fit_rms_values>4),('rms>5',fit_rms_values>5),('check|rms>4',(levels=='check')|(fit_rms_values>4)),('check|rms>3.5|nsec>1',(levels=='check')|(fit_rms_values>3.5)|np.array([feature['n']>1 for feature in features])),('level!=ok',levels!='ok')):
    print(f"{rule_name:24s} flagged {flagged.mean()*100:5.1f}%  gross caught {int((flagged&gross).sum())}/{int(gross.sum())}")
