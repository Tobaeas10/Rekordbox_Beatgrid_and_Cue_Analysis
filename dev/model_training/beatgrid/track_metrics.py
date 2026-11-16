import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json,re,collections
import os
_report_path=str(paths.EVAL_REPORTS/'eval_report.json')
report=json.load(open(_report_path)) if os.path.exists(_report_path) else []
phase_ranker_fold_ids=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'phase_ranker_folds.json')))
ACAPELLA_PATTERN=re.compile(r'acap|a cappella|acapella',re.I)
TEMPO_CHANGE_PATTERN=re.compile(r'smooth operator|bass symptom|bpm change every',re.I)
def failure_category(report_row):
    dry_run=report_row.get('dry_run') or {}
    if not dry_run.get('ok'): return 'NOGRID'
    categories=[]
    median_difference,p95_difference=dry_run['diff_median_ms'],dry_run['diff_p95_abs_ms']
    if abs(median_difference)>5: categories.append('TIMING')
    elif p95_difference>10: categories.append('SPIKE')
    if dry_run['numbering_shift']!=0: categories.append('DOWNBEAT')
    if abs(dry_run['n_built']-dry_run['n_stored'])>2: categories.append('COUNT')
    return '+'.join(categories) or 'OK'
def track_name(report_row): return f"{report_row['artist']} - {report_row['title']}"
rows=[]
for report_row in report:
    name=track_name(report_row)
    if ACAPELLA_PATTERN.search(name) or TEMPO_CHANGE_PATTERN.search(name): continue
    rows.append((report_row,failure_category(report_row),report_row['id'] in phase_ranker_fold_ids))
if __name__=='__main__':
    for label,selector in (('train(old)',lambda t:t),('HOLDOUT(new)',lambda t:not t),('all',None)):
        selected=[x for x in rows if (selector is None or selector(x[2])==True or (label=='HOLDOUT(new)' and not x[2]))]
        if label=='train(old)': selected=[x for x in rows if x[2]]
        if label=='HOLDOUT(new)': selected=[x for x in rows if not x[2]]
        if label=='all': selected=rows
        category_counts=collections.Counter(x[1] for x in selected)
        wrong_count=sum(count for category,count in category_counts.items() if category!='OK')
        print(label,len(selected),'wrong',wrong_count,f'{100*wrong_count/len(selected):.1f}%',dict(category_counts))
