import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,json,pickle,glob,re,os,numpy as np
from lib.beatgrid import rankers
from lib.beatgrid import advanced_fine_adjust
report_by_id={report['id']:report for report in json.load(open(str(paths.EVAL_REPORTS / 'eval_report_-_all_songs_wip19.json')))}
EXCLUDED_TITLES=re.compile(r'acap|accap|a cappella|smooth operator|bass symptom|bpm change every',re.I)
training_rows=[]
for feature_path in sorted(glob.glob(str(paths.BEAT_FEATURES) + '/**/*.npz',recursive=True)):
    track_id=os.path.basename(feature_path).split('.')[0]
    report=report_by_id.get(track_id)
    if report is None or EXCLUDED_TITLES.search(report['artist']+' '+report['title']): continue
    feature_file=np.load(feature_path,allow_pickle=True)
    if not feature_file['label_ok'] or int(feature_file['info_n_sections'])!=1 or abs(float(feature_file['label_diff_median_ms']))>15: continue
    try: events=pickle.load(open(f'{paths.EVENT_CACHES}/all_songs/{track_id}.ff2.pkl','rb'))
    except Exception: continue
    first_section=report['sections'][0]
    section=dict(period_s=float(feature_file['info_period_s']),start_s=float(feature_file['info_start_s']),end_s=float(feature_file['info_end_s']),anchor_s=float(feature_file['info_anchor_pre_s']),fit_rms_ms=float(feature_file['info_fit_rms_ms']),refine=first_section.get('refine'))
    try: fine_features=rankers.fine_features(events,section)
    except Exception as e: continue
    window_features_result=advanced_fine_adjust.window_features(feature_file['W'],feature_file['Wk'],feature_file['Wpar'],feature_file['Wraw'])
    if window_features_result is None: continue
    summary_features,profile_features=window_features_result
    training_rows.append(dict(id=track_id,y=-float(feature_file['label_diff_median_ms']),xb=np.asarray(fine_features,float),eng=np.asarray(summary_features,float),prof=profile_features,name=report['artist']+' - '+report['title']))
pickle.dump(training_rows,open(str(paths.WORK / str(paths.WORK / 'rows.pkl')),'wb'))
print(len(training_rows),training_rows[0]['xb'].shape,training_rows[0]['eng'].shape,training_rows[0]['prof'].shape,sum(abs(row['y'])>5 for row in training_rows))
