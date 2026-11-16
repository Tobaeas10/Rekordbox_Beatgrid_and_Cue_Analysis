import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,json,pickle,glob,re,os,numpy as np
from lib.beatgrid import rankers
report_by_id={report['id']:report for report in json.load(open(str(paths.EVAL_REPORTS / 'eval_report_-_all_songs_wip19.json')))}
EXCLUDED_TITLES=re.compile(r'acap|accap|a cappella|smooth operator|bass symptom|bpm change every|fucked',re.I)
from lib.beatgrid import advanced_fine_adjust
training_rows=[]
for feature_path in sorted(glob.glob(str(paths.BEAT_FEATURES) + '/**/*.npz',recursive=True)):
    track_id=os.path.basename(feature_path).split('.')[0]; report=report_by_id.get(track_id)
    if report is None or EXCLUDED_TITLES.search(report['artist']+' '+report['title']) or not report['sections']: continue
    dry_run=report.get('dry_run') or {}
    if not dry_run.get('ok'): continue
    feature_file=np.load(feature_path,allow_pickle=True)
    if not feature_file['label_ok'] or int(feature_file['info_n_sections'])!=1: continue
    first_section=report['sections'][0]
    if not (dry_run['numbering_match_frac']>=0.9 and abs(dry_run['n_built']-dry_run['n_stored'])<=2 and first_section['end_s']-first_section['start_s']>=30): continue
    try: events=pickle.load(open(f'{paths.EVENT_CACHES}/all_songs/{track_id}.ff2.pkl','rb'))
    except Exception: continue
    period_seconds=60/first_section['bpm']; anchor_seconds=first_section['anchor_ms']/1000; offset_model_ms=report['offset_model']['ms']
    section=dict(period_s=period_seconds,start_s=first_section['start_s'],end_s=first_section['end_s'],anchor_s=anchor_seconds,fit_rms_ms=first_section['fit_rms_ms'],refine=first_section.get('refine'))
    first_beat_index=int(np.ceil((-offset_model_ms/1000-anchor_seconds)/period_seconds-0.003/period_seconds)); first_beat_seconds=anchor_seconds+first_beat_index*period_seconds
    stored_shift=(report.get('downbeat') or {}).get('q',0); true_shift=(stored_shift+dry_run['numbering_shift'])%4
    candidate_features=rankers.downbeat_candidates(events,section,first_beat_seconds)
    if candidate_features is None: continue
    new_features=advanced_fine_adjust.downbeat_new_features(feature_file['BE'],float(feature_file['info_lattice_t0_s']),float(feature_file['info_period_s']),first_beat_seconds,first_section['start_s'],first_section['end_s'])
    if new_features is None: continue
    training_rows.append(dict(id=track_id,g=None,X=candidate_features,Xn=new_features,qs=int(true_shift)))
pickle.dump(training_rows,open(str(paths.WORK / str(paths.WORK / 'db_rows2.pkl')),'wb'))
import collections
print(len(training_rows),collections.Counter(report['qs'] for report in training_rows),training_rows[0]['X'].shape,training_rows[0]['Xn'].shape)
