import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys
import pickle,numpy as np,collections
from dev.model_training.beatgrid.genre_map import REPORT_SETS, PHASE_RANKER_FOLD_TRACK_IDS, genre_of_track
from dev.model_training.beatgrid.track_metrics import ACAPELLA_PATTERN, TEMPO_CHANGE_PATTERN
from lib.beatgrid import rankers
event_cache_folders={set_name:str(paths.EVENT_CACHES/set_name) for set_name in ('old961','new698','extra')}
seen=set(); fine_adjust_data=[]; downbeat_data=[]
for set_name in [candidate_set_name for candidate_set_name in ('old961','new698','extra') if candidate_set_name in REPORT_SETS]:
    for report_row in REPORT_SETS[set_name]:
        if report_row['id'] in seen: continue
        name=report_row['artist']+' - '+report_row['title']
        if ACAPELLA_PATTERN.search(name) or TEMPO_CHANGE_PATTERN.search(name): continue
        dry_run=report_row.get('dry_run') or {}
        if not dry_run.get('ok') or not report_row['sections']: continue
        seen.add(report_row['id'])
        genre_name=genre_of_track(report_row); genre_name={'riddim':'dubstep','tearout':'dubstep'}.get(genre_name,genre_name)
        try: events=pickle.load(open(f"{event_cache_folders[set_name]}/{report_row['id']}.ff2.pkl",'rb'))
        except Exception as e: continue
        report_section=report_row['sections'][0]
        section=dict(period_s=60/report_section['bpm'],start_s=report_section['start_s'],end_s=report_section['end_s'],anchor_s=report_section['anchor_ms']/1000,fit_rms_ms=report_section['fit_rms_ms'],refine=report_section.get('refine'))
        meta=dict(id=report_row['id'],name=name,g=genre_name,set=set_name,old=(report_row['id'] in PHASE_RANKER_FOLD_TRACK_IDS))
        if len(report_row['sections'])==1 and abs(dry_run['diff_median_ms'])<=15:
            fine_adjust_data.append(dict(meta,x=rankers.fine_features(events,section),y=-dry_run['diff_median_ms']))
        if dry_run['numbering_match_frac']>=0.9 and abs(dry_run['n_built']-dry_run['n_stored'])<=2 and report_section['end_s']-report_section['start_s']>=30:
            period_seconds=section['period_s']; anchor_seconds=section['anchor_s']; offset_model_ms=report_row['offset_model']['ms']
            first_bar_index=int(np.ceil((-offset_model_ms/1000-anchor_seconds)/period_seconds-0.003/period_seconds)); first_downbeat_seconds=anchor_seconds+first_bar_index*period_seconds
            reported_downbeat_shift=(report_row.get('downbeat') or {}).get('q',0); true_shift=(reported_downbeat_shift+dry_run['numbering_shift'])%4
            downbeat_candidates=rankers.downbeat_candidates(events,section,first_downbeat_seconds)
            if downbeat_candidates is not None: downbeat_data.append(dict(meta,X=downbeat_candidates,qs=true_shift))
pickle.dump((fine_adjust_data,downbeat_data),open(str(paths.WORK / 'alldata.pkl'),'wb'))
print(len(fine_adjust_data),len(downbeat_data),collections.Counter(row['g'] for row in fine_adjust_data),collections.Counter(row['qs'] for row in downbeat_data))
