import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys
import pickle,numpy as np
from dev.model_training.beatgrid.genre_map import REPORT_SETS
from lib.beatgrid import rankers
from dev.experiments import clean_event_features
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
fine_adjust_data,downbeat_data=pickle.load(open(str(paths.WORK / 'alldata.pkl'),'rb'))
event_cache_folders={set_name:str(paths.EVENT_CACHES / set_name) for set_name in ('old961','new698','extra')}
report_by_id={}
for set_name in ('old961','new698','extra'):
    for report_row in REPORT_SETS[set_name]: report_by_id.setdefault(report_row['id'],(set_name,report_row))
extra_features=[]
for track_data in fine_adjust_data:
    set_name,report_row=report_by_id[track_data['id']]; events=pickle.load(open(f"{event_cache_folders[track_data['set']]}/{track_data['id']}.ff2.pkl",'rb'))
    first_section=report_row['sections'][0]
    section=dict(period_s=60/first_section['bpm'],start_s=first_section['start_s'],end_s=first_section['end_s'],anchor_s=first_section['anchor_ms']/1000)
    extra_features.append(clean_event_features.extra_window_features(events,section))
extra_features=np.array(extra_features); pickle.dump(extra_features,open(str(paths.WORK / 'xe.pkl'),'wb'))
print(extra_features.shape)
