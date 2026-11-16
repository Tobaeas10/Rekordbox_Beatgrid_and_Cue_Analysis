import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys
import pickle,numpy as np,collections
from lib.beatgrid import rankers
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
fine_adjust_data,downbeat_data=pickle.load(open(str(paths.WORK / 'alldata.pkl'),'rb'))
features=np.stack([d['x'] for d in fine_adjust_data]); true_offsets=np.array([d['y'] for d in fine_adjust_data]); genres=np.array([d['g'] for d in fine_adjust_data])
def limit_and_zero_small_offsets(offsets):
    offsets=np.where(abs(offsets)<1,0,offsets); return np.clip(offsets,-15,15)
def report_errors(name,pred,mask=None):
    selected_rows=np.ones(len(true_offsets),bool) if mask is None else mask
    error_before=abs(true_offsets[selected_rows]); error_after=abs(true_offsets[selected_rows]-limit_and_zero_small_offsets(pred[selected_rows]))
    print(f"{name:14s} n={selected_rows.sum():4d} bad>5: {np.sum(error_before>5):3d} -> {np.sum(error_after>5):3d} | fixed {np.sum((error_before>5)&(error_after<=5))} broken {np.sum((error_before<=5)&(error_after>5))} | med|e| {np.median(error_before):.2f}->{np.median(error_after):.2f}")
shipped_model=rankers.GB.load(str(paths.MODELS_BEATGRID/'fine_adjust_model.json'))
shipped_predictions=shipped_model.raw(features)
print('SHIPPED (trained old961) on new data'); report_errors('all',shipped_predictions)
for genre in ('dubstep','dnb','140','jungle'): report_errors(genre,shipped_predictions,genres==genre)
cross_validated=np.zeros(len(true_offsets))
for seed in range(2):
    for train_indices,test_indices in KFold(5,shuffle=True,random_state=seed).split(features):
        model=HistGradientBoostingRegressor(loss='absolute_error',max_depth=3,max_iter=200,learning_rate=0.05,random_state=0).fit(features[train_indices],true_offsets[train_indices]); cross_validated[test_indices]+=model.predict(features[test_indices])/2
print('CV retrained on new only (5fold x2)'); report_errors('all',cross_validated)
for genre in ('dubstep','dnb','140','jungle'): report_errors(genre,cross_validated,genres==genre)
