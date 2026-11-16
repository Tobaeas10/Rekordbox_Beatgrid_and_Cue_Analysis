import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np,sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
fine_adjust_rows=pickle.load(open(str(paths.WORK / str(paths.WORK / 'rows.pkl')),'rb'))
true_offsets=np.array([row['y'] for row in fine_adjust_rows]); fine_features=np.stack([row['xb'] for row in fine_adjust_rows]); summary_features=np.stack([row['eng'] for row in fine_adjust_rows]); profile_features=np.stack([row['prof'] for row in fine_adjust_rows])
def limit_and_zero_small_offsets(offsets): offsets=np.where(abs(offsets)<1,0,offsets); return np.clip(offsets,-15,15)
def report_errors(name,pred):
    errors=abs(true_offsets-limit_and_zero_small_offsets(pred)); print(f"{name:22s} >5:{(errors>5).sum():3d} >3:{(errors>3).sum():3d} >8:{(errors>8).sum():3d} mean|e| {errors.mean():.2f}",flush=True)
report_errors('no model',np.zeros(len(true_offsets)))
feature_sets={'baseline xb':fine_features,'xb+eng':np.hstack([fine_features,summary_features]),'xb+eng+prof':np.hstack([fine_features,summary_features,profile_features]),'eng+prof only':np.hstack([summary_features,profile_features])}
selected_names=sys.argv[1:] or list(feature_sets)
for feature_set_name in selected_names:
    features=feature_sets[feature_set_name]; cross_validated=np.zeros(len(true_offsets))
    for seed in range(2):
        for train_indices,test_indices in KFold(5,shuffle=True,random_state=seed).split(features):
            model=HistGradientBoostingRegressor(loss='absolute_error',max_depth=3,max_iter=200,learning_rate=0.05,min_samples_leaf=10,random_state=0).fit(features[train_indices],true_offsets[train_indices]); cross_validated[test_indices]+=model.predict(features[test_indices])/2
    report_errors(feature_set_name,cross_validated); np.save(str(paths.WORK / f'cv_{feature_set_name.replace(" ","_").replace("+","_")}.npy'),cross_validated)
