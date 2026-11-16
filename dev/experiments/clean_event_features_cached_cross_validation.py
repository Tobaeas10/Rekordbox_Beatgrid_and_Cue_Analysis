import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
fine_adjust_data,downbeat_data=pickle.load(open(str(paths.WORK / 'alldata.pkl'),'rb')); extra_features=pickle.load(open(str(paths.WORK / 'xe.pkl'),'rb'))
baseline_features=np.stack([track_data['x'] for track_data in fine_adjust_data]); true_offsets=np.array([track_data['y'] for track_data in fine_adjust_data])
limit_and_zero_small_offsets=lambda predictions:np.clip(np.where(abs(predictions)<1,0,predictions),-15,15)
def cross_validated_predictions(features,seed_count=3,**model_settings):
    predictions_per_seed=[]
    for seed in range(seed_count):
        predictions=np.zeros(len(true_offsets))
        for train_indices,test_indices in KFold(5,shuffle=True,random_state=seed).split(features):
            predictions[test_indices]=HistGradientBoostingRegressor(loss='absolute_error',max_depth=model_settings.get('max_depth',3),max_iter=model_settings.get('max_iterations',200),learning_rate=0.05,min_samples_leaf=10,random_state=0).fit(features[train_indices],true_offsets[train_indices]).predict(features[test_indices])
        predictions_per_seed.append(predictions)
    return np.mean(predictions_per_seed,0)
def report_errors(label,predictions):
    errors=abs(true_offsets-limit_and_zero_small_offsets(predictions)); print(f"{label:22s} bad>5 {np.sum(errors>5):3d} bad>3 {np.sum(errors>3):3d} >8 {np.sum(errors>8):3d} mean {errors.mean():.3f} p95 {np.percentile(errors,95):.1f}")
report_errors('none',np.zeros(len(true_offsets)))
report_errors('baseline',cross_validated_predictions(baseline_features))
report_errors('base+extra',cross_validated_predictions(np.hstack([baseline_features,extra_features])))
report_errors('base+extra d4 300',cross_validated_predictions(np.hstack([baseline_features,extra_features]),max_depth=4,max_iterations=300))
report_errors('extra only',cross_validated_predictions(extra_features))
