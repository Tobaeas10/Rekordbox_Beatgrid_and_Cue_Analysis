import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np,warnings;warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
fine_adjust_rows=pickle.load(open(str(paths.WORK / 'rows.pkl'),'rb'))
true_offsets=np.array([row['y'] for row in fine_adjust_rows]); features=np.nan_to_num(np.hstack([np.stack([row['xb'] for row in fine_adjust_rows]),np.stack([row['eng'] for row in fine_adjust_rows]),np.stack([row['prof'] for row in fine_adjust_rows])]))
cross_validated=np.zeros(len(true_offsets)); seed_count=3
for seed in range(seed_count):
    for train_indices,test_indices in KFold(5,shuffle=True,random_state=seed).split(features):
        scaler=StandardScaler().fit(features[train_indices]); scaled_train=np.clip(scaler.transform(features[train_indices]),-5,5); scaled_test=np.clip(scaler.transform(features[test_indices]),-5,5)
        cross_validated[test_indices]+=MLPRegressor(hidden_layer_sizes=(64,),alpha=1.0,max_iter=300,early_stopping=True,random_state=seed).fit(scaled_train,np.clip(true_offsets[train_indices],-15,15)).predict(scaled_test)/seed_count
np.save(str(paths.WORK / 'cv_mlp3.npy'),cross_validated)
gradient_boosting_cross_validated=np.load(str(paths.WORK / 'cv_xb_eng_prof.npy')); baseline_cross_validated=np.load(str(paths.WORK / 'cv_baseline_xb.npy'))
def report_errors(label,predictions):
    predictions=np.where(abs(predictions)<1,0,np.clip(predictions,-15,15)); errors=abs(true_offsets-predictions); print(f'{label:18s} >5:{(errors>5).sum()} >3:{(errors>3).sum()} >8:{(errors>8).sum()} mean {errors.mean():.2f}')
report_errors('baseline',baseline_cross_validated); report_errors('mlp x3',cross_validated)
for blend_weight in (0.4,0.5,0.6): report_errors(f'blend {blend_weight}',blend_weight*cross_validated+(1-blend_weight)*gradient_boosting_cross_validated)
