import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,json,pickle,numpy as np
from lib.beatgrid import rankers
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
fine_adjust_data,downbeat_data=pickle.load(open(str(paths.WORK / 'alldata.pkl'),'rb'))
fine_adjust_features=np.stack([row['x'] for row in fine_adjust_data]); true_offsets=np.array([row['y'] for row in fine_adjust_data])
gradient_boosting_model=HistGradientBoostingRegressor(loss='absolute_error',max_depth=3,max_iter=200,learning_rate=0.05,min_samples_leaf=10,random_state=0).fit(fine_adjust_features,true_offsets)
fine_adjust_model_data=rankers.export_hgb(gradient_boosting_model); json.dump(fine_adjust_model_data,open(str(paths.WORK / 'fine_adjust_model.json'),'w'))
exported_model=rankers.GB.load(str(paths.WORK / 'fine_adjust_model.json')); 
try: exported_predictions=exported_model.raw(fine_adjust_features)
except Exception as error: print("GB err",error); exported_predictions=None
print('fine max |sklearn-export|',None if exported_predictions is None else float(np.abs(exported_predictions-gradient_boosting_model.predict(fine_adjust_features)).max()), 'rows',len(true_offsets))
true_shifts=np.array([row['qs'] for row in downbeat_data]); downbeat_features=np.stack([row['X'] for row in downbeat_data]); row_count,candidate_count,feature_count=downbeat_features.shape
is_true_shift=np.array([[int(true_shifts[row_index]==shift) for shift in range(4)] for row_index in range(row_count)]).ravel()
scaler=StandardScaler().fit(downbeat_features.reshape(-1,feature_count)); logistic_model=LogisticRegression(C=1,max_iter=3000).fit(scaler.transform(downbeat_features.reshape(-1,feature_count)),is_true_shift)
downbeat_model_data=dict(mu=scaler.mean_.tolist(),sd=scaler.scale_.tolist(),coef=logistic_model.coef_[0].tolist(),b=float(logistic_model.intercept_[0]),margin=0.3)
json.dump(downbeat_model_data,open(str(paths.WORK / 'downbeat_model.json'),'w'))
downbeat_model=rankers.LRModel.load(str(paths.WORK / 'downbeat_model.json')); probabilities=np.array([downbeat_model.proba(downbeat_features[row_index]) for row_index in range(row_count)])
moves=fixed=false_moves=wrong_moves=0
for row_index in range(row_count):
    best_shift=int(np.argmax(probabilities[row_index]))
    if best_shift!=0 and probabilities[row_index][best_shift]-probabilities[row_index][0]>0.3: moves+=1; fixed+=best_shift==true_shifts[row_index]; false_moves+=true_shifts[row_index]==0; wrong_moves+=(best_shift!=true_shifts[row_index] and true_shifts[row_index]!=0)
print('downbeat in-sample moves',moves,'fixed',fixed,'false',false_moves,'wrong',wrong_moves,'of nonzero',int((true_shifts!=0).sum()))
