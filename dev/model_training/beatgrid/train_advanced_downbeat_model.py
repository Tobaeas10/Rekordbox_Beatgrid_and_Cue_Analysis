import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,json,pickle,numpy as np,warnings;warnings.filterwarnings('ignore')
from lib.beatgrid import rankers
from lib.beatgrid import advanced_fine_adjust
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
downbeat_rows=pickle.load(open(str(paths.WORK / str(paths.WORK / 'db_rows2.pkl')),'rb'))
true_shifts=np.array([row['qs'] for row in downbeat_rows]); features=np.stack([np.concatenate([row['X'],row['Xn']],1) for row in downbeat_rows])
flat_features=features.reshape(-1,features.shape[2]); is_true_shift=np.array([[int(true_shifts[row_index]==shift) for shift in range(4)] for row_index in range(len(downbeat_rows))]).ravel()
scaler=StandardScaler().fit(flat_features); logistic_model=LogisticRegression(C=0.3,max_iter=3000).fit(scaler.transform(flat_features),is_true_shift)
gradient_boosting_model=HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=0.05,min_samples_leaf=20,random_state=0).fit(flat_features,is_true_shift)
model_data=dict(lr=dict(mu=scaler.mean_.tolist(),sd=scaler.scale_.tolist(),coef=logistic_model.coef_[0].tolist(),b=float(logistic_model.intercept_[0]),margin=0.3),hgb=rankers.export_hgb(gradient_boosting_model),margin=0.3,n_train=len(downbeat_rows))
model_path=str(paths.WORK / 'advanced_downbeat_model.json'); json.dump(model_data,open(model_path,'w'))
downbeat_model=advanced_fine_adjust.DownbeatModel(json.load(open(model_path)))
probabilities=np.stack([downbeat_model.proba(features[row_index]) for row_index in range(len(downbeat_rows))])
sklearn_probabilities=0.5*logistic_model.predict_proba(scaler.transform(flat_features))[:,1].reshape(-1,4)+0.5*gradient_boosting_model.predict_proba(flat_features)[:,1].reshape(-1,4)
print('max diff vs sklearn',np.abs(probabilities-sklearn_probabilities).max())
moves=fixed=false_moves=0
for row_index in range(len(downbeat_rows)):
    best_shift=int(np.argmax(probabilities[row_index]))
    if best_shift!=0 and probabilities[row_index][best_shift]-probabilities[row_index][0]>0.3: moves+=1; fixed+=best_shift==true_shifts[row_index]; false_moves+=true_shifts[row_index]==0
print('in-sample (not an accuracy figure): moves',moves,'fixed',fixed,'false',false_moves)
