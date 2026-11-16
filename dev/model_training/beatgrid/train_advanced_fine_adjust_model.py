import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,json,pickle,numpy as np,warnings;warnings.filterwarnings('ignore')
from lib.beatgrid import rankers
from lib.beatgrid import advanced_fine_adjust
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
fine_adjust_rows=pickle.load(open(str(paths.WORK / str(paths.WORK / 'rows.pkl')),'rb'))
true_offsets=np.array([row['y'] for row in fine_adjust_rows]); features=np.nan_to_num(np.hstack([np.stack([row['xb'] for row in fine_adjust_rows]),np.stack([row['eng'] for row in fine_adjust_rows]),np.stack([row['prof'] for row in fine_adjust_rows])]))
gradient_boosting_model=HistGradientBoostingRegressor(loss='absolute_error',max_depth=3,max_iter=200,learning_rate=0.05,min_samples_leaf=10,random_state=0).fit(features,true_offsets)
scaler=StandardScaler().fit(features); scaled_features=np.clip(scaler.transform(features),-5,5)
mlp_exports=[];mlp_predictions=[]
for seed in range(3):
    mlp=MLPRegressor(hidden_layer_sizes=(64,),alpha=1.0,max_iter=300,early_stopping=True,random_state=seed).fit(scaled_features,np.clip(true_offsets,-15,15))
    mlp_exports.append(dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),W=[weight_matrix.tolist() for weight_matrix in mlp.coefs_],b=[bias.tolist() for bias in mlp.intercepts_])); mlp_predictions.append(mlp.predict(scaled_features))
model_data=dict(hgb=rankers.export_hgb(gradient_boosting_model),mlps=mlp_exports,w_mlp=0.5,n_train=len(true_offsets))
json.dump(model_data,open(str(paths.WORK / 'advanced_fine_adjust_model.json'),'w'))
fine_adjust_model=advanced_fine_adjust.Model(json.load(open(str(paths.WORK / 'advanced_fine_adjust_model.json'))))
predictions=fine_adjust_model.predict(features); reference_predictions=0.5*np.mean(mlp_predictions,0)+0.5*gradient_boosting_model.predict(features)
print('numpy vs sklearn max abs diff',np.abs(predictions-reference_predictions).max(),'in-sample >5:',(abs(true_offsets-np.clip(np.where(abs(predictions)<1,0,predictions),-15,15))>5).sum())
