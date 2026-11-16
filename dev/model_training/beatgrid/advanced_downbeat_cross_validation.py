import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np,re,warnings;warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
rows=pickle.load(open(str(paths.WORK / str(paths.WORK / 'db_rows.pkl')),'rb'))
row_count=len(rows); true_shifts=np.array([r['qs'] for r in rows]); baseline_features=np.stack([r['X'] for r in rows]); new_features=np.stack([r['Xn'] for r in rows])
def move_statistics(probabilities,threshold):
    moved=fixed=false_moves=wrong_target=0
    for row_index in range(row_count):
        best_shift=int(np.argmax(probabilities[row_index]))
        if best_shift!=0 and probabilities[row_index][best_shift]-probabilities[row_index][0]>threshold:
            moved+=1; fixed+=best_shift==true_shifts[row_index]; false_moves+=true_shifts[row_index]==0; wrong_target+=(best_shift!=true_shifts[row_index] and true_shifts[row_index]!=0)
    return np.array([moved,fixed,false_moves,wrong_target])
def cross_validate(features,regularization,seeds=3):
    results_by_threshold={threshold:np.zeros(4) for threshold in (0.2,0.3,0.4)}
    for seed in range(seeds):
        probabilities=np.zeros((row_count,4))
        for train_indices,test_indices in StratifiedKFold(5,shuffle=True,random_state=seed).split(features[:,0],true_shifts):
            train_labels=np.array([[int(true_shifts[row_index]==shift) for shift in range(4)] for row_index in train_indices]).ravel()
            model=make_pipeline(StandardScaler(),LogisticRegression(C=regularization,max_iter=3000)).fit(features[train_indices].reshape(-1,features.shape[2]),train_labels)
            for row_index in test_indices: probabilities[row_index]=model.predict_proba(features[row_index])[:,1]
        for threshold in results_by_threshold: results_by_threshold[threshold]+=move_statistics(probabilities,threshold)/seeds
    return results_by_threshold
print('nonzero q*',int((true_shifts!=0).sum()),'of',row_count,'  [moves, fixed, false, wrongtarget]')
for variant_name,features in (('baseline',baseline_features),('new only',np.concatenate([baseline_features[:,:,:1],new_features],2)),('baseline+new',np.concatenate([baseline_features,new_features],2))):
    for regularization in (0.1,0.3,1):
        results=cross_validate(features,regularization); print(f'{variant_name:13s} C={regularization}',{threshold:v.round(1).tolist() for threshold,v in results.items()},flush=True)
