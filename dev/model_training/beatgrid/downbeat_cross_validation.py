import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys
import pickle,numpy as np
from lib.beatgrid import rankers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
fine_adjust_data,downbeat_data=pickle.load(open(str(paths.WORK / 'alldata.pkl'),'rb'))
row_count=len(downbeat_data); true_shifts=np.array([row['qs'] for row in downbeat_data]); features=np.stack([row['X'] for row in downbeat_data]); genres=np.array([row['g'] for row in downbeat_data])
def score(probabilities,threshold,mask=None):
    mask=np.ones(row_count,bool) if mask is None else mask
    moved=fixed=false_moves=wrong_target=0
    for row_index in np.where(mask)[0]:
        best_shift=int(np.argmax(probabilities[row_index]))
        if best_shift!=0 and probabilities[row_index][best_shift]-probabilities[row_index][0]>threshold:
            moved+=1; fixed+=best_shift==true_shifts[row_index]; false_moves+=true_shifts[row_index]==0; wrong_target+=(best_shift!=true_shifts[row_index] and true_shifts[row_index]!=0)
    return f"moves {moved} fixed {fixed} false {false_moves} wrongtarget {wrong_target} (nonzero q* {int((true_shifts[mask]!=0).sum())}/{mask.sum()})"
shipped_model=rankers.LRModel.load(str(paths.MODELS_BEATGRID/'downbeat_model.json'))
shipped_probabilities=np.stack([shipped_model.proba(features[row_index]) for row_index in range(row_count)])
print('SHIPPED on new data th0.3:',score(shipped_probabilities,0.3))
for genre_name in ('dnb','140','jungle'): print('  ',genre_name,score(shipped_probabilities,0.3,genres==genre_name))
for regularization in (0.3,1,3):
  for threshold in (0.2,0.3,0.4):
    totals=np.zeros(4)
    for seed in range(4):
        probabilities=np.zeros((row_count,4))
        for train_indices,test_indices in StratifiedKFold(5,shuffle=True,random_state=seed).split(features[:,0],true_shifts):
            train_labels=np.array([[int(true_shifts[row_index]==shift) for shift in range(4)] for row_index in train_indices]).ravel()
            model=make_pipeline(StandardScaler(),LogisticRegression(C=regularization,max_iter=3000)).fit(features[train_indices].reshape(-1,features.shape[2]),train_labels)
            for row_index in test_indices: probabilities[row_index]=model.predict_proba(features[row_index])[:,1]
        import re
        statistics_text=score(probabilities,threshold); totals+=np.array([int(value) for value in re.findall(r'(?:moves|fixed|false|wrongtarget) (\d+)',statistics_text)])
    print('CV new-only C',regularization,'th',threshold,(totals/4).round(1).tolist(),'[moves,fixed,false,wrongtarget]')
