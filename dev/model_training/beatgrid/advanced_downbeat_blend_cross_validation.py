import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import os,pickle,numpy as np,warnings;warnings.filterwarnings('ignore')
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'advanced_downbeat_cross_validation.py')).read().split("print('nonzero")[0])
from sklearn.ensemble import HistGradientBoostingClassifier
def cross_validated_probabilities(features,make_model,seed=0):
    probabilities=np.zeros((row_count,4))
    for train_indices,test_indices in StratifiedKFold(5,shuffle=True,random_state=seed).split(features[:,0],true_shifts):
        train_labels=np.array([[int(true_shifts[row_index]==q) for q in range(4)] for row_index in train_indices]).ravel()
        model=make_model().fit(features[train_indices].reshape(-1,features.shape[2]),train_labels)
        for row_index in test_indices: probabilities[row_index]=model.predict_proba(features[row_index])[:,1]
    return probabilities
combined_features=np.concatenate([baseline_features,new_features],2)
make_logistic_model=lambda:make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=3000))
make_boosting_model=lambda:HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=0.05,min_samples_leaf=20,random_state=0)
baseline_probabilities=cross_validated_probabilities(baseline_features,lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000)))
new_logistic_probabilities=cross_validated_probabilities(combined_features,make_logistic_model); new_boosting_probabilities=cross_validated_probabilities(combined_features,make_boosting_model)
for model_name,probabilities in (('base LR',baseline_probabilities),('new LR',new_logistic_probabilities),('new HGB',new_boosting_probabilities),('avg LR+HGB',0.5*new_logistic_probabilities+0.5*new_boosting_probabilities)):
    print(model_name,{threshold:move_statistics(probabilities,threshold).tolist() for threshold in (0.2,0.3,0.4)})
# trigger coverage: baseline margin of the CV baseline model
confident_zero_margin=np.array([baseline_probabilities[row_index][0]-max(baseline_probabilities[row_index][1:]) if True else 0 for row_index in range(row_count)])  # margin of q=0 vs best other (positive = confident 0)
decoded_shift=np.array([int(np.argmax(baseline_probabilities[row_index])) for row_index in range(row_count)])
margin=np.array([baseline_probabilities[row_index][decoded_shift[row_index]]-max(value for other_shift,value in enumerate(baseline_probabilities[row_index]) if other_shift!=decoded_shift[row_index]) for row_index in range(row_count)])
fixed_by_new_model=np.zeros(row_count,bool)
for row_index in range(row_count):
    best_shift=int(np.argmax(new_logistic_probabilities[row_index]))
    fixed_by_new_model[row_index]= (best_shift!=0 and new_logistic_probabilities[row_index][best_shift]-new_logistic_probabilities[row_index][0]>0.3 and best_shift==true_shifts[row_index] and true_shifts[row_index]!=0)
print('new model fixes',fixed_by_new_model.sum(),'(basic base fixes',sum(int(np.argmax(baseline_probabilities[row_index]))==true_shifts[row_index] and true_shifts[row_index]!=0 and baseline_probabilities[row_index][true_shifts[row_index]]-baseline_probabilities[row_index][0]>0.3 for row_index in range(row_count)),')')
for margin_limit in (0.6,0.9,1.2): 
    triggered=margin<margin_limit; print('basic margin <',margin_limit,'triggers %.1f%%'%(100*triggered.mean()),'covers fixes',(triggered&fixed_by_new_model).sum(),'/',fixed_by_new_model.sum(),'covers all nonzero q*',(triggered&(true_shifts!=0)).sum(),'/',(true_shifts!=0).sum())
