import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys
import pickle,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold,StratifiedKFold
fine_adjust_data,downbeat_data=pickle.load(open(str(paths.WORK / 'alldata.pkl'),'rb'))
limit_and_zero_small_offsets=lambda predictions:np.clip(np.where(abs(predictions)<1,0,predictions),-15,15)
# timing
timing_features=np.stack([row['x'] for row in fine_adjust_data]); timing_targets=np.array([row['y'] for row in fine_adjust_data]); timing_ids=[row['id'] for row in fine_adjust_data]
cross_validated_predictions=[]
for seed in range(3):
    predictions=np.zeros(len(timing_targets))
    for train_indices,test_indices in KFold(5,shuffle=True,random_state=seed).split(timing_features):
        predictions[test_indices]=HistGradientBoostingRegressor(loss='absolute_error',max_depth=3,max_iter=200,learning_rate=0.05,min_samples_leaf=10,random_state=0).fit(timing_features[train_indices],timing_targets[train_indices]).predict(timing_features[test_indices])
    cross_validated_predictions.append(predictions)
cross_validated_predictions=np.array(cross_validated_predictions); mean_predictions=cross_validated_predictions.mean(0); timing_errors=np.abs(timing_targets-limit_and_zero_small_offsets(mean_predictions)); timing_bad_by_id={i:is_bad for i,is_bad in zip(timing_ids,timing_errors>5)}
timing_score_by_id={i:score for i,score in zip(timing_ids,np.abs(mean_predictions)+3*cross_validated_predictions.std(0))}
# downbeat
row_count=len(downbeat_data); true_shifts=np.array([row['qs'] for row in downbeat_data]); downbeat_features=np.stack([row['X'] for row in downbeat_data]); downbeat_ids=[row['id'] for row in downbeat_data]
probabilities=np.zeros((row_count,4))
for train_indices,test_indices in StratifiedKFold(5,shuffle=True,random_state=0).split(downbeat_features[:,0],true_shifts):
    train_labels=np.array([[int(true_shifts[i]==shift) for shift in range(4)] for i in train_indices]).ravel()
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000)).fit(downbeat_features[train_indices].reshape(-1,downbeat_features.shape[2]),train_labels)
    for i in test_indices: probabilities[i]=model.predict_proba(downbeat_features[i])[:,1]
decoded_shifts=np.zeros(row_count,int); margins=np.zeros(row_count)
for i in range(row_count):
    b=int(np.argmax(probabilities[i])); decoded_shifts[i]=b if (b!=0 and probabilities[i][b]-probabilities[i][0]>0.3) else 0
    sorted_probabilities=np.sort(probabilities[i]); margins[i]=probabilities[i][decoded_shifts[i]]-max(probabilities[i][shift] for shift in range(4) if shift!=decoded_shifts[i])
downbeat_bad_by_id={i:is_bad for i,is_bad in zip(downbeat_ids,decoded_shifts!=true_shifts)}; downbeat_score_by_id={i:-margin for i,margin in zip(downbeat_ids,margins)}
common=[i for i in timing_ids if i in downbeat_bad_by_id]
common_count=len(common); print('tracks with both',common_count,'timing bad',sum(timing_bad_by_id[i] for i in common),'downbeat bad',sum(downbeat_bad_by_id[i] for i in common),'either',sum(timing_bad_by_id[i] or downbeat_bad_by_id[i] for i in common))
def rank_uncertainty(score_by_id): 
    scores=np.array([score_by_id[i] for i in common]); return np.argsort(np.argsort(-scores))/common_count   # 0=most uncertain
timing_ranks,downbeat_ranks=rank_uncertainty(timing_score_by_id),rank_uncertainty(downbeat_score_by_id); combined_ranks=np.minimum(timing_ranks,downbeat_ranks)
bad=np.array([timing_bad_by_id[i] or downbeat_bad_by_id[i] for i in common])
print('flag%  unflagged_bad  unflagged_rate%  (union of timing+downbeat flags, each at f/2... and combined min-rank)')
for flag_fraction in (0.05,0.1,0.15,0.2,0.3,0.4,0.5):
    min_rank_flags=combined_ranks<flag_fraction/2*1.0   # each score flags flag_fraction/2 so union <= f
    flagged=(timing_ranks<flag_fraction/2)|(downbeat_ranks<flag_fraction/2)
    print(f"{flag_fraction*100:4.0f}  flagged {flagged.mean()*100:4.1f}%  missed {int((bad&~flagged).sum()):3d}/{int(bad.sum())}  residual rate {100*(bad&~flagged).sum()/common_count:.2f}%")
timing_bad=np.array([timing_bad_by_id[i] for i in common]); downbeat_bad=np.array([downbeat_bad_by_id[i] for i in common])
for flag_fraction in (0.1,0.2,0.3,0.4):
    print(f"timing only flag {flag_fraction*100:.0f}% missed {int((timing_bad&~(timing_ranks<flag_fraction)).sum())}/{int(timing_bad.sum())} | downbeat only flag {flag_fraction*100:.0f}% missed {int((downbeat_bad&~(downbeat_ranks<flag_fraction)).sum())}/{int(downbeat_bad.sum())}")
absolute_predictions=np.abs(limit_and_zero_small_offsets(mean_predictions)); print('timing |pred| pct 80/85/90/95',np.percentile(absolute_predictions,[80,85,90,95]).round(2),'bad min',np.sort(absolute_predictions[timing_errors>5])[:5].round(1))
margin_values=np.array([downbeat_score_by_id[i] for i in downbeat_ids]); print('downbeat -margin pct 80/90/95',np.percentile(-margin_values,[10,5,20]).round(3))
for threshold in (0.5,0.6,0.7,0.8,0.9):
    flags=margins<threshold; print('margin<',threshold,'flagged %.1f%% missed downbeat'%(100*flags.mean()),int(((decoded_shifts!=true_shifts)&~flags).sum()),'/',int((decoded_shifts!=true_shifts).sum()))
for threshold in (1.5,2,2.5,3):
    flags=absolute_predictions>=threshold; print('|pred|>=',threshold,'flagged %.1f%%'%(100*flags.mean()),'timing bad missed',int(((timing_errors>5)&~flags).sum()),'/',int((timing_errors>5).sum()))
print('--- raw |pred| (no deadzone)')
raw_absolute_predictions=np.abs(mean_predictions)
for percentile_level in (80,85,90): print('pct',percentile_level,np.percentile(raw_absolute_predictions,percentile_level).round(2))
for threshold in (0.5,0.7,0.9,1.2):
    flags=raw_absolute_predictions>=threshold; print('raw|pred|>=',threshold,'flagged %.1f%%'%(100*flags.mean()),'timing bad missed',int(((timing_errors>5)&~flags).sum()),'/',int((timing_errors>5).sum()))
