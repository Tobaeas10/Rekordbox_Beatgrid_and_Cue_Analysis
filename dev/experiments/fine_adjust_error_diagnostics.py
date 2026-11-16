import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys,pickle,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
fine_adjust_data,downbeat_data=pickle.load(open(str(paths.WORK / 'alldata.pkl'),'rb'))
features=np.stack([track_data['x'] for track_data in fine_adjust_data]); true_offsets=np.array([track_data['y'] for track_data in fine_adjust_data]); genres=np.array([track_data['g'] for track_data in fine_adjust_data])
limit_and_zero_small_offsets=lambda predictions:np.clip(np.where(abs(predictions)<1,0,predictions),-15,15)
cross_validated_runs=[]
for seed in range(3):
    cross_validated_predictions=np.zeros(len(true_offsets))
    for train_indices,test_indices in KFold(5,shuffle=True,random_state=seed).split(features):
        cross_validated_predictions[test_indices]=HistGradientBoostingRegressor(loss='absolute_error',max_depth=3,max_iter=200,learning_rate=0.05,min_samples_leaf=10,random_state=0).fit(features[train_indices],true_offsets[train_indices]).predict(features[test_indices])
    cross_validated_runs.append(cross_validated_predictions)
cross_validated_runs=np.array(cross_validated_runs); cross_validated_predictions=cross_validated_runs.mean(0); errors_after=np.abs(true_offsets-limit_and_zero_small_offsets(cross_validated_predictions)); errors_before=np.abs(true_offsets)
print('label |y| pct 50/90/95/99',np.percentile(errors_before,[50,90,95,99]).round(1),'mean',errors_before.mean().round(2))
print('after |err| pct 50/90/95/99',np.percentile(errors_after,[50,90,95,99]).round(1),'mean',errors_after.mean().round(2),'bad>5',(errors_after>5).sum(),'bad>3',(errors_after>3).sum(),'bad>8',(errors_after>8).sum())
print('seed-to-seed pred spread (std) median',np.median(cross_validated_runs.std(0)).round(2),' for bad tracks',np.median(cross_validated_runs.std(0)[errors_after>5]).round(2))
bad_tracks=errors_after>5
print('bad: |y| median',np.median(errors_before[bad_tracks]),'pred median |p|',np.median(abs(cross_validated_predictions[bad_tracks])).round(1),'; bad with |y|>15 clipped:',(errors_before[bad_tracks]>=14.9).sum())
# raw-mode disagreement features: idx0 = snare? mode(t,cl1) ,3 = (t,cl0)
kick_snare_mode_gap=np.abs(features[:,0]-features[:,3]); fit_rms=features[:,12]
for score_name,score_values in (('kick-vs-snare mode gap',kick_snare_mode_gap),('fit rms',fit_rms),('|pred|',abs(cross_validated_predictions)),('seed spread',cross_validated_runs.std(0))):
    print('AUC bad>5 vs',score_name,round(roc_auc_score(bad_tracks,score_values),3))
# review budget: flag top-k% by seed spread+|pred| and fraction of bad caught
for score_name,score_values in (('seedspread',cross_validated_runs.std(0)),('|pred|',abs(cross_validated_predictions)),('kick gap',kick_snare_mode_gap)):
    order=np.argsort(-score_values)
    print(score_name,'flag top 5%/10%/20% catches bad',[int(bad_tracks[order[:int(len(true_offsets)*flag_fraction)]].sum()) for flag_fraction in (.05,.1,.2)],'of',bad_tracks.sum())
print('bad genre',{genre:int((bad_tracks&(genres==genre)).sum()) for genre in set(genres)}, 'bad bpm>160:',int((bad_tracks&(features[:,13]>160)).sum()))
order=np.argsort(-errors_after)[:12]
print('WORST',[(fine_adjust_data[i]['name'][:48],round(float(true_offsets[i]),1),round(float(limit_and_zero_small_offsets(cross_validated_predictions)[i]),1)) for i in order])
