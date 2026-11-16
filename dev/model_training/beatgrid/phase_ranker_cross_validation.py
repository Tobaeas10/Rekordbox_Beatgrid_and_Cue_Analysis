import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pandas as pd,numpy as np,sys,warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from lib.beatgrid import rankers
candidates=pd.read_pickle(str(paths.WORK / 'phase_cand2.pkl')); track_info=pd.read_pickle(str(paths.WORK / 'phase_cand2_info.pkl'))
feature_names=rankers.PHASE_FEATS
make_model=lambda:HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,max_iter=150,l2_regularization=1.0,random_state=0)
track_ids=np.array(sorted(candidates.id.unique()))
def out_of_fold_probabilities(seed,fold_count=5):
    random_generator=np.random.RandomState(seed); shuffled_ids=track_ids.copy(); random_generator.shuffle(shuffled_ids); fold_of_track={track_id:fold_index%fold_count for fold_index,track_id in enumerate(shuffled_ids)}
    fold_indices=candidates.id.map(fold_of_track).values; probabilities=np.zeros(len(candidates))
    for fold_index in range(fold_count):
        model=make_model().fit(candidates.loc[fold_indices!=fold_index,feature_names],candidates.label[fold_indices!=fold_index]); probabilities[fold_indices==fold_index]=model.predict_proba(candidates.loc[fold_indices==fold_index,feature_names])[:,1]
    return probabilities
def evaluate_margin(probabilities,margin):
    scored=candidates.assign(p=probabilities); gains=losses=moved=selected_ok=0
    for track_id,track_group in scored.groupby('id'):
        ours=track_group[track_group.is_ours==1]; best=track_group.loc[track_group.p.idxmax()]
        ours_probability=ours.p.max() if len(ours) else -1
        selected=best if (best.is_ours==0 and best.p-ours_probability>margin) else (ours.loc[ours.p.idxmax()] if len(ours) else best)
        ours_label=int(ours.label.max()) if len(ours) else 0
        selected_ok+=selected.label; moved+=int(selected is best and best.is_ours==0)
        gains+=int(selected.label==1 and ours_label==0); losses+=int(selected.label==0 and ours_label==1)
    return selected_ok,moved,gains,losses
print('tracks',candidates.id.nunique(),'ours ok',int(track_info.ours_ok.sum()))
probability_sets=[out_of_fold_probabilities(s) for s in (0,1,2)]
for model in (0,0.1,0.2,0.3):
    result=np.mean([evaluate_margin(shuffled_ids,model) for shuffled_ids in probability_sets],0); print(f'margin {model}: selected_ok {result[0]:.1f} moved {result[1]:.1f} gains {result[2]:.1f} losses {result[3]:.1f}')
