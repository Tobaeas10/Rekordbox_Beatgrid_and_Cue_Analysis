import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np,sys,collections
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from dev.model_training.cues.cue_labels import labels
from lib.cues.fakeout_predictor import fakeout_candidate_features
tracks_by_id={track['id']:track for track in pickle.load(open(str(paths.WORK / 'cue_dataset.pkl'),'rb'))}
meta=pickle.load(open(str(paths.WORK / 'drop_candidate_meta.pkl'),'rb'))
rows=[]
def fklabel(track,drop_start_bar,fakeout_end,fakeout_flag):
    if fakeout_end is not None or fakeout_flag>0: return fakeout_end
    is_riddim='Riddim' in track['pl'] or 'Liquid Riddim' in track['pl']; max_offset=10 if is_riddim else 6
    o=[m-drop_start_bar for m in track['mem'] if 0.2<m-drop_start_bar<max_offset and abs(m-drop_start_bar-8)>0.01]
    return drop_start_bar+o[0] if o else None
feature_matrix=np.lib.format.open_memmap(str(paths.WORK / 'fakeout_features.npy'),mode='w+',dtype=np.float32,shape=(311660,646));pos=0;target_labels=[];group_ids=[];I=[]
for ti,m in enumerate(meta):
    track=tracks_by_id[m[3]];first_drop_start,first_drop_end,second_drop_start,second_drop_end=m[2];L=labels(track,first_drop_start,second_drop_start)
    for drop_index,(drop_start_bar,fakeout_end,fakeout_flag,st) in enumerate([(first_drop_start,L['fk1'],L['f1'],L['s1']),(second_drop_start,L['fk2'],L['f2'],L['s2'])]):
        if drop_start_bar is None: continue
        if drop_index==1 and st in('none','unk2'): continue
        if st=='ambig_fk16': continue
        if fakeout_flag>0 and fakeout_end is None: continue     # tag but no mc: unknown
        fakeout_end=fklabel(track,drop_start_bar,fakeout_end,fakeout_flag)
        beat_offsets,features=fakeout_candidate_features(track,drop_start_bar)
        if len(beat_offsets)<20: continue
        track_labels=np.zeros(len(beat_offsets),int)
        if fakeout_end is not None:
            label_index=int(round((fakeout_end-drop_start_bar)*4))
            if label_index<1 or label_index>beat_offsets[-1]: continue
            track_labels[label_index-1]=1
        features=np.concatenate([features,np.full((len(beat_offsets),1),drop_index)],1)
        feature_matrix[pos:pos+len(features)]=features;pos+=len(features);target_labels.append(track_labels);group_ids.append(np.full(len(beat_offsets),len(rows)));rows.append((ti,drop_index,drop_start_bar,fakeout_end,fakeout_flag,m[4]))
del tracks_by_id,meta;assert pos==311660,pos;feature_matrix.flush();target_labels=np.concatenate(target_labels);group_ids=np.concatenate(group_ids)
print(feature_matrix.shape,target_labels.sum(),len(rows),flush=True)
pickle.dump((rows,target_labels,group_ids),open(str(paths.WORK / 'fakeout_rows.pkl'),'wb'))
oof=np.zeros(len(target_labels))
for train_indices,test_indices in GroupKFold(5).split(feature_matrix,target_labels,group_ids):
    rng=np.random.RandomState(1);train_indices=train_indices[(target_labels[train_indices]==1)|(rng.rand(len(train_indices))<0.35)]
    model=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.08,max_leaf_nodes=24,l2_regularization=1.0,random_state=0)
    model.fit(feature_matrix[np.sort(train_indices)],target_labels[np.sort(train_indices)]);test_indices=np.sort(test_indices)
    for chunk_start in range(0,len(test_indices),40000): oof[test_indices[chunk_start:chunk_start+40000]]=model.predict_proba(feature_matrix[test_indices[chunk_start:chunk_start+40000]])[:,1];print('fold',flush=True)
np.save(str(paths.WORK / 'fakeout_oof_probabilities.npy'),oof)
