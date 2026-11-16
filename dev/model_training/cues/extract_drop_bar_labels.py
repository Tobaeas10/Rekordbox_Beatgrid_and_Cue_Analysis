import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np,collections
from lib.cues.drop_features import drop_ends, bar_level_features
cue_dataset=pickle.load(open(str(paths.WORK / 'cue_dataset.pkl'),'rb'))
feature_blocks=[];label_blocks=[];group_blocks=[];track_meta=[];W=[]
for i,track in enumerate(cue_dataset):
    first_drop_start,first_drop_end,second_drop_start,second_drop_end=drop_ends(track)
    bar_features,bar_count=bar_level_features(track)
    bar_labels=np.zeros(bar_count,int);ok=True
    first_drop_start_bar=int(round(first_drop_start))
    if first_drop_end is None: continue
    bar_labels[max(first_drop_start_bar,0):int(round(first_drop_end))]=1
    if second_drop_start is not None and second_drop_end is not None: bar_labels[int(round(second_drop_start)):int(round(second_drop_end))]=1
    elif second_drop_start is not None: continue
    feature_blocks.append(bar_features);label_blocks.append(bar_labels);group_blocks.append(np.full(bar_count,i));track_meta.append((first_drop_start,first_drop_end,second_drop_start,second_drop_end,bar_count))
print(len(feature_blocks),'tracks with usable labels of',len(cue_dataset))
pickle.dump((np.vstack(feature_blocks),np.concatenate(label_blocks),np.concatenate(group_blocks),track_meta,[cue_dataset[i]['id'] for i in sorted(set(np.concatenate(group_blocks)))]),open(str(paths.WORK / 'drop_bar_labels.pkl'),'wb'))
print(np.vstack(feature_blocks).shape,np.concatenate(label_blocks).mean())
