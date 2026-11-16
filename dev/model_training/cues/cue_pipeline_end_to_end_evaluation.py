"""End-to-end OOF eval of the cue pipeline: OOF cand6 probs -> decode -> OOF fakeout end (fk_oof.npy, only where predicted D/H equals truth) -> cue_rules.derive vs the user's real cues."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np,collections
from lib.cues.cue_predictor import decode, DECODER_PENALTIES
from lib.cues.cue_rules import derive
tracks_by_id={track['id']:track for track in pickle.load(open(str(paths.WORK / 'cue_dataset.pkl'),'rb'))};meta=pickle.load(open(str(paths.WORK / 'drop_candidate_meta.pkl'),'rb'))
drop_start_scores=np.load(str(paths.WORK / 'drop_candidate_start_oof.npy'));drop_end_scores=np.load(str(paths.WORK / 'drop_candidate_end_oof.npy'))
fakeout_rows,fakeout_labels,fakeout_groups=pickle.load(open(str(paths.WORK / 'fakeout_rows.pkl'),'rb'));fakeout_oof_probabilities=np.load(str(paths.WORK / 'fakeout_oof_probabilities.npy'))
fakeout_ends={}
for g,decoded in enumerate(fakeout_rows):
    i=np.where(fakeout_groups==g)[0];probabilities=fakeout_oof_probabilities[i];best_index=int(np.argmax(probabilities))
    fakeout_ends[(decoded[0],decoded[1])]=(decoded[2]+(best_index+1)/4.0) if probabilities[best_index]>0.4 else None
HOT_CUE_KINDS={'A':1,'B':2,'C':3,'D':5,'E':6,'F':7,'G':8,'H':9}
is_close=lambda a,b:a is not None and b is not None and abs(a-b)<0.01
counts=collections.Counter()
for ti,m in enumerate(meta):
    candidates,bar_count,(true_first_drop_start,true_first_drop_end,true_second_drop_start,true_second_drop_end),_,_,offset=m;track=tracks_by_id[m[3]]
    decoded=decode(drop_start_scores[offset:offset+len(candidates)],drop_end_scores[offset:offset+len(candidates)],bar_count,DECODER_PENALTIES)
    counts['n']+=1
    if decoded is None: continue
    predicted_first_drop_start,predicted_first_drop_end,predicted_second_drop_start,predicted_second_drop_end=decoded
    first_fakeout_end=fakeout_ends.get((ti,0)) if is_close(predicted_first_drop_start,true_first_drop_start) else None; second_fakeout_end=fakeout_ends.get((ti,1)) if is_close(predicted_second_drop_start,true_second_drop_start) else None
    if first_fakeout_end is not None and predicted_first_drop_end is not None and first_fakeout_end>=predicted_first_drop_end: first_fakeout_end=None
    hot_cues,memory_cues=derive(predicted_first_drop_start,predicted_first_drop_end,predicted_second_drop_start,predicted_second_drop_end,first_fakeout_end,second_fakeout_end)
    true_hot_cues={k:track['hot'].get(v) for k,v in HOT_CUE_KINDS.items()}
    counts['hotABCD_ok']+=all(is_close(hot_cues.get(k),true_hot_cues[k]) if true_hot_cues[k] is not None else k not in hot_cues for k in 'ABCD')
    counts['hotAll_ok']+=all(is_close(hot_cues.get(k),true_hot_cues[k]) if true_hot_cues[k] is not None else k not in hot_cues for k in 'ABCDEFGH')
    true_memory_bars=set(round(x,2) for x in track['mem']);predicted_memory_bars=set(round(x,2) for x in memory_cues)
    counts['mem_hit']+=len(true_memory_bars&predicted_memory_bars);counts['mem_true']+=len(true_memory_bars);counts['mem_pred']+=len(predicted_memory_bars)
    # core mems: truth mems within the first drop incl. its fakeout (D..e1)
    core_memory_bars=set(x for x in true_memory_bars if true_first_drop_start-8<=x<=true_first_drop_end);counts['core_true']+=len(core_memory_bars);counts['core_hit']+=len(core_memory_bars&predicted_memory_bars)
    counts['core_all']+=core_memory_bars<=predicted_memory_bars
track_count=counts['n'];print('tracks',track_count)
print('hot A-D all correct %.3f | hot A-H all correct %.3f'%(counts['hotABCD_ok']/track_count,counts['hotAll_ok']/track_count))
print('mem recall %.3f precision %.3f | first-drop core mem recall %.3f, tracks with ALL core mems %.3f'%(counts['mem_hit']/counts['mem_true'],counts['mem_hit']/counts['mem_pred'],counts['core_hit']/counts['core_true'],counts['core_all']/track_count))
