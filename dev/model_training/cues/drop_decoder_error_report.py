import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,collections
from dev.model_training.cues.drop_decoder_cross_validation import *
penalties=DECODER_PENALTIES
decoded_results=[decode(drop_start_scores[m[5]:m[5]+len(m[0])],drop_end_scores[m[5]:m[5]+len(m[0])],m[1],penalties) for m in meta]
pickle.dump(decoded_results,open(str(paths.WORK / 'drop_decoder_results.pkl'),'wb'))
pattern_counts=collections.Counter();ex_=[];start_shift_counts=collections.Counter();end_shift_counts=collections.Counter()
for decoded,m in zip(decoded_results,meta):
    true_first_drop_start,true_first_drop_end,H,e2=m[2]
    pattern=('D' if not is_close(decoded[0],true_first_drop_start) else '.')+('e' if not is_close(decoded[1],true_first_drop_end) else '.')
    pattern_counts[(pattern,true_first_drop_start%4!=0)]+=1
    if pattern=='.e': end_shift_counts[round(decoded[1]-true_first_drop_end)]+=1
    if pattern[0]=='D': start_shift_counts[round(decoded[0]-true_first_drop_start)]+=1
n=len(meta);print('D+e1 patterns (pattern, D off 4-grid):',sorted(pattern_counts.items(),key=lambda z:-z[1]))
print('D err shifts',start_shift_counts.most_common(10));print('e1 err shifts (D ok)',end_shift_counts.most_common(8))
offgrid_hits=[(is_close(decoded[0],m[2][0]),is_close(decoded[1],m[2][1])) for decoded,m in zip(decoded_results,meta) if m[2][0]%4!=0]
print('off-grid D tracks',len(offgrid_hits),'D ok',sum(a for a,b in offgrid_hits),'D+e1 ok',sum(a and b for a,b in offgrid_hits))
