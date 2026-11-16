import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json,sys,collections
from lib.cues.cue_rules import derive
path=sys.argv[1] if len(sys.argv)>1 else str(paths.CUE_TRAINING / 'cues.jsonl')
HOT_CUE_KINDS={'A':1,'B':2,'C':3,'D':5,'E':6,'F':7,'G':8,'H':9}
stats=collections.Counter();bad=[]
for line in open(path):
    record=json.loads(line);cue_list=record['cues']
    hot_cues={k:x['bar'] for x in cue_list if x['kind'] in HOT_CUE_KINDS.values() for k,v in HOT_CUE_KINDS.items() if v==x['kind']}
    if 'D' not in hot_cues or any(hot_cues.get(k,99)<hot_cues['D'] for k in 'EF'): continue
    memory_bars=sorted(x['bar'] for x in cue_list if x['kind']==0)
    first_drop_start=hot_cues['D'];second_drop_start=hot_cues.get('H')
    # drop end/fakeout from mem as in build.py: first mc >=16 bars after start, multiple of 8
    def ends(start_bar,limit_bar):
        fakeout_end=None;drop_end=None
        for m in memory_bars:
            offset_bars=m-start_bar
            if offset_bars<=0 or (limit_bar is not None and m>=limit_bar): continue
            if offset_bars>=16 and offset_bars%8==0 and drop_end is None: drop_end=m
            elif drop_end is None and fakeout_end is None and offset_bars<16: fakeout_end=m
        return fakeout_end,drop_end
    first_fakeout_end,first_drop_end=ends(first_drop_start,second_drop_start);second_fakeout_end,second_drop_end=ends(second_drop_start,None) if second_drop_start is not None else (None,None)
    if first_drop_end is None: stats['no_e1']+=1; continue
    derived_hot_cues,m=derive(first_drop_start,first_drop_end,second_drop_start,second_drop_end,first_fakeout_end,second_fakeout_end)
    stats['n']+=1
    hot_ok=all(abs(derived_hot_cues.get(k,-1)-hot_cues.get(k,-2))<0.01 for k in hot_cues if k in 'ABCEFG')
    stats['hot_ok']+=hot_ok
    stats['hot_missing_pred']+=sum(1 for k in hot_cues if k not in derived_hot_cues)
    true_memory_set=set(round(x,2) for x in memory_bars)
    predicted_memory_set=set(round(x,2) for x in m)
    stats['mem_precision_num']+=len(predicted_memory_set&true_memory_set);stats['mem_pred']+=len(predicted_memory_set)
    stats['pre_hit']+=len({x for x in predicted_memory_set if x<derived_hot_cues.get('A',first_drop_start)}&true_memory_set);stats['pre_pred']+=len([x for x in predicted_memory_set if x<derived_hot_cues.get('A',first_drop_start)])
    if not hot_ok and len(bad)<8: bad.append((record['name'][:40],first_drop_start,second_drop_start,first_drop_end,{k:hot_cues[k] for k in hot_cues},{k:derived_hot_cues.get(k) for k in derived_hot_cues}))
print(dict(stats))
print('hot A-C,E-G exact: %.3f'%(stats['hot_ok']/stats['n']),' mem precision %.3f'%(stats['mem_precision_num']/stats['mem_pred']),' pre-A precision %.3f'%(stats['pre_hit']/stats['pre_pred']))
for b in bad:print(b)
