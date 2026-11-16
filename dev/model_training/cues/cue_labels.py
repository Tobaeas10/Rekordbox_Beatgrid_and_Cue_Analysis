"""Ground truth v2 (wip34): drop ends + fakeout ends using the fakeout lengths encoded in the user's titles.
Title: '<name> (<f1>B ...) (2D <f2>B ...)'; f = fakeout length rounded DOWN to 2^n bars (so true length in [f,2f)).
No number = no fakeout (<1 bar). Missing '2D' tag -> f2 unknown (assume = f1, flagged).
Rules: drops >=16 bars; non-riddim fakeouts <6 bars; riddim fakeouts <=~20 (one 28 outlier)."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import re
def parse_title(title):
    match=re.search(r'\((\d+)B\b[^)]*\)',title.split('(2D')[0])
    first_fakeout_length=int(match.group(1)) if match else 0
    has_second_drop_tag='(2D' in title
    second_match=re.search(r'\(2D\s*(\d+)B\b',title)
    second_fakeout_length=int(second_match.group(1)) if second_match else 0
    return first_fakeout_length,(second_fakeout_length if has_second_drop_tag else None)
def _near8(offset): return abs(offset/8-round(offset/8))<0.02
def label_drop(drop_start_bar,limit,memory_cue_bars,fakeout_length,unknown=False):
    """S drop start bar, limit = bar where next drop's pre-section can start (H-16 or song end)
    returns (end_bar or None, fk_end_bar or None, status)"""
    offsets=[memory_bar-drop_start_bar for memory_bar in memory_cue_bars if drop_start_bar<memory_bar<=limit+0.01]
    status='ok'
    fakeout_offset=None
    if fakeout_length>0:
        upper_offset=2*fakeout_length if fakeout_length<16 else 32
        candidates=[offset for offset in offsets if fakeout_length-0.3<=offset<upper_offset]
        if fakeout_length<16:
            if candidates: fakeout_offset=candidates[0]
        else:
            non_multiple_candidates=[offset for offset in candidates if not _near8(offset)]
            if non_multiple_candidates: fakeout_offset=non_multiple_candidates[0]
            else:
                multiple_of_eight_candidates=[offset for offset in candidates if _near8(offset)]
                if len(multiple_of_eight_candidates)>=2: fakeout_offset=multiple_of_eight_candidates[0]        # e.g. fakeout ends 16, drop ends 32
                elif len(multiple_of_eight_candidates)==1: status='ambig_fk16'
    end_offsets=[offset for offset in offsets if offset>=16 and _near8(offset) and offset!=fakeout_offset]
    if fakeout_length>=16 and fakeout_offset is None and status=='ok': pass
    end_offset=end_offsets[0] if end_offsets else None
    if end_offset is None and status=='ok': status='noend'
    return (None if end_offset is None else drop_start_bar+end_offset),(None if fakeout_offset is None else drop_start_bar+fakeout_offset),status
def labels(track,first_drop_start,second_drop_start):
    memory_cue_bars=track['mem'];first_fakeout_length,second_fakeout_length=parse_title(track['name'])
    second_length_unknown=second_fakeout_length is None
    if second_length_unknown: second_fakeout_length=first_fakeout_length
    first_drop_limit=(second_drop_start-8) if second_drop_start is not None else 1e9
    first_drop_end,first_fakeout_end,first_status=label_drop(first_drop_start,first_drop_limit,memory_cue_bars,first_fakeout_length)
    second_drop_end=second_fakeout_end=None;second_status='none'
    if second_drop_start is not None: second_drop_end,second_fakeout_end,second_status=label_drop(second_drop_start,1e9,memory_cue_bars,second_fakeout_length); 
    if second_length_unknown and second_fakeout_length>=16 and second_status=='ok': second_status='unk2'
    return dict(e1=first_drop_end,fk1=first_fakeout_end,s1=first_status,e2=second_drop_end,fk2=second_fakeout_end,s2=second_status,f1=first_fakeout_length,f2=second_fakeout_length)
if __name__=='__main__':
    import pickle,collections
    tracks_by_id={track['id']:track for track in pickle.load(open(str(paths.WORK / 'cue_dataset.pkl'),'rb'))}
    candidate_meta=pickle.load(open(str(paths.WORK / 'drop_candidate_meta.pkl'),'rb'))
    status_counts=collections.Counter();changed_count=0;examples=[];fakeout_length_counts=collections.Counter()
    for meta_row in candidate_meta:
        track=tracks_by_id[meta_row[3]];first_drop_start,first_drop_end,second_drop_start,second_drop_end=meta_row[2]
        new_labels=labels(track,first_drop_start,second_drop_start);status_counts[new_labels['s1']]+=1;status_counts['2:'+new_labels['s2']]+=1
        if new_labels['fk1'] is not None: fakeout_length_counts[(new_labels['f1'],round(new_labels['fk1']-first_drop_start,2))]+=1
        if new_labels['e1']!=first_drop_end or (second_drop_start is not None and new_labels['e2']!=second_drop_end):
            changed_count+=1
            if len(examples)<14: examples.append((meta_row[4][:50],first_drop_start,first_drop_end,new_labels['e1'],second_drop_start,second_drop_end,new_labels['e2'],new_labels['f1'],new_labels['f2'],[memory_bar-first_drop_start for memory_bar in track['mem'] if first_drop_start<memory_bar<first_drop_start+70]))
    print(status_counts,'changed vs old',changed_count,len(candidate_meta))
    for e in examples: print(e)
    print(sorted(fakeout_length_counts.items(),key=lambda z:-z[1])[:25])
