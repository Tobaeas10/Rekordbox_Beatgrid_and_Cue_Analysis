"""Offline replay of the downbeat rule on a finished run: python dev/downbeat_replay.py [eval_out]. Needs eval_report.json and cache/<id>.ff2.pkl (with bass envelope).
Truth = Rekordbox first_downbeat_ms relative to the first grid beat."""
import sys,json,pickle,numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import pipeline as P
D=(sys.argv[1] if len(sys.argv)>1 else 'eval_out').rstrip('/\\')+'/'
rep=json.load(open(D+'eval_report.json',encoding='utf-8'))
rows=[]
for e in rep:
    if not e['sections'] or not e['dry_run'] or not e['dry_run'].get('ok'): continue
    ev=pickle.load(open(D+f"cache/{e['id']}.ff2.pkl",'rb'))
    if 'bass' not in ev: print('nobass',e['id']); continue
    s=e['sections'][0]; T=60000/s['bpm']
    off=e['offset_model']['ms']; fb=e['dry_run']['first_built_ms']
    k=round((e['rb_grid']['first_downbeat_ms']-fb)/T)%4
    sec=dict(period_s=T/1000,anchor_s=s['anchor_ms']/1000,start_s=s['start_s'],end_s=s['end_s'])
    t0=(fb-off)/1000
    rows.append(dict(name=e['artist'][:14]+' - '+e['title'][:18],k=k,ev=ev,sec=sec,t0=t0,good=abs(e['dry_run']['diff_median_ms'])<20))
print(len(rows),'tracks; truth k counts',np.bincount([r['k'] for r in rows],minlength=4))
if __name__=='__main__':
    bad=[];mv=[]
    for r in rows:
        q,_,info=P.pick_downbeat(r['ev'],r['sec'],r['t0'])
        if q: mv.append((r['name'],r['k'],q))
        if q!=r['k']: bad.append((r['name'],r['k'],q))
    print('moved:',mv); print(len(bad),'tracks with wrong beat-1 phase (default 0 would be',sum(1 for r in rows if r['k']),'):'); [print(' ',b) for b in bad]
