import json,pickle,sys,numpy as np
rep=json.load(open('/mnt/user-data/uploads/1794435802639_eval_report.json'))
rows={x['id']:x for x in json.load(open('scope.json'))}
CD='/home/claude/work/cache_x/cache/'
def truth_anchor(r):
    """truth anchor (s, decoded time) and period for the longest section"""
    if not r['sections']: return None
    s=max(r['sections'],key=lambda s:s['end_s']-s['start_s'])
    if 'offset_mod_beat_ms' not in s: return None
    T=60.0/s['bpm']
    tr=(s['anchor_ms']+s['offset_mod_beat_ms']-r['offset_model']['ms'])/1000.0
    return s,T,tr
def fold(nov,dt,T,tr,t0,t1,binms=1.0):
    n=len(nov); tt=np.arange(n)*dt; m=(tt>=t0)&(tt<=t1)
    ph=((tt[m]-tr+T/2)%T)-T/2
    nb=int(round(T*1000/binms)); edges=np.linspace(-T/2,T/2,nb+1)
    h,_=np.histogram(ph,bins=edges,weights=nov[m].astype(np.float64))
    return h,edges
out={}
for r in rep:
    if r['id'] not in rows: continue
    ta=truth_anchor(r)
    if not ta: continue
    s,T,tr=ta
    ev=pickle.load(open(CD+r['id']+'.ff2.pkl','rb'))
    if 'nov_low' not in ev: continue
    res={}
    for b in ('low','mid'):
        h,e=fold(ev['nov_'+b],ev['nov_dt'],T,tr,s['start_s'],s['end_s'])
        hs=np.convolve(np.concatenate([h[-3:],h,h[:3]]),np.ones(5)/5,'valid')
        c=len(hs)//2
        res[b]=dict(arg=float(1000*(e[int(np.argmax(hs))]+ (e[1]-e[0])/2)),h0=float(hs[c-2:c+3].max()/hs.max()),med=float(np.median(hs)/hs.max()))
    out[r['id']]=res
json.dump(out,open('foldstat.json','w'))
print(len(out))
