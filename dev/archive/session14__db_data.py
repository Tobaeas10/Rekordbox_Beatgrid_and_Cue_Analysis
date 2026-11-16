import sys; sys.path.insert(0,"bundle/llm_beatgrid_bundle_v6")
import pickle,numpy as np,collections
from metric import *
import ranker
D=[]
for r,c,tr in rows:
    dr=r.get('dry_run') or {}
    if not dr.get('ok') or not r['sections']: continue
    if dr['numbering_match_frac']<0.9 or abs(dr['n_built']-dr['n_stored'])>2: continue
    s=r['sections'][0]
    if s['end_s']-s['start_s']<30: continue
    ev=pickle.load(open(f"cache/{r['id']}.ff2.pkl",'rb'))
    sec=dict(period_s=60/s['bpm'],start_s=s['start_s'],end_s=s['end_s'],anchor_s=s['anchor_ms']/1000)
    T=sec['period_s']; a=sec['anchor_s']; mm=r['offset_model']['ms']
    k0=int(np.ceil((-mm/1000-a)/T-0.003/T)); t0=a+k0*T
    q0=(r.get('downbeat') or {}).get('q',0)
    qs=(q0+dr['numbering_shift'])%4
    X=ranker.downbeat_candidates(ev,sec,t0)
    if X is None: continue
    D.append(dict(id=r['id'],name=name(r),X=X,qs=qs,train=tr,q0=q0,cat=c))
pickle.dump(D,open('db.pkl','wb'))
print(len(D),collections.Counter(d['qs'] for d in D),collections.Counter(d['qs'] for d in D if not d['train']))
