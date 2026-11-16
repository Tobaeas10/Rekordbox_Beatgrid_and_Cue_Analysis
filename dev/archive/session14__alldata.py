import sys; sys.path.insert(0,"bundle/llm_beatgrid_bundle_v6")
import pickle,numpy as np,collections
from gm import *
import ranker
CD={'old961':'cache','new698':'new/c698/cache','extra':'new/cx/cache'}
seen=set(); FD=[]; DD=[]
for k in ('old961','new698','extra'):
    for r in SETS[k]:
        if r['id'] in seen: continue
        n=r['artist']+' - '+r['title']
        if AC.search(n) or CONT.search(n): continue
        dr=r.get('dry_run') or {}
        if not dr.get('ok') or not r['sections']: continue
        seen.add(r['id'])
        g=genre(r); g={'riddim':'dubstep','tearout':'dubstep'}.get(g,g)
        try: ev=pickle.load(open(f"{CD[k]}/{r['id']}.ff2.pkl",'rb'))
        except Exception as e: continue
        s=r['sections'][0]
        sec=dict(period_s=60/s['bpm'],start_s=s['start_s'],end_s=s['end_s'],anchor_s=s['anchor_ms']/1000,fit_rms_ms=s['fit_rms_ms'],refine=s.get('refine'))
        meta=dict(id=r['id'],name=n,g=g,set=k,old=(r['id'] in FOLDS))
        if len(r['sections'])==1 and abs(dr['diff_median_ms'])<=15:
            FD.append(dict(meta,x=ranker.fine_features(ev,sec),y=-dr['diff_median_ms']))
        if dr['numbering_match_frac']>=0.9 and abs(dr['n_built']-dr['n_stored'])<=2 and s['end_s']-s['start_s']>=30:
            T=sec['period_s']; a=sec['anchor_s']; mm=r['offset_model']['ms']
            k0=int(np.ceil((-mm/1000-a)/T-0.003/T)); t0=a+k0*T
            q0=(r.get('downbeat') or {}).get('q',0); qs=(q0+dr['numbering_shift'])%4
            X=ranker.downbeat_candidates(ev,sec,t0)
            if X is not None: DD.append(dict(meta,X=X,qs=qs))
pickle.dump((FD,DD),open('alldata.pkl','wb'))
print(len(FD),len(DD),collections.Counter(d['g'] for d in FD),collections.Counter(d['qs'] for d in DD))
