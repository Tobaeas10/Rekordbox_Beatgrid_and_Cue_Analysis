import sys; sys.path.insert(0,"bundle/llm_beatgrid_bundle_v6")
import pickle,numpy as np
from metric import *
import ranker
D=[]
for r,c,tr in rows:
    dr=r.get('dry_run') or {}
    if not dr.get('ok') or len(r['sections'])!=1 or abs(dr['diff_median_ms'])>15: continue
    s=r['sections'][0]
    ev=pickle.load(open(f"cache/{r['id']}.ff2.pkl",'rb'))
    sec=dict(period_s=60/s['bpm'],start_s=s['start_s'],end_s=s['end_s'],anchor_s=s['anchor_ms']/1000,fit_rms_ms=s['fit_rms_ms'],refine=s.get('refine'))
    D.append(dict(id=r['id'],name=name(r),x=ranker.fine_features(ev,sec),y=-dr['diff_median_ms'],train=tr))
pickle.dump(D,open('fine2.pkl','wb')); print(len(D))
