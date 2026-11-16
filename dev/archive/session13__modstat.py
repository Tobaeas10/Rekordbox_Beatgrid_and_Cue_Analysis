import json,pickle,numpy as np,sys
sys.path.insert(0,'/home/claude/work/llm_beatgrid_bundle_v6'); import pipeline as P
rep={r['id']:r for r in json.load(open('/mnt/user-data/uploads/1794435802639_eval_report.json'))}
rows={x['id']:x for x in json.load(open('scope.json'))}
fs=json.load(open('foldstat.json'))
CD='/home/claude/work/cache_x/cache/'
def wrap(x,p): return (x+p/2)%p-p/2
mod=[i for i,x in rows.items() if 'med' in x and 5<abs(x['med'])<=20 and x['ns'] is not None]
print(len(mod),'moderate tracks')
print(' med  p95 | mid_arg low_arg | snare_mode(n) kick_mode(n) (event residuals vs truth, ms) | refine n_sn | name')
for i in sorted(mod,key=lambda i:rows[i]['med']):
    r=rep[i]; s=max(r['sections'],key=lambda s:s['end_s']-s['start_s'])
    T=60.0/s['bpm']; tr=(s['anchor_ms']+s['offset_mod_beat_ms']-r['offset_model']['ms'])/1000
    ev=pickle.load(open(CD+i+'.ff2.pkl','rb'))
    m=(ev['t']>=s['start_s'])&(ev['t']<=s['end_s'])
    res=wrap(ev['t'][m]-tr,T/2)*1000
    sn=P.residual_mode(res[ev['c'][m]==1],ev['w'][m][ev['c'][m]==1]); ki=P.residual_mode(res[ev['c'][m]==0],ev['w'][m][ev['c'][m]==0])
    f=fs[i]; rf=s.get('refine') or {}
    print(f"{rows[i]['med']:5.0f} {rows[i]['p95']:4.0f} | {f['mid']['arg']:6.0f} {f['low']['arg']:6.0f} | {sn[0]:6.1f}({sn[1]:3d}) {ki[0]:6.1f}({ki[1]:3d}) | {rf.get('shift_ms')!s:>6} {rf.get('n_snares')!s:>4} {rf.get('method','')[:5]} | {rows[i]['name'][:40]}")
