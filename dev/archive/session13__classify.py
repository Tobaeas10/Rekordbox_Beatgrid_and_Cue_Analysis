import json,re,sys,collections
d=json.load(open('/mnt/user-data/uploads/1794435802639_eval_report.json'))
def cls(r):
    dr=r.get('dry_run') or {}
    out={'id':r['id'],'name':f"{r['artist']} - {r['title']}"}
    if not dr.get('ok'): out['cat']='NODRY'; out['why']=str(dr)[:80]; return out
    med=dr['diff_median_ms']; p95=dr['diff_p95_abs_ms']; mx=dr['diff_max_abs_ms']
    ns=dr['numbering_shift']; nm=dr['numbering_match_frac']
    out.update(med=med,p95=p95,mx=mx,ns=ns,nm=nm,nb=dr['n_built'],nst=dr['n_stored'],
               nsec=len(r['sections']),lvl=r['assess']['level'])
    cats=[]
    if abs(med)>5: cats.append('TIMING')   # whole-track offset
    elif p95>10: cats.append('SPIKE')       # partial (tempo change / local)
    if nm<0.5 and ns!=0: cats.append('DOWNBEAT')
    if abs(dr['n_built']-dr['n_stored'])>2: cats.append('COUNT')
    out['cat']='+'.join(cats) or 'OK'
    return out
rows=[cls(r) for r in d]
ac=re.compile(r'acap|a cappella|acapella',re.I)
for x,r in zip(rows,d):
    x['acap']=bool(ac.search(x['name']))
json.dump(rows,open('rows.json','w'))
C=collections.Counter(x['cat'] for x in rows if not x['acap'])
print(len(rows),'tracks; acapella',sum(x['acap'] for x in rows))
print(C)
