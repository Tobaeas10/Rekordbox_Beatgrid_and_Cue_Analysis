import json,re,collections,sys
R=json.load(open('/mnt/user-data/uploads/eval_report.json'))
F=json.load(open('bundle/llm_beatgrid_bundle_v6/dev/session13/folds.json'))
AC=re.compile(r'acap|a cappella|acapella',re.I)
CONT=re.compile(r'smooth operator|bass symptom|bpm change every',re.I)
def cat(r):
    dr=r.get('dry_run') or {}
    if not dr.get('ok'): return 'NOGRID'
    c=[]
    med,p95=dr['diff_median_ms'],dr['diff_p95_abs_ms']
    if abs(med)>5: c.append('TIMING')
    elif p95>10: c.append('SPIKE')
    if dr['numbering_shift']!=0: c.append('DOWNBEAT')
    if abs(dr['n_built']-dr['n_stored'])>2: c.append('COUNT')
    return '+'.join(c) or 'OK'
def name(r): return f"{r['artist']} - {r['title']}"
rows=[]
for r in R:
    n=name(r)
    if AC.search(n) or CONT.search(n): continue
    rows.append((r,cat(r),r['id'] in F))
if __name__=='__main__':
    for lab,sel in (('train(old)',lambda t:t),('HOLDOUT(new)',lambda t:not t),('all',None)):
        s=[x for x in rows if (sel is None or sel(x[2])==True or (lab=='HOLDOUT(new)' and not x[2]))]
        if lab=='train(old)': s=[x for x in rows if x[2]]
        if lab=='HOLDOUT(new)': s=[x for x in rows if not x[2]]
        if lab=='all': s=rows
        c=collections.Counter(x[1] for x in s)
        bad=sum(v for k,v in c.items() if k!='OK')
        print(lab,len(s),'wrong',bad,f'{100*bad/len(s):.1f}%',dict(c))
