import pandas as pd,numpy as np,sys
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
df=pd.read_pickle('phase_cand.pkl')
FEATS=[c for c in df.columns if c not in('id','pos','label','dpos','tru','cur_err','ncand')]
def run(make,seed,nf=5,feats=FEATS):
    ids=df.id.unique(); rng=np.random.RandomState(seed); rng.shuffle(ids)
    fold={i:k%nf for k,i in enumerate(ids)}; f=df.id.map(fold).values
    p=np.zeros(len(df))
    for k in range(nf):
        tr=f!=k; te=f==k
        m=make(); m.fit(df.loc[tr,feats],df.label[tr]); p[te]=m.predict_proba(df.loc[te,feats])[:,1]
    return p
def evaluate(p,margin=0.0,verbose=False):
    d=df.assign(p=p); res=[]
    for i,g in d.groupby('id'):
        ours=g[g.is_ours==1]
        best=g.loc[g.p.idxmax()]
        po=ours.p.max() if len(ours) else -1
        use=best if (best.is_ours==0 and best.p-po>margin) else (ours.loc[ours.p.idxmax()] if len(ours) else best)
        res.append(dict(id=i,ours_ok=int(abs(g.tru.iloc[0])<=TOL),sel_ok=int(use.label),moved=int(use.is_ours==0 and len(ours)>0)))
    r=pd.DataFrame(res); return r
TOL=8.0
if __name__=='__main__':
    mk={'hgb':lambda:HistGradientBoostingClassifier(max_depth=3,learning_rate=0.06,max_iter=150,l2_regularization=1.0,random_state=0),
        'lr':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=0.5,max_iter=2000))}
    for name,make in mk.items():
        ps=[run(make,s) for s in (0,1,2)]
        for margin in (0.0,0.1,0.2,0.3,0.5):
            out=[]
            for p in ps:
                r=evaluate(p,margin)
                out.append((r.ours_ok.sum(),r.sel_ok.sum(),r.moved.sum(),((r.sel_ok==1)&(r.ours_ok==0)).sum(),((r.sel_ok==0)&(r.ours_ok==1)).sum()))
            o=np.mean(out,0)
            print(f'{name} margin {margin:.1f}: ours_ok {o[0]:.0f}/753  selected_ok {o[1]:.1f}  moved {o[2]:.1f}  gains {o[3]:.1f}  losses {o[4]:.1f}')
