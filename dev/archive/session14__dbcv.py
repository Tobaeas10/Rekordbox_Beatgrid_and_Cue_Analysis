import pickle,numpy as np,sys
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
D=pickle.load(open('db.pkl','rb')); n=len(D)
qs=np.array([d['qs'] for d in D]); X=np.stack([d['X'] for d in D])
def cv(mk,seed,X=X):
    P=np.zeros((n,4))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X[:,0],qs):
        Xtr=X[tr].reshape(-1,X.shape[2]); ytr=np.array([[int(qs[i]==q) for q in range(4)] for i in tr]).ravel()
        m=mk().fit(Xtr,ytr)
        for i in te: P[i]=m.predict_proba(X[i])[:,1]
    return P
def score(P,th):
    mv=fix=false=wm=0
    for i in range(n):
        b=int(np.argmax(P[i]))
        if b!=0 and P[i][b]-P[i][0]>th:
            mv+=1; fix+= b==qs[i]; false+= (qs[i]==0); wm+= (b!=qs[i] and qs[i]!=0)
    return np.array([mv,fix,false,wm])
if __name__=='__main__':
    for C in (0.01,0.03,0.1,0.3,1,3):
        tot={th:np.zeros(4) for th in (0.1,0.2,0.3,0.4,0.5)}
        for seed in range(6):
            P=cv(lambda:make_pipeline(StandardScaler(),LR(C=C,max_iter=3000)),seed)
            for th in tot: tot[th]+=score(P,th)
        print(C,{th:(v/6).round(1).tolist() for th,v in tot.items()})
