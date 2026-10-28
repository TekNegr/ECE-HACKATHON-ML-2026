import os,sys
os.environ['OMP_NUM_THREADS']='4'
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
tag=sys.argv[1]
B=np.load(f'out/blend_{tag}.npz',allow_pickle=True);CL=list(B['classes'])
nz=lambda P:(lambda Q:Q/Q.sum(1,keepdims=True))(np.clip(P,1e-9,None))
Po,Pt=nz(B['oof']),nz(B['test'])
tr=pd.read_csv('data/train.csv');te=pd.read_csv('data/test.csv')
y=tr.intensity.map({c:i for i,c in enumerate(CL)}).values
Nu=pd.concat([tr.user_id,te.user_id]).value_counts()
tr['ym']=tr.year*12+tr.month;te['ym']=te.year*12+te.month
def assign(logp,caps):
    caps=np.maximum(caps,0)
    if caps.sum()<len(logp):caps=caps+int(np.ceil((len(logp)-caps.sum())/3))
    cols=np.repeat(np.arange(3),caps);r,c=linear_sum_assignment(-logp[:,cols]);o=np.empty(len(logp),int);o[r]=cols[c];return o
def nbfeat(uk,ymk,yk,uq,ymq,w1):
    # counts of known labels same user, same month (and +-1 month weighted w1)
    key=pd.DataFrame({'u':uk,'m':ymk,'y':yk});g=key.groupby(['u','m','y']).size().unstack(fill_value=0).reindex(columns=range(3),fill_value=0)
    d={ix:v for ix,v in zip(g.index,g.values)}
    out=np.zeros((len(uq),3))
    for i,(u,m) in enumerate(zip(uq,ymq)):
        for dm,w in ((0,1.),(-1,w1),(1,w1)):
            v=d.get((u,m+dm))
            if v is not None:out[i]+=w*v
    return out
def run(beta,a,w1,quota=True,folds=20,seed=11):
    sc=[]
    for tri,va in StratifiedKFold(folds,shuffle=True,random_state=seed).split(tr,y):
        kn=pd.crosstab(tr.user_id.values[tri],y[tri]).reindex(columns=range(3),fill_value=0)
        uu=np.r_[tr.user_id.values[va],te.user_id.values];mm=np.r_[tr.ym.values[va],te.ym.values]
        P=np.vstack([Po[va],Pt])
        if beta:
            C=nbfeat(tr.user_id.values[tri],tr.ym.values[tri],y[tri],uu,mm,w1)
            # user prior from known counts
            pr=kn.reindex(uu).fillna(1).values+1.0;pr=pr/pr.sum(1,keepdims=True)
            F=(C+a*pr)/(C.sum(1,keepdims=True)+a)
            L=np.log(P)+beta*np.log(F/pr)
        else:L=np.log(P)
        pred=L.argmax(1)
        if quota:
            for u,idx in pd.Series(np.arange(len(uu))).groupby(uu).indices.items():
                k=kn.reindex([u]).fillna(0).values[0].astype(int)
                pred[idx]=assign(L[idx],int(np.ceil(Nu[u]/3))-k)
        sc.append(f1_score(y[va],pred[:len(va)],average='macro'))
    return np.mean(sc)
if __name__=='__main__':
    print('base',run(0,1,0,False),run(0,1,0,True))
    for beta in (0.5,1.0):
        for a in (3,):
            for w1 in (0,0.5):
                print(beta,a,w1,run(beta,a,w1,False),run(beta,a,w1,True),flush=True)
