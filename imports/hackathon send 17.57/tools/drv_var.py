import os
os.environ['OMP_NUM_THREADS']='4'
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
nz=lambda P:(lambda Q:Q/Q.sum(1,keepdims=True))(np.clip(P,1e-9,None))
A=np.load('out/blend_s1608.npz',allow_pickle=True);Bc=np.load('out/blend_c4.npz',allow_pickle=True)
CL=list(A['classes']);assert CL==list(Bc['classes'])
tr=pd.read_csv('data/train.csv');te=pd.read_csv('data/test.csv')
y=tr.intensity.map({c:i for i,c in enumerate(CL)}).values
Nu=pd.concat([tr.user_id,te.user_id]).value_counts()
def assign(logp,caps):
    caps=np.maximum(caps,0)
    if caps.sum()<len(logp):caps=caps+int(np.ceil((len(logp)-caps.sum())/3))
    cols=np.repeat(np.arange(3),caps);r,c=linear_sum_assignment(-logp[:,cols]);o=np.empty(len(logp),int);o[r]=cols[c];return o
def exact_counts(n,N,known):
    r=np.maximum(N/3-known,0);r=r*n/max(r.sum(),1e-9);k=np.floor(r).astype(int)
    for j in np.argsort(-(r-k))[:n-k.sum()]:k[j]+=1
    return k
def balance(P,target,it=50):
    # per-person prior shift so that sum of probas matches target counts
    w=np.ones(3)
    for _ in range(it):
        Q=P*w;Q/=Q.sum(1,keepdims=True);w*=(target/np.maximum(Q.sum(0),1e-9))**0.7
    Q=P*w;return Q/Q.sum(1,keepdims=True)
def evaluate(Po,Pt,mode,folds=20,seed=11,T=1.0):
    sc=[]
    for tri,va in StratifiedKFold(folds,shuffle=True,random_state=seed).split(tr,y):
        kn=pd.crosstab(tr.user_id.values[tri],y[tri]).reindex(columns=range(3),fill_value=0)
        uu=np.r_[tr.user_id.values[va],te.user_id.values];P=np.vstack([Po[va],Pt])
        pred=P.argmax(1).copy()
        for u,idx in pd.Series(np.arange(len(uu))).groupby(uu).indices.items():
            k=kn.reindex([u]).fillna(0).values[0].astype(int);n=len(idx);N=Nu[u]
            ex=exact_counts(n,N,k);Pi=P[idx]
            if 'bal' in mode: Pi=balance(Pi,ex.astype(float))
            L=np.log(Pi)/T
            if mode.endswith('caps'):pred[idx]=assign(L,int(np.ceil(N/3))-k)
            elif mode.endswith('exact'):pred[idx]=assign(L,ex)
            elif mode=='bal':pred[idx]=L.argmax(1)
        sc.append(f1_score(y[va],pred[:len(va)],average='macro'))
    return np.mean(sc)
if __name__=='__main__':
    for nm,(Po,Pt) in {'s1608':(nz(A['oof']),nz(A['test'])),'c4':(nz(Bc['oof']),nz(Bc['test'])),
        'mixarith':(nz(A['oof']+Bc['oof']),nz(A['test']+Bc['test'])),
        'mixgeo':(nz(np.sqrt(A['oof']*Bc['oof'])),nz(np.sqrt(A['test']*Bc['test'])))}.items():
        print(nm,{m:round(evaluate(Po,Pt,m),4) for m in ('caps','exact','bal','bal+caps','bal+exact')},flush=True)
