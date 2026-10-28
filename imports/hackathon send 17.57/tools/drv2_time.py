import os, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[1]
tag = os.environ.get("TAG", "s1608")
B = np.load(ROOT/"out"/f"blend_{tag}.npz", allow_pickle=True)
CL = list(B["classes"])
nz = lambda P: (lambda Q: Q/Q.sum(1, keepdims=True))(np.clip(P,1e-9,None))
Po, Pt = nz(B["oof"]), nz(B["test"])
tr = pd.read_csv(ROOT/"data/train.csv"); te = pd.read_csv(ROOT/"data/test.csv")
y = tr.intensity.map({c:i for i,c in enumerate(CL)}).values
Nu = pd.concat([tr.user_id, te.user_id]).value_counts()
tm_tr = (tr.year*12+tr.month).values; tm_te = (te.year*12+te.month).values
def exact_counts(n, N, known):
    r = np.maximum(N/3-known,0); r = r*n/max(r.sum(),1e-9); k=np.floor(r).astype(int)
    for j in np.argsort(-(r-k))[:n-k.sum()]: k[j]+=1
    return k
def balance(P, target, it=50):
    w=np.ones(3)
    for _ in range(it):
        Q=P*w; Q/=Q.sum(1,keepdims=True); w*=(target/np.maximum(Q.sum(0),1e-9))**0.7
    return (P*w).argmax(1)
def pred_time(Pu, uu, tu, known_df, known_rows, NB, A, LAM):
    """known_rows: (user, time, label) des connus."""
    pred = Pu.argmax(1).copy()
    kg = known_rows.groupby('u')
    for u, idx in pd.Series(np.arange(len(uu))).groupby(uu).indices.items():
        kn = known_df.reindex([u]).fillna(0).values[0].astype(int)
        tgt = exact_counts(len(idx), Nu[u], kn).astype(float)
        if NB == 1: pred[idx] = balance(Pu[idx], tgt); continue
        kr = kg.get_group(u) if u in kg.groups else known_rows.iloc[:0]
        # buckets de temps par quantiles de toutes les seances de la personne (connus+inconnus)
        alltm = np.r_[kr.t.values, tu[idx]]
        edges = np.quantile(alltm, np.linspace(0,1,NB+1)[1:-1])
        bu = np.searchsorted(edges, tu[idx], side='right'); bk = np.searchsorted(edges, kr.t.values, side='right')
        prior = tgt/tgt.sum()
        T = np.zeros((NB,3))
        for b in range(NB):
            nb = (bu==b).sum()
            if nb==0: continue
            kc = np.bincount(kr.y.values[bk==b], minlength=3).astype(float)
            q = (kc + A*prior)/(kc.sum()+A)   # part de classe dans le bucket (rétrécie)
            T[b] = nb*q
        # IPF vers les comptes globaux
        for _ in range(30):
            T *= (tgt/np.maximum(T.sum(0),1e-9))
            T *= (np.bincount(bu,minlength=NB)/np.maximum(T.sum(1),1e-9))[:,None]
        Tm = LAM*T + (1-LAM)*np.outer(np.bincount(bu,minlength=NB), prior)
        for b in range(NB):
            ii = idx[bu==b]
            if len(ii): pred[ii] = balance(Pu[ii], Tm[b]+1e-6)
    return pred
def run(seed, NB, A, LAM, folds=20):
    f=[]
    for trn, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(tr,y):
        known = pd.crosstab(tr.user_id.values[trn], y[trn]).reindex(columns=range(3), fill_value=0)
        kr = pd.DataFrame({'u':tr.user_id.values[trn],'t':tm_tr[trn],'y':y[trn]})
        Pu = np.vstack([Po[va],Pt]); uu = np.r_[tr.user_id.values[va], te.user_id.values]; tu=np.r_[tm_tr[va],tm_te]
        p = pred_time(Pu,uu,tu,known,kr,NB,A,LAM)[:len(va)]
        f.append(f1_score(y[va],p,average='macro'))
    return np.mean(f)
if __name__=="__main__":
    for NB,A,LAM in [(1,0,0),(2,5,1),(3,5,1),(3,10,1),(3,5,.5),(4,10,1)]:
        print(NB,A,LAM,[round(run(s,NB,A,LAM),4) for s in (11,33)], flush=True)
