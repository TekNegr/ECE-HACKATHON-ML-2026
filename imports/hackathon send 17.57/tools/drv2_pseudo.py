# Idée 5 : pseudo-étiquetage itératif. Séances très sûres après équilibrage => "connues" pour recalculer comptes des autres.
import sys, numpy as np, pandas as pd
sys.path.insert(0,'tools')
from drv2_time import *
def pred_pseudo(Pu, uu, known_df, TH, ROUNDS):
    pred = Pu.argmax(1).copy()
    for u, idx in pd.Series(np.arange(len(uu))).groupby(uu).indices.items():
        kn = known_df.reindex([u]).fillna(0).values[0].astype(int)
        P = Pu[idx]; tgt = exact_counts(len(idx),Nu[u],kn).astype(float)
        p = balance(P,tgt)
        for _ in range(ROUNDS):
            conf = P[np.arange(len(P)),p] >= TH
            if conf.all() or not conf.any(): break
            # séances sûres : comptes fixés ; les autres : cible = total - sûres
            t2 = np.maximum(tgt - np.bincount(p[conf],minlength=3),0)
            rest = ~conf
            if t2.sum()<=0: break
            t2 = t2*rest.sum()/t2.sum()
            p2 = p.copy(); p2[rest] = balance(P[rest], t2+1e-6); p = p2
        pred[idx]=p
    return pred
def run(seed,TH,R,folds=20):
    f=[]
    for trn,va in StratifiedKFold(folds,shuffle=True,random_state=seed).split(tr,y):
        known = pd.crosstab(tr.user_id.values[trn], y[trn]).reindex(columns=range(3),fill_value=0)
        Pu=np.vstack([Po[va],Pt]); uu=np.r_[tr.user_id.values[va],te.user_id.values]
        f.append(f1_score(y[va],pred_pseudo(Pu,uu,known,TH,R)[:len(va)],average='macro'))
    return np.mean(f)
for TH,R in [(.9,1),(.8,2),(.7,3)]:
    print(TH,R,[round(run(s,TH,R),4) for s in (11,33)],flush=True)
