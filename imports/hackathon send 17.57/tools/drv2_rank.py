# Idée 1b : terme de rang temporel appris (logreg 2e niveau sur log P + rang relatif dans la personne), puis équilibrage.
import sys, numpy as np, pandas as pd
sys.path.insert(0,'tools')
from drv2_time import *
from sklearn.linear_model import LogisticRegression
allu = pd.concat([tr.user_id, te.user_id]).values; allt = np.r_[tm_tr, tm_te]
rk = pd.Series(allt).groupby(allu).rank(pct=True, method='average').values
rtr, rte = rk[:len(tr)], rk[len(tr):]
def feats(P,r): L=np.log(P); return np.c_[L, r[:,None]*L, r, r**2]
def run(seed, C, folds=20, use_rank=True):
    f=[]
    for trn,va in StratifiedKFold(folds,shuffle=True,random_state=seed).split(tr,y):
        known = pd.crosstab(tr.user_id.values[trn], y[trn]).reindex(columns=range(3),fill_value=0)
        # 2e niveau appris sur OOF des plis d'entraînement seulement
        m = LogisticRegression(C=C,max_iter=500).fit(feats(Po[trn],rtr[trn]) if use_rank else np.log(Po[trn]), y[trn])
        Pv = np.vstack([Po[va],Pt]); rv=np.r_[rtr[va],rte]
        Q = m.predict_proba(feats(Pv,rv) if use_rank else np.log(Pv))
        uu=np.r_[tr.user_id.values[va],te.user_id.values]
        pred=Q.argmax(1).copy()
        for u,idx in pd.Series(np.arange(len(uu))).groupby(uu).indices.items():
            kn=known.reindex([u]).fillna(0).values[0].astype(int)
            pred[idx]=balance(Q[idx],exact_counts(len(idx),Nu[u],kn).astype(float))
        f.append(f1_score(y[va],pred[:len(va)],average='macro'))
    return np.mean(f)
for ur in (False,True):
    print('rank',ur,[round(run(s,1.0,use_rank=ur),4) for s in (11,33)],flush=True)
