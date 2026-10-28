# Idée 4 : équilibrage par jeu de probas (s1608, c4, spw) puis vote des décisions (somme des one-hot, égalité -> s1608).
import sys, numpy as np, pandas as pd
sys.path.insert(0,'tools')
from drv2_time import *
tags=['s1608','c4','spw']
S={t:(nz(np.load(ROOT/f'out/blend_{t}.npz',allow_pickle=True)['oof']),nz(np.load(ROOT/f'out/blend_{t}.npz',allow_pickle=True)['test'])) for t in tags}
def run(seed,use,folds=20):
    f=[]
    for trn,va in StratifiedKFold(folds,shuffle=True,random_state=seed).split(tr,y):
        known=pd.crosstab(tr.user_id.values[trn],y[trn]).reindex(columns=range(3),fill_value=0)
        uu=np.r_[tr.user_id.values[va],te.user_id.values]
        votes=np.zeros((len(uu),3))
        for k,t in enumerate(use):
            Pu=np.vstack([S[t][0][va],S[t][1]]); pr=np.empty(len(uu),int)
            for u,idx in pd.Series(np.arange(len(uu))).groupby(uu).indices.items():
                kn=known.reindex([u]).fillna(0).values[0].astype(int)
                pr[idx]=balance(Pu[idx],exact_counts(len(idx),Nu[u],kn).astype(float))
            votes[np.arange(len(uu)),pr]+=1+(1e-3 if k==0 else 0)
        f.append(f1_score(y[va],votes.argmax(1)[:len(va)],average='macro'))
    return np.mean(f)
for use in (['s1608'],['c4'],['s1608','c4','spw'],['s1608','c4']):
    print(use,[round(run(s,use),4) for s in (11,33)],flush=True)
