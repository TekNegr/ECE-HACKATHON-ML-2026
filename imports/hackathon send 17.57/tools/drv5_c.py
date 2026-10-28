import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('tools/drv5_b.py').read().split('# 2. dominance')[0])
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
# demeaning par personne x sport + rang intra personne x sport
key=d.user_id.astype(str)+'_'+d.sport
base=F[[c for c in F.columns if not c.startswith('sp_')]]
Fd=pd.concat([base-base.groupby(key).transform('mean'), base.groupby(key).rank(pct=True)-.5, base.groupby(d.user_id).rank(pct=True).add_suffix('_pr')-.5],axis=1).fillna(0)
rng=np.random.default_rng(0); users=d.user_id.values
def pairs(idxs,n_per=400):
    X=[];Y=[]
    for u in np.unique(users[idxs]):
        ii=idxs[users[idxs]==u]
        if len(ii)<3: continue
        yy=d.yi.values[ii]; a=rng.choice(len(ii),n_per);b=rng.choice(len(ii),n_per)
        k=yy[a]!=yy[b]; a,b=ii[a[k]],ii[b[k]]
        X.append(Fd.values[a]-Fd.values[b]);Y.append(d.yi.values[a]>d.yi.values[b])
    X=np.vstack(X);Y=np.concatenate(Y); return np.vstack([X,-X]),np.concatenate([Y,~Y])
trn=np.where(d.tr)[0]; score=np.zeros(len(d))
for a,b in GroupKFold(5).split(trn,groups=users[trn]):
    X,Y=pairs(trn[a]); m=LogisticRegression(C=1,max_iter=500,fit_intercept=False).fit(X,Y)
    ib=np.where(np.isin(users,np.unique(users[trn[b]])))[0]; score[ib]=Fd.values[ib]@m.coef_[0]
pred=np.zeros(len(d),int)
for u,g in d.groupby('user_id'):
    r=rankdata(score[g.index],method='ordinal');pred[g.index]=np.floor((r-1)*3/len(g)).astype(int)
mm=d.tr.values; print('F1',f1_score(d.yi[mm],pred[mm],average='macro'))
