import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv'); te['intensity']=np.nan
d=pd.concat([tr,te],ignore_index=True); m=d.intensity.notna().values
yi=d.intensity.map({'easy':0,'moderate':1,'hard':2}).values
alt=d.altitude_mean_m.replace(-9999,np.nan); has=alt.notna().astype(float)
g=d.altitude_gain_m.replace(-9999,np.nan).fillna(0)
F=pd.DataFrame({'ls':np.log(d.speed_kmh.clip(.1)),'lms':np.log(d.max_speed_kmh.clip(.1)),'ld':np.log(d.duration_min.clip(.1)),
'ldist':np.log(d.distance_km.clip(.01)),'lstd':np.log(d.speed_std_kmh.clip(.01)),'lg':np.log1p(g),'lnp':np.log(d.n_points.clip(1)),
'gd':g/d.distance_km.clip(.1)})
F['ls2']=F.ls**2;F['ls_ld']=F.ls*F.ld;F['ls_lg']=F.ls*F.lg
sp=pd.get_dummies(d.sport).astype(float)
for c in sp: F['s_'+c]=0  # placeholder
F=F.iloc[:,:11]
# centrer par personne (sport) pour que les différences soient pertinentes
Z=(F-F.groupby(d.user_id).transform('mean'))/F.groupby(d.user_id).transform('std').replace(0,1)
Z=Z.fillna(0).values
rng=np.random.default_rng(0)
def pairs(idx):
    X=[];Y=[]
    for u,gi in d.iloc[idx].groupby('user_id').groups.items():
        gi=np.array(list(gi)); 
        for _ in range(min(60,len(gi)*3)):
            a,b=rng.choice(gi,2,replace=False)
            if yi[a]!=yi[b]: X.append(Z[a]-Z[b]); Y.append(yi[a]>yi[b])
    return np.array(X),np.array(Y)
def tiers(s,rows):
    pred=np.zeros(len(d),int)
    for u,gi in d.groupby('user_id').groups.items():
        gi=np.array(list(gi)); r=rankdata(s[gi],method='ordinal'); pred[gi]=np.floor((r-1)*3/len(gi)).astype(int)
    return pred
users=d.user_id.values; tri=np.where(m)[0]
oof=np.zeros(len(d)); 
for a,b in GroupKFold(5).split(tri,groups=users[tri]):
    X,Y=pairs(tri[a]); lr=LogisticRegression(C=1,fit_intercept=False,max_iter=500).fit(X,Y)
    oof[tri[b]]=Z[tri[b]]@lr.coef_[0]
    print('pairacc',lr.score(X,Y))
# évaluer tiers: score final test inclus avec coef plein
X,Y=pairs(tri); lr=LogisticRegression(C=1,fit_intercept=False,max_iter=500).fit(X,Y)
print(dict(zip(F.columns,lr.coef_[0].round(2))))
s=Z@lr.coef_[0]; p=tiers(s,None)
print('macroF1 tiers (in-sample coef)',f1_score(yi[m],p[m],average='macro'))
# honnête: coefficients hors-pli pour les personnes du pli, tiers calculés sur tout
s2=s.copy(); s2[tri]=np.where(True,oof[tri],0)
# test non scoré hors pli: utilise s pour test
p2=tiers(s2,None); print('macroF1 tiers (OOF coef)',f1_score(yi[m],p2[m],average='macro'))
# dominance
viol=tot=0
cols=['speed_kmh','max_speed_kmh','duration_min','distance_km','speed_std_kmh','n_points']
for (u,sp_),gg in d[m].groupby(['user_id','sport']):
    if len(gg)<10: continue
    A=gg[cols].values; y=yi[gg.index]
    for i in range(len(gg)):
        dom=(A[i]>=A).all(1); lower=dom&(y<y[i])
        # i domine j mais y_i<y_j
        dd=(A[i]>=A).all(1)&(y>y[i]); viol+=dd.sum()
        tot+=((A[i]>=A).all(1)&(y!=y[i])).sum()
print('dominance violations',viol,'/',tot)
