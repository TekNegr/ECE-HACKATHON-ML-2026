import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv'); te['intensity']=np.nan
d=pd.concat([tr,te],ignore_index=True); d['tr']=d.intensity.notna()
C=['easy','moderate','hard']
# 1. unité de rang: variance des comptes
def varratio(keys):
    num=0;den=0
    for k,g in d.groupby(keys):
        N=len(g);n=g.tr.sum()
        if n<2 or N<3: continue
        for c in C:
            K=N/3; p=K/N
            hv=n*p*(1-p)*(N-n)/(N-1)
            ob=((g.intensity==c).sum()-n/3)**2
            num+=ob;den+=hv
    return num/den
for k in (['user_id'],['user_id','sport'],['user_id','year'],['user_id','sport','year']):
    print(k,varratio(k))
print(d.sport.value_counts().head(8))
# score candidates
d['alt']=d.altitude_mean_m.replace(-9999,np.nan)
d['spt']=d.n_points/d.duration_min
feats={'speed':d.speed_kmh,'dur':d.duration_min,'dist':d.distance_km,'maxs':d.max_speed_kmh,'std':d.speed_std_kmh,
'gain':d.altitude_gain_m.replace(-9999,np.nan),'gain/dist':d.altitude_gain_m.replace(-9999,np.nan)/d.distance_km.clip(.01),
'spd*dur':d.speed_kmh*d.duration_min,'spd2*dur':d.speed_kmh**2*d.duration_min,'spd2':d.speed_kmh**2,'npts':d.n_points,'spt':d.spt,
'std/speed':d.speed_std_kmh/d.speed_kmh.clip(.1),'maxs/speed':d.max_speed_kmh/d.speed_kmh.clip(.1)}
ymap={'easy':0,'moderate':1,'hard':2}
d['yi']=d.intensity.map(ymap)
def evalf(s,keys=['user_id']):
    s=pd.Series(np.asarray(s,float),index=d.index).fillna(s.median() if hasattr(s,'median') else 0)
    pred=np.zeros(len(d),int); sp=[]
    for k,g in d.groupby(keys):
        r=rankdata(s[g.index].values,method='ordinal'); N=len(g)
        # tiers: reste vers hard puis moderate
        q=np.floor((r-1)*3/N).astype(int); pred[g.index]=q
        m=g.tr.values
        if m.sum()>5 and g.yi[m].nunique()>1:
            from scipy.stats import spearmanr
            sp.append(spearmanr(s[g.index][m],g.yi[m])[0])
    m=d.tr.values
    return f1_score(d.yi[m],pred[m],average='macro'),np.nanmean(sp)
for n,s in feats.items():
    print(n,evalf(s),evalf(s,['user_id','sport']))
