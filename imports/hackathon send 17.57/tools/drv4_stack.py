import os, sys
os.environ["OMP_NUM_THREADS"]="3"
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from pathlib import Path
R=Path(__file__).resolve().parents[1]; CL=["easy","moderate","hard"]
tr=pd.read_csv(R/"data/train.csv"); te=pd.read_csv(R/"data/test.csv"); y=tr.intensity.values
M={"v3-ligne-de-base-locale":"extratrees,hgb,hgb_slow,knn,logreg,rf","v4-effort-heure-relative":"extratrees,hgb,hgb_slow,rf","v6-pentes-par-personne":"extratrees,hgb,logreg"}
O=[];T=[];TF=[]
for v,ms in M.items():
    for m in ms.split(","):
        O.append(np.load(R/f"out/oof/{v}/{m}.npy")); tf=np.load(R/f"out/oof/{v}/{m}_testfolds.npy"); TF.append(tf)
L=lambda a:np.log(np.clip(a,1e-6,1))
Ftr=np.hstack([L(o) for o in O]); nf=TF[0].shape[0]
Fte=[np.hstack([L(t[i]) for t in TF]) for i in range(nf)]
# méta-features (sans labels)
al=pd.concat([tr[["user_id","sport","year","month"]].assign(s="tr"),te[["user_id","sport","year","month"]].assign(s="te")],ignore_index=True)
al["t"]=al.year*12+al.month
al["n"]=al.groupby("user_id").user_id.transform("size")
al["trank"]=al.groupby("user_id").t.rank(pct=True)
sports=sorted(al.sport.unique()); S=np.stack([(al.sport==s).values.astype(float) for s in sports],1)
ntr=len(tr)
# p moyen (hard-easy) par modèle-moyenne
pm_tr=np.mean(O,0); pm_te=np.mean([np.mean([t[i] for t in TF],0) for i in range(nf)],0)
d=np.concatenate([pm_tr[:,2]-pm_tr[:,0],pm_te[:,2]-pm_te[:,0]]); al["d"]=d
al["drank"]=al.groupby("user_id").d.rank(pct=True)
al["drank_sp"]=al.groupby(["user_id","sport"]).d.rank(pct=True)
al["dz"]=(al.d-al.groupby("user_id").d.transform("mean"))/(al.groupby("user_id").d.transform("std").fillna(1)+1e-3)
logn=np.log(al.n.values)[:,None]
A=np.hstack([S,logn,al[["trank"]].values,S*al[["trank"]].values])
B=np.hstack([A,al[["drank","drank_sp","dz"]].values])
dl=(Ftr[:,-3:]) 
def build(extra): 
    return extra
def run(name,extra,kind,C=0.1):
    X=np.hstack([Ftr,extra[:ntr]]) if extra is not None else Ftr
    Xt=[np.hstack([f,extra[ntr:]]) if extra is not None else f for f in Fte]
    # standardise extras for LR
    mu=X.mean(0);sd=X.std(0)+1e-6
    if kind=="lr": X=(X-mu)/sd; Xt=[(x-mu)/sd for x in Xt]
    mk=(lambda: LogisticRegression(C=C,max_iter=3000)) if kind=="lr" else (lambda: HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=400,early_stopping=True,validation_fraction=0.15,n_iter_no_change=30,random_state=0))
    oof=np.zeros((ntr,3))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=7).split(X,y):
        m=mk().fit(X[a],y[a]); oof[b]=m.predict_proba(X[b])[:,[list(m.classes_).index(c) for c in CL]]
    m=mk().fit(X,y); ix=[list(m.classes_).index(c) for c in CL]
    pt=np.mean([m.predict_proba(x)[:,ix] for x in Xt],0)
    sc=f1_score(y,np.array(CL)[oof.argmax(1)],average="macro"); print(name,round(sc,4),flush=True)
    np.savez(R/f"out/blend_drv4_{name}.npz",oof=oof,test=pt,classes=np.array(CL)); return sc
#run("base",None,"lr",0.1)
#run("A",A,"lr",0.03)
#run("B",B,"lr",0.03)
#run("Bhgb",B,"hgb")
# idée 2 : log-probas centrées par personne (moyenne sur train+test) + rangs des 3 classes (moyenne modèles)
P=np.vstack([pm_tr,pm_te]); lp=L(P)
cen=lp-pd.DataFrame(lp).groupby(al.user_id.values).transform("mean").values
rk=np.stack([pd.Series(P[:,k]).groupby(al.user_id.values).rank(pct=True).values for k in range(3)],1)
run("C",np.hstack([cen,rk]),"lr",0.03)
run("D",np.hstack([cen]),"lr",0.03)
