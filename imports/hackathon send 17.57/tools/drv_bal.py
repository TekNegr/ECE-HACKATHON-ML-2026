import sys,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0,'tools')
import drv_var as V,numpy as np,pandas as pd
nz=V.nz;A,Bc=V.A,V.Bc
Po,Pt=V.nz(Bc['oof']),V.nz(Bc['test'])
for seed in (11,22):
    print('seed',seed,'caps',round(V.evaluate(Po,Pt,'caps',seed=seed),4),'bal',round(V.evaluate(Po,Pt,'bal',seed=seed),4),'argmax-only?',flush=True)
# candidate: balance on test only (known = all train)
tr,te=V.tr,V.te
kn=pd.crosstab(tr.user_id.values,V.y).reindex(columns=range(3),fill_value=0)
for nm,P in (('c4',Pt),('mixarith',nz(A['test']+Bc['test']))):
    pred=np.zeros(len(te),int)
    for u,idx in pd.Series(np.arange(len(te))).groupby(te.user_id.values).indices.items():
        k=kn.reindex([u]).fillna(0).values[0].astype(int)
        ex=V.exact_counts(len(idx),V.Nu[u],k)
        pred[idx]=V.balance(P[idx],ex.astype(float)).argmax(1)
    sub=pd.DataFrame({'workout_id':te.workout_id,'intensity':np.array(V.CL)[pred]})
    s=pd.read_csv('data/sample_submission.csv');assert (sub.workout_id.values==s.workout_id.values).all() and list(sub.columns)==list(s.columns)
    sub.to_csv(f'out/drv/sub_bal_{nm}.csv',index=False);print(nm,sub.intensity.value_counts(normalize=True).round(3).to_dict())
