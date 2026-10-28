import numpy as np,pandas as pd
tr=pd.read_csv('data/train.csv');te=pd.read_csv('data/test.csv')
tr['t']=0;te['t']=1
d=pd.concat([tr,te],ignore_index=True)
d['ym']=d.year*12+d.month
print(np.corrcoef(d.workout_id,d.ym)[0,1])
y=d.intensity.map({'easy':0,'moderate':1,'hard':2})
d['y']=y
# neighbor by workout_id order within user, and by (ym) order
for key in (['workout_id'],['ym','workout_id']):
    s=d.sort_values(['user_id']+key).reset_index(drop=True)
    same=[];tot=0
    prev=s.groupby('user_id').y.shift(1);nxt=s.groupby('user_id').y.shift(-1)
    m=prev.notna()&s.y.notna()
    print(key,'adj same',(prev[m]==s.y[m]).mean(), 'n',m.sum())
    # chance baseline
    pc=s.y.value_counts(normalize=True);print((pc**2).sum())
# test position
print(d.groupby('t').workout_id.describe())
print(d.groupby('t').ym.describe())
