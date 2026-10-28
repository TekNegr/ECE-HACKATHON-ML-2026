import sys,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0,'tools')
import drv_var as V,numpy as np
def mk(lam):
    def bal(P,target,it=50):
        w=np.ones(3)
        for _ in range(it):
            Q=P*w;Q/=Q.sum(1,keepdims=True);w*=(target/np.maximum(Q.sum(0),1e-9))**0.7
        Q=P*w**lam;return Q/Q.sum(1,keepdims=True)
    return bal
Po,Pt=V.nz(V.Bc['oof']),V.nz(V.Bc['test'])
for lam in (0.6,1.0,1.4):
    V.balance=mk(lam);print(lam,round(V.evaluate(Po,Pt,'bal'),4),flush=True)
