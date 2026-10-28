"""drv3 : score latent par régression HGB du code 0/1/2 (OOF, mêmes plis), puis rang du score parmi toutes les séances
de la personne (train OOF + test moyenne des plis), calibré par régression logistique (CV) -> probas.
Sorties : out/oof/drv3-latent/lat.npy + lat_testfolds.npy"""
import contextlib, io, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
src = (ROOT / "hackathon.py").read_text(encoding="utf-8")
src = src[: src.index("# Cache par version de features")]
src = src.replace(src[src.index("# ## 4. Que veut dire"): src.index("# ## 5. Familles")], "\n")
os.environ.setdefault("HACK_JOBS", "4")
g = {"__file__": str(ROOT / "hackathon.py"), "__name__": "drv3l"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "hackathon.py", "exec"), g)
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
X, XT, y = g["X"], g["X_test"], g["y"]
order = {c: i for i, c in enumerate(g["CLASS_ORDER"])}
t = y.map(order).values.astype(float)
CL = g["CLASSES"]
n, nt = len(X), len(XT)
oofs = np.zeros(n); tfs = np.zeros((g["N_SPLITS"], nt))
for i, (tr, va) in enumerate(g["CV"].split(X, y, g["GROUPS"])):
    est = HistGradientBoostingRegressor(learning_rate=0.08, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0,
                                        early_stopping=True, n_iter_no_change=30, random_state=g["SEED"])
    pipe = Pipeline([("prep", g["make_preprocessor"](False)), ("clf", est)])
    # y numérique pour le régresseur ; le TargetEncoder multiclasse a besoin des étiquettes : on passe par la pipeline avec y texte
    prep = g["make_preprocessor"](False).fit(X.iloc[tr], y.iloc[tr])
    est.fit(prep.transform(X.iloc[tr]), t[tr])
    oofs[va] = est.predict(prep.transform(X.iloc[va]))
    tfs[i] = est.predict(prep.transform(XT))
    print(i, flush=True)
U = pd.concat([g["train"][g["USER"]], g["test"][g["USER"]]], ignore_index=True)
S = pd.concat([g["train"][g["SPORT"]], g["test"][g["SPORT"]]], ignore_index=True)
def feats(sc_tr, sc_te):
    s = pd.Series(np.r_[sc_tr, sc_te])
    r = s.groupby(U.values).rank(pct=True).values
    r2 = s.groupby(U.astype(str).values + S.astype(str).values).rank(pct=True).values
    zs = ((s - s.groupby(U.values).transform("mean")) / s.groupby(U.values).transform("std")).values
    return np.c_[s.values, r, r2, zs]
oof_p = np.zeros((n, 3))
Fo = feats(oofs, tfs.mean(0))
yy = y.values
for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(Fo[:n], yy):
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Fo[tr], yy[tr])
    oof_p[va] = lr.predict_proba(Fo[va])[:, [list(lr.classes_).index(c) for c in CL]]
lr = LogisticRegression(C=1.0, max_iter=2000).fit(Fo[:n], yy)
idx = [list(lr.classes_).index(c) for c in CL]
tf = np.zeros((g["N_SPLITS"], nt, 3))
for i in range(g["N_SPLITS"]):
    Fi = feats(oofs, tfs[i])
    tf[i] = lr.predict_proba(Fi[n:])[:, idx]
print("corr score/code", np.corrcoef(oofs, t)[0, 1])
print("macro-F1 latent calibre", f1_score(yy, np.array(CL)[oof_p.argmax(1)], average="macro"))
d = ROOT / "out" / "oof" / "drv3-latent"; d.mkdir(parents=True, exist_ok=True)
np.save(d / "lat.npy", oof_p); np.save(d / "lat_testfolds.npy", tf)
