"""drv3 : features supplémentaires (charge, rangs transductifs croisés) + HGB/ET, mêmes plis que hackathon.py.
Usage : python tools/drv3_feat.py <nom_oof> <hgb|et|hgbs>"""
import contextlib, io, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
src = (ROOT / "hackathon.py").read_text(encoding="utf-8")
src = src[: src.index("# Cache par version de features")]
src = src.replace(src[src.index("# ## 4. Que veut dire"): src.index("# ## 5. Familles")], "\n")
os.environ.setdefault("HACK_JOBS", "4")
g = {"__file__": str(ROOT / "hackathon.py"), "__name__": "drv3"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "hackathon.py", "exec"), g)
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
g["N_JOBS"] = 4
name, kind = sys.argv[1], sys.argv[2]
extra = sys.argv[3] if len(sys.argv) > 3 else "all"
g["OOF_DIR"] = g["OUT"] / "oof" / "drv3-" + "" if False else g["OUT"] / "oof" / f"drv3-{name}"
g["OOF_DIR"].mkdir(parents=True, exist_ok=True)
X, XT = g["X"], g["X_test"]
A = pd.concat([X, XT], ignore_index=True)
tr, te = g["train"], g["test"]
raw = pd.concat([tr.drop(columns=[g["TARGET"]]), te], ignore_index=True)
U, S = g["USER"], g["SPORT"]
F = pd.DataFrame(index=A.index)
sp, du, di = raw[g["SPEED_AVG"]], raw[g["DURATION"]], raw[g["DISTANCE"]]
eff = A["effort_speed"]
key = raw[U].astype(str) + "|" + raw[S].astype(str)
tm = A["t_month"]
# 1. charge : speed relative x duree (TRIMP-like), rang par personne et personne x sport
rel_us = sp / sp.groupby(key).transform("mean")
rel_eff = eff / eff.groupby(key).transform("mean")
F["trimp"] = rel_us * du
F["trimp_eff"] = rel_eff * du
F["trimp_dist"] = rel_eff * di
for c in ["trimp", "trimp_eff", "trimp_dist"]:
    F[c + "_rk_u"] = F[c].groupby(raw[U]).rank(pct=True)
    F[c + "_rk_us"] = F[c].groupby(key).rank(pct=True)
# 2. rang de composite : moyenne des rangs perso x sport de vitesse/effort/vmax/duree
rk = lambda s, k: s.groupby(k).rank(pct=True)
comp = pd.concat([rk(sp, key), rk(eff, key), rk(raw[g["SPEED_MAX"]], key), rk(du, key), rk(di, key)], axis=1)
F["comp_mean_us"] = comp.mean(1)
F["comp_speed_eff_us"] = comp.iloc[:, :3].mean(1)
F["comp_rk_u"] = F["comp_mean_us"].groupby(raw[U]).rank(pct=True)
F["comp_rk_us2"] = F["comp_mean_us"].groupby(key).rank(pct=True)
# rang parmi les séances de même sport ET même durée (tertile de durée perso)
dq = pd.qcut(du.groupby(key).rank(pct=True), 3, labels=False, duplicates="drop")
k2 = key + "|" + dq.astype(str)
F["eff_rk_usd"] = rk(eff, k2); F["sp_rk_usd"] = rk(sp, k2)
# 3. charge d'entrainement récente (sans label) : volume personne sur mois-1, -2, -3 et ratio au mois courant
d = pd.DataFrame({"u": raw[U], "m": tm, "dur": du, "n": 1.0})
mon = d.groupby(["u", "m"])[["dur", "n"]].sum()
def lag(l, col):
    idx = pd.MultiIndex.from_arrays([d["u"], d["m"] - l])
    return mon[col].reindex(idx).fillna(0).values
for l in (1, 2, 3):
    F[f"load_dur_l{l}"] = lag(l, "dur"); F[f"load_n_l{l}"] = lag(l, "n")
F["load_dur_l123"] = F[["load_dur_l1", "load_dur_l2", "load_dur_l3"]].sum(1)
F["load_dur_now"] = lag(0, "dur")
F["load_ratio"] = (F["load_dur_now"] + 1) / (F["load_dur_l123"] / 3 + 1)
umean = mon.groupby("u")["dur"].transform("mean")
# 4. rang parmi les séances du même mois et personne (transductif)
um = raw[U].astype(str) + "|" + tm.astype(str)
F["eff_rk_um"] = rk(eff, um); F["sp_rk_um"] = rk(sp, um)
F["comp_rk_um"] = rk(F["comp_mean_us"], um)
F = F.replace([np.inf, -np.inf], np.nan)
if extra != "all":
    keep = {"trimp": ["trimp", "trimp_eff", "trimp_dist"], }
    sel = {"load": [c for c in F if c.startswith("load_")], "comp": [c for c in F if c.startswith(("comp_", "eff_rk", "sp_rk"))],
           "trimp": [c for c in F if c.startswith("trimp")]}
    F = F[sel[extra]]
n = len(X)
X2 = pd.concat([X, F.iloc[:n].reset_index(drop=True)], axis=1)
XT2 = pd.concat([XT, F.iloc[n:].reset_index(drop=True)], axis=1)
g["X"], g["X_test"] = X2, XT2
g["NUM_COLS"] = g["NUM_COLS"] + list(F.columns)
g["PP_FEATS"] = g["PP_FEATS"]
J, SEED = 4, g["SEED"]
if kind == "hgb":
    est = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=800, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, n_iter_no_change=40, random_state=SEED)
elif kind == "hgbs":
    est = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=2000, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=2.0, max_features=0.5, early_stopping=True, n_iter_no_change=60, random_state=SEED + 1)
else:
    est = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=J, random_state=SEED)
g["MODELS"][kind] = (est, False)
g["run_cv"](kind, note=f"drv3 {name} {extra}")
os.replace(g["OOF_DIR"] / f"{kind}.npy", g["OOF_DIR"] / f"{kind}.npy")
