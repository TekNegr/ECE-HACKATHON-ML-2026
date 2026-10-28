"""Banc d'essais parallèles : une variante (modèle ou features) évaluée avec la même CV que hackathon.py.

Usage : python tools/screen.py <essai> [<essai> ...]      (liste : python tools/screen.py --list)
Chaque essai écrit ses OOF dans out/oof/<FEATURES_VERSION>/<essai>.npy (réutilisables par blend_from_cache.py)
et une ligne dans out/experiments.csv (note = description de l'essai). Threads bornés par OMP_NUM_THREADS / HACK_JOBS.
"""
import contextlib
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
src = (ROOT / "hackathon.py").read_text(encoding="utf-8")
src = src[: src.index("# Cache par version de features")]
src = src.replace(src[src.index("# ## 4. Que veut dire"): src.index("# ## 5. Familles")], "\n")
os.environ.setdefault("HACK_JOBS", "4")
g = {"__file__": str(ROOT / "hackathon.py"), "__name__": "screen"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "hackathon.py", "exec"), g)
g["N_JOBS"] = int(os.environ["HACK_JOBS"])
g["OOF_DIR"] = g["OUT"] / "oof" / g["FEATURES_VERSION"]
g["OOF_DIR"].mkdir(parents=True, exist_ok=True)

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

J, SEED = g["N_JOBS"], g["SEED"]


def hgb(**kw):
    base = dict(learning_rate=0.05, max_iter=800, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
                n_iter_no_change=40, random_state=SEED)
    base.update(kw)
    return HistGradientBoostingClassifier(**base), False


from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402


class Ordinal(BaseEstimator, ClassifierMixin):
    """Classification ordinale (Frank & Hall) : deux classifieurs cumulatifs P(y>=moderate), P(y=hard).
    P(easy)=1-P1, P(moderate)=P1-P2, P(hard)=P2 : respecte l'ordre easy < moderate < hard."""

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        order = {c: i for i, c in enumerate(g["CLASS_ORDER"])}
        t = np.array([order[v] for v in np.asarray(y)])
        self.m1_ = clone(self.base).fit(X, t >= 1)
        self.m2_ = clone(self.base).fit(X, t >= 2)
        self.classes_ = np.array(g["CLASS_ORDER"])
        return self

    def predict_proba(self, X):
        p1, p2 = self.m1_.predict_proba(X)[:, 1], self.m2_.predict_proba(X)[:, 1]
        p2 = np.minimum(p1, p2)
        P = np.c_[1 - p1, p1 - p2, p2].clip(1e-6)
        return P / P.sum(1, keepdims=True)


import numpy as np  # noqa: E402

# nom -> (description, fabrique de (estimateur, scale), réglages de globals)
EXPERIMENTS = {
    "hgb_fast": ("HGB pas 0,1 (tri rapide)", lambda: hgb(learning_rate=0.1, n_iter_no_change=20), {}),
    "hgb_nopp": ("HGB sans pentes par personne (contrôle)", lambda: hgb(learning_rate=0.1, n_iter_no_change=20), {"USE_PP": False}),
    "hgb_l63": ("HGB 63 feuilles, min_leaf 50, l2 3", lambda: hgb(max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=3.0), {}),
    "hgb_mf3": ("HGB max_features 0,3", lambda: hgb(max_features=0.3, random_state=SEED + 7), {}),
    "hgb_s2": ("HGB lent graine 2", lambda: hgb(learning_rate=0.03, max_iter=2000, max_leaf_nodes=63, min_samples_leaf=40,
                                                l2_regularization=2.0, max_features=0.5, n_iter_no_change=60,
                                                random_state=SEED + 2), {}),
    "et_big": ("ExtraTrees 1000 arbres, max_features 0,3, min_leaf 3",
               lambda: (ExtraTreesClassifier(n_estimators=1000, max_features=0.3, min_samples_leaf=3, n_jobs=J,
                                             random_state=SEED), False), {}),
    "et_sqrt": ("ExtraTrees 1000 arbres, sqrt, min_leaf 1",
                lambda: (ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=1, n_jobs=J, random_state=SEED + 3), False), {}),
    "rf_big": ("RandomForest 800 arbres, max_features 0,3, min_leaf 2",
               lambda: (RandomForestClassifier(n_estimators=800, max_features=0.3, min_samples_leaf=2, n_jobs=J,
                                               random_state=SEED), False), {}),
    "ord_hgb": ("Ordinal : 2 HGB cumulatifs", lambda: (Ordinal(hgb()[0]), False), {}),
    "ord_et": ("Ordinal : 2 ExtraTrees cumulatifs", lambda: (Ordinal(ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                                                                           n_jobs=J, random_state=SEED)), False), {}),
    "et_s2": ("ExtraTrees 500 graine 2", lambda: (ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=J,
                                                                       random_state=SEED + 2), False), {}),
    "pp2_hgb": ("HGB + pentes par personne et par personne×sport, lam 3/10/40", lambda: hgb(learning_rate=0.1, n_iter_no_change=20),
                {"PP_KEYS": ["te_us", "te_u"], "PP_LAMS": (3.0, 10.0, 40.0)}),
    "pp2_logreg": ("logreg + pentes par personne et par personne×sport, lam 3/10/40", lambda: (LogisticRegression(max_iter=3000, C=0.5), True),
                   {"PP_KEYS": ["te_us", "te_u"], "PP_LAMS": (3.0, 10.0, 40.0)}),
    "pp2_et": ("ExtraTrees + pentes par personne et par personne×sport, lam 3/10/40",
               lambda: (ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=J, random_state=SEED), False),
               {"PP_KEYS": ["te_us", "te_u"], "PP_LAMS": (3.0, 10.0, 40.0)}),
    "knnp_et": ("ExtraTrees + k-NN par personne (k 10/30)",
                lambda: (ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=J, random_state=SEED), False),
                {"USE_PPKNN": True}),
    "knnp_logreg": ("logreg + k-NN par personne (k 10/30)", lambda: (LogisticRegression(max_iter=3000, C=0.5), True),
                    {"USE_PPKNN": True}),
    "logreg_c2": ("Régression logistique C=2", lambda: (LogisticRegression(max_iter=4000, C=2.0), True), {}),
}

if "--list" in sys.argv or len(sys.argv) < 2:
    for k, (d, _, _) in EXPERIMENTS.items():
        print(f"{k:12s} {d}")
    sys.exit(0)

for name in sys.argv[1:]:
    desc, make, overrides = EXPERIMENTS[name]
    saved = {k: g[k] for k in overrides}
    g.update(overrides)
    g["MODELS"][name] = make()
    g["run_cv"](name, note=desc)
    g.update(saved)
