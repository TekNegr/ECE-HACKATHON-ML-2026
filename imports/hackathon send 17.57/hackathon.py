# %% [markdown]
# # Hackathon ML 2026 : cette séance sera-t-elle difficile pour cette personne ?
#
# Pipeline unique : chargement, choix du schéma de validation, features relatives à la personne,
# comparaison de familles de modèles en validation croisée (macro-F1 out-of-fold), ensemble,
# analyse d'erreur, soumission.
#
# Exécution : `python hackathon.py` (script) ou `python tools/export_notebook.py` pour le notebook rendu.
# Variables d'environnement : `HACK_MODELS=logreg,hgb` (modèles à évaluer), `HACK_SUBMIT=0` (pas de fichier de soumission),
# `HACK_EXPLAIN=1` (importance par permutation), `HACK_DATA` / `HACK_OUT` (dossiers).

# %%
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

show = globals().get("display", print)

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA = Path(os.environ.get("HACK_DATA", ROOT / "data" if (ROOT / "data" / "train.csv").exists() else ROOT))
OUT = Path(os.environ.get("HACK_OUT", ROOT / "out"))
SUBS = OUT.parent / "submissions" if "HACK_OUT" in os.environ else ROOT / "submissions"
for d in (OUT / "figures", OUT / "oof", OUT / "test_proba", SUBS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
ID, TARGET = "workout_id", "intensity"
CLASS_ORDER = ["easy", "moderate", "hard"]
FEATURES_VERSION = "v7-pentes-personne-et-sport"  # incrémenter à chaque changement de features (tracé dans experiments.csv)
RUN_MODELS = os.environ.get("HACK_MODELS", "logreg,knn,rf,extratrees,hgb").split(",")
N_JOBS = max(1, (os.cpu_count() or 2) - 1)
try:  # priorité basse : les runs parallèles prennent toute la machine sans la figer
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
except Exception:
    pass

# Si la détection automatique des colonnes se trompe, forcer ici : {"USER": "user_id", "SPORT": "sport", ...}
OVERRIDE: dict[str, str | None] = {"ELEV": "altitude_gain_m", "SPEED_AVG": "speed_kmh", "START": "start_hour_utc"}

# %% [markdown]
# ## 1. Données et schéma

# %%
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
for _df in (train, test):  # capteur absent codé -9999 : valeur manquante, pas une altitude
    for _c in [c for c in _df.columns if "altitude" in c.lower()]:
        _df.loc[_df[_c] <= -999, _c] = np.nan
sample = pd.read_csv(DATA / "sample_submission.csv")
print("train", train.shape, "| test", test.shape, "| sample", sample.shape)
show(train.head())
print(train.dtypes.to_string())
print("\nValeurs manquantes (train) :\n", train.isna().mean().round(3)[lambda s: s > 0].to_string() or "aucune")
print("\nCible :\n", train[TARGET].value_counts(normalize=True).round(3).to_string())


# %%
def find_col(df: pd.DataFrame, keys: list[str], numeric: bool | None = None, exclude: tuple = ()) -> str | None:
    """Première colonne dont le nom contient une clé (les clés sont testées par ordre de priorité)."""
    for k in keys:
        for c in df.columns:
            if c in (ID, TARGET) or c in exclude:
                continue
            if k in c.lower() and (numeric is None or pd.api.types.is_numeric_dtype(df[c]) == numeric):
                return c
    return None


USER = find_col(train, ["user", "utilisateur", "athlete", "person"])
SPORT = find_col(train, ["sport", "activity", "activite"])
GENDER = find_col(train, ["gender", "genre", "sex"])
START = find_col(train, ["start", "debut", "début", "timestamp", "date", "heure", "hour"], exclude=(USER,))
DURATION = find_col(train, ["duration", "duree", "durée", "elapsed", "moving_time"], numeric=True)
DISTANCE = find_col(train, ["distance", "dist"], numeric=True)
ELEV = find_col(train, ["elev", "ascent", "denivel", "dénivel", "climb", "altitude"], numeric=True)
SPEED_COLS = [c for c in train.columns if any(k in c.lower() for k in ("speed", "vitesse", "pace", "allure"))
              and pd.api.types.is_numeric_dtype(train[c])]
SPEED_MAX = next((c for c in SPEED_COLS if "max" in c.lower()), None)
SPEED_AVG = next((c for c in SPEED_COLS if any(k in c.lower() for k in ("avg", "mean", "moy"))), None)
for _k, _v in OVERRIDE.items():
    globals()[_k] = _v

print({"USER": USER, "SPORT": SPORT, "GENDER": GENDER, "START": START, "DURATION": DURATION,
       "DISTANCE": DISTANCE, "ELEV": ELEV, "SPEED_AVG": SPEED_AVG, "SPEED_MAX": SPEED_MAX})

# %% [markdown]
# ## 2. Schéma de validation : la CV doit imiter le découpage train/test
#
# Si les utilisateurs du test sont aussi dans le train, on peut apprendre « qui » est la personne
# (encodage cible par utilisateur, validé en StratifiedKFold). Sinon, on valide par groupe d'utilisateurs
# et l'identité de l'utilisateur n'apporte rien.

# %%
user_overlap = float(test[USER].isin(set(train[USER])).mean()) if USER else 0.0
USE_USER_TE = USER is not None and user_overlap >= 0.5
if USER:
    print(f"Utilisateurs : {train[USER].nunique()} (train), {test[USER].nunique()} (test), "
          f"séances test dont l'utilisateur est connu : {user_overlap:.1%}")
    print("Séances par utilisateur (train) :", train[USER].value_counts().describe().round(1).to_dict())
if USE_USER_TE:
    CV, GROUPS = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED), None
    print("=> StratifiedKFold + encodage cible par utilisateur (cross-fitted)")
else:
    CV = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED) if USER else StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    GROUPS = train[USER] if USER else None
    print("=> validation groupée par utilisateur, pas d'encodage cible utilisateur")

# %% [markdown]
# ## 2b. Hypothèse « difficile = tiers par personne » (prémisse mesurée, pas supposée)
#
# Le briefing dit que l'intensité est mesurée séparément pour chaque personne et que les classes sont équilibrées.
# Si les labels sont des tiers calculés par utilisateur, la part de chaque classe par utilisateur est beaucoup plus
# serrée autour de 1/3 qu'un tirage aléatoire (binomial). Ce test décide si le post-traitement par quotas (6b) a un sens.

# %%
PER_USER_TERCILES = False
if USER:
    _counts = pd.crosstab(train[USER], train[TARGET])
    _n = _counts.sum(1)
    _big = _n >= 9
    _prop = _counts[_big].div(_n[_big], axis=0)
    _obs_sd = float(_prop.std().mean())
    _binom_sd = float(np.sqrt((1 / 3) * (2 / 3) / _n[_big]).mean())
    PER_USER_TERCILES = _obs_sd < 0.6 * _binom_sd
    print(f"Écart-type observé de la part de chaque classe par utilisateur : {_obs_sd:.3f} "
          f"vs attendu si aléatoire : {_binom_sd:.3f} => tiers par utilisateur : {PER_USER_TERCILES}")
    show(_prop.describe().round(3))


# %% [markdown]
# ## 3. Features : « difficile » est relatif à la personne
#
# Une même vitesse est facile pour l'un, difficile pour l'autre. On décrit donc chaque séance par rapport
# aux habitudes de son utilisateur (z-score, ratio, rang dans son historique), globalement et par sport.
# Les statistiques sont calculées sur train + test sans la cible (aucune fuite de label, aucune donnée externe).


# %%
def safe_div(a, b):
    return a / b.replace(0, np.nan)


def time_features(df: pd.DataFrame) -> pd.DataFrame:
    if START is None:
        return pd.DataFrame(index=df.index)
    s = df[START]
    f = pd.DataFrame(index=df.index)
    if pd.api.types.is_numeric_dtype(s) and s.max() <= 48:
        hour = s.astype(float)
        ts = None
    else:
        if pd.api.types.is_numeric_dtype(s):
            ts = pd.to_datetime(s, unit="ms" if s.max() > 1e12 else "s", errors="coerce")
        else:
            ts = pd.to_datetime(s, errors="coerce", format="mixed")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
        hour = ts.dt.hour + ts.dt.minute / 60
        if ts.dt.normalize().nunique() > 1:  # vraie date disponible (sinon heure seule)
            f["dow"] = ts.dt.dayofweek
            f["month"] = ts.dt.month
        else:
            ts = None
    f["hour"] = hour
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    if ts is not None and USER:
        # Récupération : temps depuis la séance précédente du même utilisateur.
        days = (ts - ts.min()).dt.total_seconds() / 86400
        order = days.groupby(df[USER]).rank(method="first")
        tmp = pd.DataFrame({"u": df[USER], "d": days}).sort_values(["u", "d"])
        f["days_since_prev"] = tmp.groupby("u")["d"].diff().reindex(df.index)
        f["days_to_next"] = (-tmp.groupby("u")["d"].diff(-1)).reindex(df.index)
        f["session_index_user"] = order
    return f


def derived_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    if DISTANCE and DURATION:
        f["speed_calc"] = safe_div(df[DISTANCE], df[DURATION])
    if ELEV and DISTANCE:
        f["elev_per_dist"] = safe_div(df[ELEV], df[DISTANCE])
    if ELEV and DURATION:
        f["elev_per_time"] = safe_div(df[ELEV], df[DURATION])
    if SPEED_MAX and SPEED_AVG:
        f["speed_max_over_avg"] = safe_div(df[SPEED_MAX], df[SPEED_AVG])
    if "year" in df and "month" in df:
        f["t_month"] = df["year"] * 12 + df["month"]  # mois absolu : position dans l'historique
    if DISTANCE and DURATION:
        # Vitesse équivalente-effort : 100 m de montée comptent comme 1 km de plat (règle de Naismith simplifiée).
        f["effort_speed"] = safe_div(df[DISTANCE] + (df[ELEV].fillna(0) / 100 if ELEV else 0), df[DURATION] / 60)
    if "weekday" in df:
        f["weekend"] = (df["weekday"] >= 5).astype(int)
    return f


def load_features(df: pd.DataFrame) -> pd.DataFrame:
    """Charge d'entraînement (sans cible) : volume du mois et de l'année, progression dans l'historique."""
    f = pd.DataFrame(index=df.index)
    if USER is None or "t_month" not in df:
        return f
    for keys, tag in [([USER, "t_month"], "um"), ([USER, SPORT, "t_month"], "usm"), ([USER, SPORT, "year"], "usyL")]:
        g = df.groupby(keys, observed=True, dropna=False)
        f[f"n_{tag}"] = g[USER].transform("size")
        if DURATION:
            f[f"dur_sum_{tag}"] = g[DURATION].transform("sum")
    f["t_rank_us"] = df.groupby([USER, SPORT], observed=True)["t_month"].rank(pct=True)
    f["sess_idx_us"] = df.groupby([USER, SPORT], observed=True)["t_month"].rank(method="first")
    if START and pd.api.types.is_numeric_dtype(df[START]) and df[START].max() <= 48:
        # Heure UTC : le fuseau de chacun est inconnu, on la compare à l'heure habituelle de la personne.
        ang = 2 * np.pi * df[START] / 24
        mean_ang = np.arctan2(np.sin(ang).groupby(df[USER]).transform("mean"), np.cos(ang).groupby(df[USER]).transform("mean"))
        f["hour_rel_user"] = (((ang - mean_ang) + np.pi) % (2 * np.pi) - np.pi) * 24 / (2 * np.pi)
    f = pd.concat([f, local_baseline(df)], axis=1)
    return f


LOCAL_WINDOW = 2  # mois de part et d'autre


def local_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """La forme progresse : on compare la séance aux autres séances de la même personne dans ce sport
    sur une fenêtre de +/- LOCAL_WINDOW mois (séance elle-même exclue). Sans cible : train + test autorisés."""
    f = pd.DataFrame(index=df.index)
    keys = [USER, SPORT]
    cols = [c for c in (SPEED_AVG, SPEED_MAX, "speed_calc", "effort_speed", DURATION, DISTANCE, ELEV, "elev_per_dist") if c and c in df]
    m = df.groupby(keys + ["t_month"], observed=True)[cols].agg(["sum", "count"])
    m.columns = [f"{a}__{b}" for a, b in m.columns]
    m = m.reset_index()
    acc = None
    for d in range(-LOCAL_WINDOW, LOCAL_WINDOW + 1):
        sh = m.assign(t_month=m["t_month"] + d).set_index(keys + ["t_month"])
        acc = sh if acc is None else acc.add(sh, fill_value=0)
    acc = acc.reset_index()
    rows = df[keys + ["t_month"]].merge(acc, on=keys + ["t_month"], how="left")
    rows.index = df.index
    us_std = df.groupby(keys, observed=True)[cols].transform("std")
    us_mean = df.groupby(keys, observed=True)[cols].transform("mean")
    for c in cols:
        S, C = rows[f"{c}__sum"], rows[f"{c}__count"]
        local = (S - df[c].fillna(0)) / (C - 1).where(C > 1)
        f[f"loc_n_{c}"] = C
        f[f"loc_diff_{c}"] = (df[c] - local) / (us_std[c] + 1e-9)
        f[f"loc_ratio_{c}"] = safe_div(df[c], local)
        f[f"loc_level_{c}"] = safe_div(local, us_mean[c])  # forme du moment vs moyenne de la personne
    return f


def relative_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    groupings = [([USER], "u"), ([USER, SPORT], "us"), ([USER, SPORT, "year"], "usy"), ([SPORT], "s"), ([SPORT, GENDER], "sg")]
    for keys, tag in groupings:
        if any(k is None for k in keys):
            continue
        g = df.groupby(keys, observed=True, dropna=False)
        out[f"n_{tag}"] = g[keys[0]].transform("size")
        for c in cols:
            m, sd = g[c].transform("mean"), g[c].transform("std")
            out[f"{c}__z_{tag}"] = safe_div(df[c] - m, sd)
            out[f"{c}__ratio_{tag}"] = safe_div(df[c], m)
            if tag in ("u", "us", "usy"):
                out[f"{c}__rank_{tag}"] = g[c].rank(pct=True)
    return pd.DataFrame(out, index=df.index)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    all_df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    base = all_df.drop(columns=[c for c in (ID, START) if c], errors="ignore")
    num = [c for c in base.select_dtypes("number").columns if c != USER]
    feats = pd.concat([base[num], time_features(all_df), derived_features(all_df)], axis=1)
    feats = pd.concat([feats, load_features(pd.concat([feats, all_df[[c for c in (USER, SPORT, START) if c and c not in feats]]], axis=1))], axis=1)
    calendar = {"year", "month", "weekday", "dow", "hour", "hour_sin", "hour_cos", "session_index_user", "t_month", "t_rank_us", "weekend", "hour_rel_user", START}
    rel_cols = [c for c in feats.columns if c not in calendar and not c.startswith(("n_", "dur_sum_", "loc_", "sess_idx"))]
    keys = pd.concat([feats[["year"]] if "year" in feats else pd.DataFrame(index=feats.index),
                      all_df[[c for c in (USER, SPORT, GENDER) if c]]], axis=1)
    feats = pd.concat([feats, relative_features(pd.concat([feats[rel_cols], keys], axis=1), rel_cols)], axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    cat = [c for c in base.columns if c not in num and c != USER and base[c].nunique() <= 50]
    num_cols = list(feats.columns)
    # Clés d'encodage cible hiérarchique : la personne, puis la personne dans ce sport, cette année, ce mois.
    te = pd.DataFrame(index=all_df.index)
    if USER:
        parts = {"u": [USER], "us": [USER, SPORT], "usy": [USER, SPORT, "year"], "usm": [USER, SPORT, "t_month"],
                 "um": [USER, "t_month"]}
        src = pd.concat([all_df[[c for c in (USER, SPORT) if c]], feats[[c for c in ("year", "t_month") if c in feats]]], axis=1)
        for tag, cols in parts.items():
            if all(c in src for c in cols):
                te[f"te_{tag}"] = src[cols].astype(str).agg("|".join, axis=1)
    X_all = pd.concat([feats, all_df[cat + ([USER] if USER else [])], te], axis=1)
    return (X_all.iloc[: len(train)].reset_index(drop=True), X_all.iloc[len(train):].reset_index(drop=True),
            num_cols, cat, list(te.columns))


X, X_test, NUM_COLS, CAT_COLS, TE_COLS = build_features(train, test)
y = train[TARGET].astype(str).reset_index(drop=True)
CLASSES = [c for c in CLASS_ORDER if c in set(y)] or sorted(y.unique())
print(f"{len(NUM_COLS)} features numériques, catégorielles : {CAT_COLS}, encodage utilisateur : {USE_USER_TE}")

# %% [markdown]
# ## 4. Que veut dire « difficile » ? (information mutuelle avec la cible)

# %%
_rng = np.random.default_rng(SEED)
_idx = _rng.choice(len(X), size=min(10000, len(X)), replace=False)
_Xmi = SimpleImputer(strategy="median").fit_transform(X.iloc[_idx][NUM_COLS])
mi = pd.Series(mutual_info_classif(_Xmi, y.iloc[_idx], random_state=SEED), index=NUM_COLS).sort_values(ascending=False)
print("Top 25 features par information mutuelle :\n", mi.head(25).round(4).to_string())
if SPORT:
    print("\nIntensité par sport :\n", pd.crosstab(train[SPORT], train[TARGET], normalize="index").round(2).to_string())

# %% [markdown]
# ## 5. Familles de modèles comparées en validation croisée
#
# Chaque modèle est un pipeline complet (imputation, encodage, modèle) refit dans chaque pli : pas de fuite.
# Les probabilités out-of-fold sont sauvegardées pour l'ensemble et l'analyse d'erreur ; chaque run est tracé
# dans `out/experiments.csv`.


# %%
from sklearn.base import BaseEstimator, TransformerMixin


class PerPersonRidge(BaseEstimator, TransformerMixin):
    """Chaque personne a sa propre sensibilité (la durée compte plus pour l'un, la vitesse pour l'autre).
    Régression ridge du code d'intensité (0, 1, 2) par (personne, sport), rétrécie vers le modèle global
    (pénalité lam vers les coefficients globaux). Sortie : un score d'intensité par séance.
    Anti-fuite par construction, comme TargetEncoder : fit_transform renvoie des scores cross-fitted
    (chaque ligne prédite par un modèle qui ne l'a pas vue), transform utilise le modèle appris sur tout le train."""

    def __init__(self, lams=(10.0, 40.0), cv=5, random_state=0):
        self.lams, self.cv, self.random_state = lams, cv, random_state

    def _prep(self, Xs):
        A = Xs.iloc[:, 1:].astype(float).replace([np.inf, -np.inf], np.nan)
        A = ((A.fillna(self.med_) - self.mu_) / self.sd_).clip(-5, 5).values
        return Xs.iloc[:, 0].astype(str).values, np.c_[np.ones(len(A)), A]

    @staticmethod
    def _ridge(Z, t, lam, prior):
        R = lam * np.eye(Z.shape[1]); R[0, 0] = 1e-6
        return prior + np.linalg.solve(Z.T @ Z + R, Z.T @ (t - Z @ prior))

    def _fit_models(self, keys, Z, t):
        bg = self._ridge(Z, t, 1.0, np.zeros(Z.shape[1]))
        per = {}
        for k, idx in pd.Series(np.arange(len(keys))).groupby(keys).indices.items():
            per[k] = [self._ridge(Z[idx], t[idx], lam, bg) for lam in self.lams]
        return bg, per

    def _score(self, keys, Z, models):
        bg, per = models
        out = np.zeros((len(Z), len(self.lams)))
        for k, idx in pd.Series(np.arange(len(keys))).groupby(keys).indices.items():
            bs = per.get(k, [bg] * len(self.lams))
            out[idx] = np.column_stack([Z[idx] @ b for b in bs])
        return out

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def _width(self):
        return len(self.lams)

    def fit_transform(self, X, y):
        A = X.iloc[:, 1:].astype(float).replace([np.inf, -np.inf], np.nan)
        self.med_ = A.median(); self.mu_ = A.fillna(self.med_).mean(); self.sd_ = A.fillna(self.med_).std() + 1e-9
        keys, Z = self._prep(X)
        t = pd.Series(np.asarray(y)).map({c: i for i, c in enumerate(CLASS_ORDER)}).astype(float).values
        out = np.zeros((len(Z), self._width()))
        for tr, va in StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state).split(Z, t):
            out[va] = self._score(keys[va], Z[va], self._fit_models(keys[tr], Z[tr], t[tr]))
        self.models_ = self._fit_models(keys, Z, t)
        return out

    def transform(self, X):
        keys, Z = self._prep(X)
        return self._score(keys, Z, self.models_)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"pp_score_lam{l:g}" for l in self.lams])


class PerPersonKNN(PerPersonRidge):
    """Version non linéaire des pentes personnelles : part de chaque classe parmi les k séances les plus
    semblables de la même personne (dans ce sport), en features standardisées. Même cross-fitting que PerPersonRidge."""

    def __init__(self, ks=(10, 30), cv=5, random_state=0):
        self.ks, self.cv, self.random_state = ks, cv, random_state

    def _width(self):
        return 3 * len(self.ks)

    def _fit_models(self, keys, Z, t):
        return {k: (Z[idx, 1:], t[idx]) for k, idx in pd.Series(np.arange(len(keys))).groupby(keys).indices.items()}, t.mean()

    def _score(self, keys, Z, models):
        per, prior = models
        out = np.full((len(Z), 3 * len(self.ks)), np.nan)
        for k, idx in pd.Series(np.arange(len(keys))).groupby(keys).indices.items():
            if k not in per:
                continue
            Zr, tr_ = per[k]
            d = ((Z[idx, None, 1:] - Zr[None]) ** 2).sum(-1)
            order = np.argsort(d, axis=1)
            for j, kk in enumerate(self.ks):
                nb = tr_[order[:, : min(kk, len(tr_))]]
                for c in range(3):
                    out[idx, 3 * j + c] = (nb == c).mean(1)
        return np.nan_to_num(out, nan=1 / 3)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"ppknn_k{k}_{c}" for k in self.ks for c in CLASS_ORDER])


PPKNN_FEATS = ["loc_diff_speed_kmh", "loc_diff_effort_speed", "loc_diff_max_speed_kmh", "duration_min__z_us",
               "distance_km__z_us", "altitude_gain_m__z_us", "t_rank_us"]
USE_PPKNN = False

PP_FEATS = [c for c in NUM_COLS if (c.startswith(("loc_diff_", "loc_level_")) or "__z_us" in c) and "__z_usy" not in c]
PP_FEATS += [c for c in ("hour_rel_user", "weekend", "t_rank_us", "n_points") if c in NUM_COLS]
USE_PP = USE_USER_TE and "te_us" in TE_COLS
PP_KEYS = ["te_us", "te_u"]  # pentes personnelles : par personne × sport et par personne tous sports
PP_LAMS = (3.0, 10.0, 40.0)  # rétrécissement vers le modèle global (essai pp2 : +0,003)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    transformers = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("sc", StandardScaler() if scale else "passthrough")]), NUM_COLS)]
    if CAT_COLS:
        transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                                  sparse_output=False), CAT_COLS))
    if USE_USER_TE:
        transformers.append(("user_te", TargetEncoder(target_type="multiclass", cv=5, shuffle=True,
                                                      random_state=SEED), TE_COLS))
    if USE_PP:
        for k in PP_KEYS:
            transformers.append((f"pp_{k}", PerPersonRidge(lams=PP_LAMS, random_state=SEED), [k] + PP_FEATS))
    if USE_PPKNN and USE_USER_TE:
        transformers.append(("ppknn", PerPersonKNN(random_state=SEED), ["te_us"] + [c for c in PPKNN_FEATS if c in NUM_COLS]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


MODELS = {
    "logreg": (LogisticRegression(max_iter=3000, C=0.5), True),
    "knn": (KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=N_JOBS), True),
    "rf": (RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=N_JOBS, random_state=SEED), False),
    "extratrees": (ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=N_JOBS, random_state=SEED), False),
    "hgb": (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=800, max_leaf_nodes=31, l2_regularization=1.0,
                                           early_stopping=True, n_iter_no_change=40, random_state=SEED), False),
    "hgb_mf3": (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=800, max_leaf_nodes=31, l2_regularization=1.0,
                                               max_features=0.3, early_stopping=True, n_iter_no_change=40,
                                               random_state=SEED + 7), False),
    "hgb_slow": (HistGradientBoostingClassifier(learning_rate=0.03, max_iter=2000, max_leaf_nodes=63, min_samples_leaf=40,
                                                l2_regularization=2.0, max_features=0.5, early_stopping=True,
                                                n_iter_no_change=60, random_state=SEED + 1), False),
}


def make_pipeline(name: str) -> Pipeline:
    est, scale = MODELS[name]
    return Pipeline([("prep", make_preprocessor(scale)), ("clf", clone(est))])


def macro_f1(y_true, proba, weights=None) -> float:
    p = proba * (weights if weights is not None else 1)
    return f1_score(y_true, np.array(CLASSES)[p.argmax(1)], average="macro")


def run_cv(name: str, note: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Probabilités out-of-fold sur le train, et prédictions du test par le modèle de chaque pli (pour les quotas)."""
    t0 = time.time()
    oof = np.zeros((len(X), len(CLASSES)))
    test_folds = np.zeros((N_SPLITS, len(X_test), len(CLASSES)))
    fold_scores = []
    for i, (tr, va) in enumerate(CV.split(X, y, GROUPS)):
        pipe = make_pipeline(name).fit(X.iloc[tr], y.iloc[tr])
        cols = [list(pipe.classes_).index(c) for c in CLASSES]
        oof[va] = pipe.predict_proba(X.iloc[va])[:, cols]
        test_folds[i] = pipe.predict_proba(X_test)[:, cols]
        fold_scores.append(macro_f1(y.iloc[va], oof[va]))
    score = macro_f1(y, oof)
    np.save(OOF_DIR / f"{name}.npy", oof)
    np.save(OOF_DIR / f"{name}_testfolds.npy", test_folds)
    rec = {"time": datetime.now().strftime("%H:%M"), "model": name, "features": FEATURES_VERSION,
           "cv": type(CV).__name__, "macro_f1_oof": round(score, 5), "fold_std": round(float(np.std(fold_scores)), 5),
           "seconds": round(time.time() - t0, 1), "note": note}
    log_path = OUT / "experiments.csv"
    pd.DataFrame([rec]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"{name:<11} macro-F1 OOF = {score:.4f} (± {np.std(fold_scores):.4f} entre plis, {rec['seconds']} s)")
    return oof, test_folds


# Cache par version de features : un modèle déjà évalué (par ce run ou un run parallèle) est rechargé, et
# l'ensemble considère tous les modèles en cache pour cette version. HACK_REUSE=0 force le recalcul.
OOF_DIR = OUT / "oof" / FEATURES_VERSION
OOF_DIR.mkdir(parents=True, exist_ok=True)
REUSE = os.environ.get("HACK_REUSE", "1") == "1"
_cv_results = {}
for name in MODELS:
    cached = (OOF_DIR / f"{name}.npy").exists() and (OOF_DIR / f"{name}_testfolds.npy").exists()
    if name in RUN_MODELS and not (REUSE and cached):
        _cv_results[name] = run_cv(name)
    elif cached and (REUSE or name not in RUN_MODELS):
        _cv_results[name] = (np.load(OOF_DIR / f"{name}.npy"), np.load(OOF_DIR / f"{name}_testfolds.npy"))
        print(f"{name:<11} macro-F1 OOF = {macro_f1(y, _cv_results[name][0]):.4f} (cache {FEATURES_VERSION})")
OOF = {n: r[0] for n, r in _cv_results.items()}
OOF_TEST = {n: r[1] for n, r in _cv_results.items()}

# %% [markdown]
# ## 6. Ensemble : sélection gloutonne de modèles (moyenne de probabilités) et poids de classe pour le macro-F1

# %%
def greedy_blend(oofs: dict[str, np.ndarray], max_steps: int = 10) -> list[str]:
    """Ajoute à chaque pas le modèle (avec remise) qui améliore le plus le macro-F1 OOF de la moyenne."""
    chosen: list[str] = []
    best = -1.0
    for _ in range(max_steps):
        cand = {n: macro_f1(y, np.mean([oofs[m] for m in chosen + [n]], axis=0)) for n in oofs}
        n, s = max(cand.items(), key=lambda kv: kv[1])
        if s <= best + 1e-4:
            break
        chosen.append(n)
        best = s
    return chosen


def tune_class_weights(proba: np.ndarray) -> np.ndarray:
    w = np.ones(len(CLASSES))
    for _ in range(3):
        for j in range(len(CLASSES)):
            best_v, best_s = w[j], macro_f1(y, proba, w)
            for v in np.linspace(0.7, 1.4, 29):
                w2 = w.copy()
                w2[j] = v
                s = macro_f1(y, proba, w2)
                if s > best_s + 1e-5:
                    best_v, best_s = v, s
            w[j] = best_v
    return w


BLEND = greedy_blend(OOF)
P_blend = np.mean([OOF[m] for m in BLEND], axis=0)
W = tune_class_weights(P_blend)
gain = macro_f1(y, P_blend, W) - macro_f1(y, P_blend)
if gain < 0.002:  # gain trop faible pour être distingué du bruit : on garde l'argmax simple
    W = np.ones(len(CLASSES))
print("Ensemble retenu :", BLEND, f"| macro-F1 OOF = {macro_f1(y, P_blend):.4f}",
      f"| poids de classe {dict(zip(CLASSES, W.round(2)))} (gain {gain:+.4f})")
print("\nClassement :\n", pd.Series({n: macro_f1(y, o) for n, o in OOF.items()}).sort_values(ascending=False).round(4).to_string())

# %% [markdown]
# ## 6b. Modèle final : stacking multi-versions
#
# Chaque version de features (v3 ligne de base locale, v4 effort et heure relative, v6 pentes par personne) a été
# évaluée par ce même script ; ses probabilités out-of-fold et ses prédictions test par pli sont dans
# `out/oof/<version>/` (cf. LOG.md). Les modèles se trompent différemment : une régression logistique très
# régularisée apprend, sur leurs log-probabilités OOF, à qui faire confiance. Le méta-modèle est évalué en CV
# 5 plis, puis appliqué aux prédictions test de chaque pli (même calibration que les OOF) et moyenné : pas de refit.

# %%
STACK_MODELS = [f"v3-ligne-de-base-locale:{m}" for m in ("extratrees", "hgb", "hgb_slow", "knn", "logreg", "rf")]     + [f"v4-effort-heure-relative:{m}" for m in ("extratrees", "hgb", "hgb_slow", "rf")]     + [f"v6-pentes-par-personne:{m}" for m in ("extratrees", "hgb", "logreg")]     + [f"v6-pentes-par-personne:{m}" for m in ("hgb_slow", "et_s2", "hgb_mf3", "ord_hgb")]     + [f"v7-pentes-personne-et-sport:{m}" for m in ("hgb_slow", "extratrees", "hgb_mf3", "knnp_et", "knnp_logreg", "logreg", "ord_hgb")]
# 24 modèles (stack « c4 », 17:52) : 13 de la soumission 16:08 + variantes v6/v7 (ordinal, k-NN par personne, max_features)


def load_cached(vm: str):
    v, m = vm.split(":")
    d = OUT / "oof" / v
    return np.load(d / f"{m}.npy"), np.load(d / f"{m}_testfolds.npy")


STACKED = all((OUT / "oof" / vm.split(":")[0] / f"{vm.split(':')[1]}.npy").exists() for vm in STACK_MODELS)
if STACKED:
    cached = [load_cached(vm) for vm in STACK_MODELS]
    F_oof = np.hstack([np.log(np.clip(o, 1e-6, 1)) for o, _ in cached])
    F_test_folds = [np.hstack([np.log(np.clip(t[i], 1e-6, 1)) for _, t in cached]) for i in range(cached[0][1].shape[0])]
    best = None
    for C in (0.01, 0.03, 0.1, 0.3):
        meta_oof = np.zeros((len(F_oof), len(CLASSES)))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(F_oof, y):
            lr = LogisticRegression(C=C, max_iter=2000).fit(F_oof[tr], y.values[tr])
            meta_oof[va] = lr.predict_proba(F_oof[va])[:, [list(lr.classes_).index(c) for c in CLASSES]]
        sc = macro_f1(y, meta_oof)
        print(f"stacking C = {C} : macro-F1 OOF {sc:.4f}")
        if best is None or sc > best[0]:
            best = (sc, C, meta_oof)
    lr = LogisticRegression(C=best[1], max_iter=2000).fit(F_oof, y.values)
    idx = [list(lr.classes_).index(c) for c in CLASSES]
    P_final_oof = best[2]
    P_final_test = np.mean([lr.predict_proba(Ft)[:, idx] for Ft in F_test_folds], axis=0)
    print(f"Stacking retenu : C = {best[1]}, {len(STACK_MODELS)} modèles, macro-F1 OOF {best[0]:.4f}")
else:  # cache absent (exécution à neuf) : l'ensemble glouton de la version courante tient lieu de modèle final
    print("Cache multi-versions absent : modèle final = ensemble glouton de la version courante")
    P_final_oof, P_final_test = P_blend, None

# %% [markdown]
# ## 6c. Tiers exacts par personne : affectation sous quotas
#
# Mesure : les labels sont des tiers **exacts** par personne sur tout son historique (train + test). La variance des
# comptes de classe du train vaut 1,12 x celle d'un tirage sans remise depuis des tiers exacts, contre 0,28 x un tirage
# au hasard. Donc chaque personne a, parmi ses séances inconnues, au plus ceil(N/3) - (comptes connus) séances de chaque
# classe. On affecte ses séances inconnues en maximisant la somme des log-probabilités sous ces plafonds
# (problème d'affectation, soumis à 17:01), puis mieux : équilibrage souple (17:30), qui décale le prior de chaque
# classe pour la personne jusqu'à ce que ses probabilités somment aux comptes attendus, puis argmax. Validation honnête : dans chaque pli, les inconnues d'une personne sont ses séances de
# validation ET de test, les comptes connus ceux des autres plis ; le vrai test (inconnues = test seul) est plus
# contraint, donc au moins aussi favorable. Variantes rejetées (LOG 17:05) : comptes exacts, coupe ordinale.

# %%
from scipy.optimize import linear_sum_assignment

N_PER_USER = pd.concat([train[USER], test[USER]]).value_counts() if USER else None


def assign_with_caps(logp: np.ndarray, caps: np.ndarray) -> np.ndarray:
    caps = np.maximum(caps, 0)
    if caps.sum() < len(logp):  # garde-fou d'arrondi
        caps = caps + int(np.ceil((len(logp) - caps.sum()) / len(CLASSES)))
    cols = np.repeat(np.arange(len(CLASSES)), caps)
    r, c = linear_sum_assignment(-logp[:, cols])
    out = np.empty(len(logp), int)
    out[r] = cols[c]
    return out


def expected_counts(n_unknown: int, N: int, known: np.ndarray) -> np.ndarray:
    """Comptes attendus parmi les séances inconnues : au plus proche de N/3 - connus, de somme n_unknown."""
    r = np.maximum(N / 3 - known, 0)
    r = r * n_unknown / max(r.sum(), 1e-9)
    k = np.floor(r).astype(int)
    k[np.argsort(-(r - k))[: n_unknown - k.sum()]] += 1  # comptes entiers (plus forts restes), comme sub_1731
    return k.astype(float)


# Poids de décision après équilibrage (LOG 17:51) : le macro-F1 pèse autant la classe faible ; pousser moderate de 1,2
# rapporte plus qu'il ne coûte (plateau entre 1,15 et 1,25 en validation 20 plis, sur deux graines).
DECISION_WEIGHTS = np.array([1.2 if c == "moderate" else 1.0 for c in CLASSES])


def soft_balance(P: np.ndarray, target: np.ndarray, it: int = 50) -> np.ndarray:
    """Décale le prior de chaque classe (pour une personne) jusqu'à ce que la somme de ses probabilités égale les
    comptes attendus, puis argmax : les tiers sont tenus en espérance, sans plafond dur."""
    w = np.ones(len(CLASSES))
    for _ in range(it):
        Q = P * w
        Q /= Q.sum(1, keepdims=True)
        w *= (target / np.maximum(Q.sum(0), 1e-9)) ** 0.7
    return (P * w * DECISION_WEIGHTS).argmax(1)


def quota_predict(P: np.ndarray, users: np.ndarray, known: pd.DataFrame, method: str = "soft") -> np.ndarray:
    pred, logp = P.argmax(1), np.log(np.clip(P, 1e-9, 1))
    for u, idx in pd.Series(np.arange(len(users))).groupby(users).indices.items():
        kn = known.reindex([u]).fillna(0).values[0].astype(int)
        if method == "caps":
            pred[idx] = assign_with_caps(logp[idx], int(np.ceil(N_PER_USER[u] / 3)) - kn)
        else:
            pred[idx] = soft_balance(P[idx], expected_counts(len(idx), N_PER_USER[u], kn))
    return pred


USE_QUOTA = False
if USER and P_final_test is not None:
    y_code = y.map({c: i for i, c in enumerate(CLASSES)}).values
    q_oof, q_caps, gains = np.empty(len(y), int), np.empty(len(y), int), []
    # 20 plis : 5 % du train inconnu par pli, au plus près du vrai test (avec 5 plis : 0,6285, 10 plis : 0,6298)
    for tr, va in StratifiedKFold(20, shuffle=True, random_state=11).split(X, y):
        known = pd.crosstab(train[USER].values[tr], y_code[tr]).reindex(columns=range(len(CLASSES)), fill_value=0)
        P_u, u_u = np.vstack([P_final_oof[va], P_final_test]), np.r_[train[USER].values[va], test[USER].values]
        q_oof[va] = quota_predict(P_u, u_u, known)[: len(va)]
        q_caps[va] = quota_predict(P_u, u_u, known, "caps")[: len(va)]
        gains.append(f1_score(y_code[va], q_oof[va], average="macro") - f1_score(y_code[va], P_final_oof[va].argmax(1), average="macro"))
    base, q_score = macro_f1(y, P_final_oof), f1_score(y_code, q_oof, average="macro")
    USE_QUOTA = q_score > base + 0.002  # public : 0,64276 -> 0,65118 à modèle identique (LOG 17:01)
    print(f"Plafonds durs (soumis 17:01, public 0,65118) : {f1_score(y_code, q_caps, average='macro'):.4f}")
    print(f"Équilibrage souple : macro-F1 OOF {base:.4f} -> {q_score:.4f} | plis gagnants {sum(g > 0 for g in gains)}/{len(gains)} => adopté : {USE_QUOTA}")


# %% [markdown]
# ## 7. Analyse d'erreur (matrice de confusion pour le Gate 2)

# %%
def error_analysis(proba: np.ndarray, label: str, weights=None) -> None:
    pred = np.array(CLASSES)[(proba * (weights if weights is not None else 1)).argmax(1)]
    cm = confusion_matrix(y, pred, labels=CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=axes[0], colorbar=False, values_format="d")
    ConfusionMatrixDisplay(cm / cm.sum(1, keepdims=True), display_labels=CLASSES).plot(ax=axes[1], colorbar=False, values_format=".2f")
    axes[0].set_title(f"{label} : effectifs (OOF)")
    axes[1].set_title("normalisée par vraie classe (rappel)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"confusion_{label}.png", dpi=120)
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)
    print(classification_report(y, pred, labels=CLASSES, digits=3))
    print("Matrice (lignes = vrai, colonnes = prédit, ordre", CLASSES, ") :")
    print(cm)
    off = cm.astype(float).copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    print(f"Confusion principale : vrai '{CLASSES[i]}' prédit '{CLASSES[j]}' ({int(cm[i, j])} séances, "
          f"{cm[i, j] / cm[i].sum():.0%} des '{CLASSES[i]}'). Erreurs extrêmes easy<->hard : "
          f"{int(cm[0, -1] + cm[-1, 0])}.")
    if SPORT:
        per_sport = pd.DataFrame({"sport": train[SPORT].values, "y": y, "p": pred}).groupby("sport").apply(
            lambda d: pd.Series({"n": len(d), "macro_f1": f1_score(d.y, d.p, average="macro")}), include_groups=False)
        print("\nmacro-F1 par sport :\n", per_sport.sort_values("n", ascending=False).round(3).to_string())


if USE_QUOTA:  # matrice du modèle final (stacking + quotas), celle du Gate 2
    error_analysis(np.eye(len(CLASSES))[q_oof], "final_stack_quota")
elif STACKED:
    error_analysis(P_final_oof, "final_stack")
else:
    error_analysis(P_blend, "blend", W)

# %% [markdown]
# ## 8. Entraînement final et soumission

# %%
def fit_predict_test(name: str) -> np.ndarray:
    pipe = make_pipeline(name).fit(X, y)
    cols = [list(pipe.classes_).index(c) for c in CLASSES]
    proba = pipe.predict_proba(X_test)[:, cols]
    np.save(OUT / "test_proba" / f"{name}.npy", proba)
    return proba


def write_submission(proba: np.ndarray, tag: str, weights=None) -> Path:
    labels = np.array(CLASSES)[(proba * (weights if weights is not None else 1)).argmax(1)]
    sub = pd.DataFrame({ID: test[ID].values, TARGET: labels}).set_index(ID).loc[sample[ID]].reset_index()
    assert len(sub) == len(sample) and sub[TARGET].notna().all() and set(sub[TARGET]) <= set(CLASSES)
    assert list(sub.columns) == list(sample.columns), (list(sub.columns), list(sample.columns))
    path = SUBS / f"sub_{datetime.now():%H%M}_{tag}.csv"
    sub.to_csv(path, index=False)
    print(f"Soumission écrite : {path}\nDistribution prédite : {sub[TARGET].value_counts(normalize=True).round(3).to_dict()}")
    return path


if os.environ.get("HACK_SUBMIT", "1") == "1" and STACKED:
    if USE_QUOTA:
        known = pd.crosstab(train[USER], y.map({c: i for i, c in enumerate(CLASSES)})).reindex(columns=range(len(CLASSES)), fill_value=0)
        write_submission(np.eye(len(CLASSES))[quota_predict(P_final_test, test[USER].values, known)], "final-stack-quota")
    else:
        write_submission(P_final_test, "final-stack")
elif os.environ.get("HACK_SUBMIT", "1") == "1":
    test_probas = {m: fit_predict_test(m) for m in dict.fromkeys(BLEND)}
    P_test = np.mean([test_probas[m] for m in BLEND], axis=0)
    tag = "blend-" + "+".join(dict.fromkeys(BLEND))
    if USE_QUOTA:
        n_all = pd.concat([QKEY_TR, QKEY_TE]).value_counts()
        labels = user_quota_assign(P_test, QKEY_TE.values, QKEY_TR.values, y.values, n_all)
        P_test = (labels[:, None] == np.array(CLASSES)[None, :]).astype(float)
        tag += "-quota"
    write_submission(P_test, tag, None if USE_QUOTA else W)

# %% [markdown]
# ## 9. Explication du modèle (pour la fiche papier) : importance par permutation sur un pli de validation

# %%
if os.environ.get("HACK_EXPLAIN") == "1":
    best_single = max(OOF, key=lambda n: macro_f1(y, OOF[n]))
    tr, va = next(CV.split(X, y, GROUPS))
    pipe = make_pipeline(best_single).fit(X.iloc[tr], y.iloc[tr])
    va_sample = va[: min(4000, len(va))]
    imp = permutation_importance(pipe, X.iloc[va_sample], y.iloc[va_sample], scoring="f1_macro",
                                 n_repeats=3, random_state=SEED, n_jobs=N_JOBS)
    print(f"Importance par permutation ({best_single}, baisse de macro-F1) :\n",
          pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(20).round(4).to_string())
