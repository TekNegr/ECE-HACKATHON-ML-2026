"""Modèle par paires : « laquelle de ces deux séances de la même personne était la plus dure ? ».

Motivation (LOG 17:00) : les labels sont des tiers du rang de la séance dans tout l'historique de la personne.
Un classifieur apprend sur des paires de séances d'une même personne de classes différentes (différences de features
+ niveau de la séance), cible = la première est plus dure. Pour une séance inconnue, on calcule sa probabilité moyenne
d'être plus dure que les séances CONNUES de la personne classées easy, moderate et hard : trois scores qui entrent dans
le stacking comme un modèle de plus (sortie out/oof/v8-paires/pw.npy et pw_testfolds.npy).
Anti-fuite : 5 plis ; pour le pli k, modèle appris sur les paires des autres plis, comparaisons aux séances des autres plis.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "oof" / "v8-paires"
OUT.mkdir(parents=True, exist_ok=True)
train, test = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
CLASSES = ["easy", "moderate", "hard"]
y = train.intensity.map({c: i for i, c in enumerate(CLASSES)}).values
N_PAIRS = int(os.environ.get("PW_PAIRS", 300000))


def feats(df):
    f = pd.DataFrame(index=df.index)
    for c in ("duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh", "altitude_mean_m",
              "altitude_gain_m", "altitude_loss_m", "n_points", "start_hour_utc", "month", "weekday"):
        v = df[c].astype(float)
        f[c] = v.where(v > -9000)
    f["t"] = df.year + (df.month - 1) / 12
    f["effort_speed"] = (f.distance_km + f.altitude_gain_m.fillna(0) / 100) / (f.duration_min / 60).clip(lower=1e-3)
    f["sport"] = df.sport.astype("category").cat.codes
    return f


ALL = pd.concat([train, test], ignore_index=True)
FA = feats(ALL).values.astype(float)
USERS = ALL.user_id.values
SPORT = FA[:, -1]
NTR = len(train)


def pair_X(i, j):
    """Paires (i, j) : niveau de i, différences i - j, même sport."""
    return np.c_[FA[i, :-1], FA[i, :-1] - FA[j, :-1], SPORT[i], SPORT[j], SPORT[i] == SPORT[j]]


def sample_pairs(rows, rng):
    by_user = pd.Series(rows).groupby(USERS[rows]).apply(np.array)
    I, J = [], []
    per_user = max(N_PAIRS // len(by_user), 10)
    for idx in by_user:
        if len(idx) < 3:
            continue
        a, b = rng.choice(idx, per_user), rng.choice(idx, per_user)
        keep = y[a] != y[b]
        I.append(a[keep]); J.append(b[keep])
    I, J = np.concatenate(I), np.concatenate(J)
    return I, J, (y[I] > y[J]).astype(int)


def score(model, unknown, known):
    """Pour chaque séance inconnue : P(plus dure) moyenne contre les séances connues de sa personne, par classe connue."""
    out = np.full((len(unknown), 3), 0.5)
    ku = pd.Series(known).groupby(USERS[known]).apply(np.array)
    I, J, owner = [], [], []
    for pos, i in enumerate(unknown):
        k = ku.get(USERS[i])
        if k is None:
            continue
        I.append(np.full(len(k), i)); J.append(k); owner.append(np.full(len(k), pos))
    I, J, owner = np.concatenate(I), np.concatenate(J), np.concatenate(owner)
    p = model.predict_proba(pair_X(I, J))[:, 1]
    df = pd.DataFrame({"o": owner, "c": y[J], "p": p}).groupby(["o", "c"]).p.mean().unstack()
    for c in range(3):
        if c in df:
            out[df.index.values, c] = df[c].fillna(0.5).values
    return out


rng = np.random.default_rng(0)
oof = np.zeros((NTR, 3))
test_folds = []
test_rows = np.arange(NTR, len(ALL))
t0 = time.time()
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train, y)):
    I, J, t = sample_pairs(tr, rng)
    m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=400, max_leaf_nodes=63, early_stopping=True,
                                       n_iter_no_change=20, random_state=k).fit(pair_X(I, J), t)
    oof[va] = score(m, va, tr)
    test_folds.append(score(m, test_rows, tr))
    print(f"pli {k} : {len(t)} paires, {time.time() - t0:.0f} s", flush=True)

np.save(OUT / "pw_raw.npy", oof)
# Pour le stacking : les trois scores sont mis sous forme de « probabilités » positives normalisées (le méta-modèle
# prend leur log ; la normalisation garde l'information relative, l'échelle absolue est apprise par le méta-modèle).
def as_proba(s):
    s = np.clip(s, 1e-3, 1 - 1e-3)
    z = np.c_[1 - s[:, 0], s[:, 0] * (1 - s[:, 2]), s[:, 2]]  # plus dure que les easy -> pas easy ; plus dure que les hard -> hard
    return z / z.sum(1, keepdims=True)
np.save(OUT / "pw.npy", as_proba(oof))
np.save(OUT / "pw_testfolds.npy", np.stack([as_proba(t) for t in test_folds]))
from sklearn.metrics import f1_score
print("macro-F1 de la règle seule (argmax) :", round(f1_score(y, as_proba(oof).argmax(1), average="macro"), 4))
