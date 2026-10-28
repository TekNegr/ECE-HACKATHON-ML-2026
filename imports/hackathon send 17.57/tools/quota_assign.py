"""Affectation sous contrainte de tiers par personne.

Mesure (LOG 17:05) : les labels sont des tiers EXACTS par personne sur tout son historique (train + test) :
la variance des comptes de classe du train vaut 1,12 x celle d'un tirage sans remise depuis des tiers exacts,
contre 0,28 x un tirage au hasard. Donc pour chaque personne, le nombre de séances de chaque classe encore
inconnues est borné : au plus ceil(N/3) - (comptes connus).
On affecte les séances inconnues d'une personne aux classes en maximisant la somme des log-probabilités
sous ces capacités (problème d'affectation, linear_sum_assignment).

Validation honnête : pour le pli k, les inconnues sont les séances de validation ET celles du test de la
personne ; les comptes connus sont ceux des autres plis. On évalue sur la validation seulement.
Le vrai test (inconnues = test seul) est plus contraint, donc au moins aussi favorable.

Usage : python tools/quota_assign.py <tag du blend>   (lit out/blend_<tag>.npz, écrit submissions/sub_HHMM_<tag>-quota.csv)
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
tag = sys.argv[1]
B = np.load(ROOT / "out" / f"blend_{tag}.npz", allow_pickle=True)
CLASSES = list(B["classes"])
P_oof, P_te = np.clip(B["oof"], 1e-9, None), np.clip(B["test"], 1e-9, None)
P_oof, P_te = P_oof / P_oof.sum(1, keepdims=True), P_te / P_te.sum(1, keepdims=True)
train, test = pd.read_csv(ROOT / "data" / "train.csv"), pd.read_csv(ROOT / "data" / "test.csv")
y = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).values
N_user = pd.concat([train.user_id, test.user_id]).value_counts()


def assign(logp, caps):
    """logp (n, 3), caps (3,) entiers : affectation max somme logp avec au plus caps[k] lignes en classe k."""
    caps = np.maximum(caps, 0)
    if caps.sum() < len(logp):  # garde-fou (arrondis) : on relâche uniformément
        caps = caps + int(np.ceil((len(logp) - caps.sum()) / 3))
    cols = np.repeat(np.arange(3), caps)
    r, c = linear_sum_assignment(-logp[:, cols])
    out = np.empty(len(logp), int)
    out[r] = cols[c]
    return out


def exact_counts(n_unknown, N, known):
    """Comptes entiers par classe pour les inconnues : au plus proche de N/3 - connus, de somme n_unknown."""
    r = np.maximum(N / 3 - known, 0)
    r = r * n_unknown / max(r.sum(), 1e-9)
    k = np.floor(r).astype(int)
    for j in np.argsort(-(r - k))[: n_unknown - k.sum()]:
        k[j] += 1
    return k


def ordinal_cut(P, counts):
    """Trie les séances par intensité attendue P(moderate) + 2 P(hard) et coupe aux comptes voulus."""
    order = np.argsort(P[:, 1] + 2 * P[:, 2], kind="stable")
    out = np.empty(len(P), int)
    out[order] = np.repeat(np.arange(3), counts)
    return out


def balance(P, target, it=50):
    """Équilibrage souple (LOG 17:28) : décale le prior de chaque classe pour la personne jusqu'à ce que la somme de
    ses probabilités sur ses séances inconnues égale les comptes attendus, puis argmax (pas de plafond dur)."""
    w = np.ones(3)
    for _ in range(it):
        Q = P * w
        Q /= Q.sum(1, keepdims=True)
        w *= (target / np.maximum(Q.sum(0), 1e-9)) ** 0.7
    cw = np.ones(3)
    cw[CLASSES.index("moderate")] = float(os.environ.get("QMOD", 1))  # poids global de décision sur moderate
    cw[CLASSES.index("hard")] = float(os.environ.get("QHARD", 1))
    return (P * w * cw).argmax(1)


METHOD = sys.argv[2] if len(sys.argv) > 2 else "caps"


def quota_predict(P_unknown, users_unknown, known_counts):
    """known_counts : DataFrame index user, colonnes 0..2."""
    pred = P_unknown.argmax(1).copy()
    logp = np.log(P_unknown)
    for u, idx in pd.Series(np.arange(len(users_unknown))).groupby(users_unknown).indices.items():
        kn = known_counts.reindex([u]).fillna(0).values[0].astype(int)
        if METHOD == "caps":
            pred[idx] = assign(logp[idx], int(np.ceil(N_user[u] / 3)) - kn)
        elif METHOD == "exact":
            pred[idx] = assign(logp[idx], exact_counts(len(idx), N_user[u], kn))
        elif METHOD == "bal":
            pred[idx] = balance(P_unknown[idx] ** float(os.environ.get("QT", 1)), exact_counts(len(idx), N_user[u], kn).astype(float))
        else:
            pred[idx] = ordinal_cut(P_unknown[idx], exact_counts(len(idx), N_user[u], kn))
    return pred


# Validation honnête (les plis ici sont indépendants de ceux du modèle : on ne réapprend rien)
base, quota = [], []
pred_q = np.empty(len(y), int)
for tr, va in StratifiedKFold(int(os.environ.get("QFOLDS", 5)), shuffle=True, random_state=int(os.environ.get("QSEED", 11))).split(train, y):
    known = pd.crosstab(train.user_id.values[tr], y[tr]).reindex(columns=range(3), fill_value=0)
    Pu = np.vstack([P_oof[va], P_te])
    uu = np.r_[train.user_id.values[va], test.user_id.values]
    pq = quota_predict(Pu, uu, known)[: len(va)]
    pred_q[va] = pq
    base.append(f1_score(y[va], P_oof[va].argmax(1), average="macro"))
    quota.append(f1_score(y[va], pq, average="macro"))
print(f"argmax : {np.mean(base):.4f} | affectation sous quotas : {np.mean(quota):.4f} (par pli : {np.round(quota, 4).tolist()})")
print(f"macro-F1 OOF global avec quotas : {f1_score(y, pred_q, average='macro'):.4f}")

known = pd.crosstab(train.user_id, y).reindex(columns=range(3), fill_value=0)
pred_te = quota_predict(P_te, test.user_id.values, known)
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": np.array(CLASSES)[pred_te]})
sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
assert list(sub.columns) == list(sample.columns) and (sub.workout_id.values == sample.workout_id.values).all()
# contrôle : avec ces prédictions, chaque personne a bien des tiers (écart max au tiers exact)
tot = pd.crosstab(pd.concat([train.user_id, test.user_id]).values, np.r_[y, pred_te])
print("écart max par personne au tiers (train + test prédit) :", float((tot.sub(tot.sum(1) / 3, axis=0)).abs().max().max()))
out = ROOT / "submissions" / f"sub_{datetime.now():%H%M}_{tag}-quota-{METHOD}.csv"
sub.to_csv(out, index=False)
print("écrit :", out.name, "| changement vs argmax :", round(float((pred_te != P_te.argmax(1)).mean()), 3))
