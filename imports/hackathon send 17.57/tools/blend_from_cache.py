"""Soumission depuis le cache OOF : moyenne des prédictions test des modèles de chaque pli (bagging 5 plis).

Usage : python tools/blend_from_cache.py <version_features> <modele>[,<modele>...] [--weights easy,moderate,hard]
Un modèle répété compte plusieurs fois (même pondération que la sélection gloutonne).
--greedy : sélection gloutonne avec remise sur tous les modèles en cache (models = "all"), puis réglage des poids de classe.
Les versions peuvent se mélanger : "v3-...:hgb,v4-...:extratrees".
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["easy", "moderate", "hard"]
ap = argparse.ArgumentParser()
ap.add_argument("version"); ap.add_argument("models"); ap.add_argument("--weights", default="1,1,1"); ap.add_argument("--tag", default=""); ap.add_argument("--greedy", action="store_true")
ap.add_argument("--stack", action="store_true", help="méta-modèle : régression logistique sur log-probas OOF (évaluée en CV 5 plis)")
a = ap.parse_args()
train = pd.read_csv(ROOT / "data" / "train.csv"); test = pd.read_csv(ROOT / "data" / "test.csv")
y = train["intensity"].values


def path(m):  # "version:modele" ou "modele" (version par défaut)
    v, _, n = m.rpartition(":")
    return ROOT / "out" / "oof" / (v or a.version), n


FOLDS = {}


def load(m):
    d, n = path(m)
    tf = np.load(d / f"{n}_testfolds.npy")
    FOLDS[m] = tf
    return np.load(d / f"{n}.npy"), tf.mean(0)


def f1(p):
    return f1_score(y, np.array(CLASSES)[p.argmax(1)], average="macro")


if a.models == "all":
    names = []
    for v in a.version.split("+"):
        names += [f"{v}:{f.stem}" for f in sorted((ROOT / "out" / "oof" / v).glob("*.npy")) if not f.stem.endswith("_testfolds")]
else:
    names = a.models.split(",")
cache = {m: load(m) for m in set(names)}
for m in sorted(cache):
    print(f"  {m:45s} {f1(cache[m][0]):.4f}")
if a.stack:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    ms = sorted(cache)
    F = np.hstack([np.log(np.clip(cache[m][0], 1e-6, 1)) for m in ms])
    n_folds = FOLDS[ms[0]].shape[0]
    Ft_folds = [np.hstack([np.log(np.clip(FOLDS[m][i], 1e-6, 1)) for m in ms]) for i in range(n_folds)]
    best = None
    for C in (0.01, 0.03, 0.1, 0.3):
        meta_oof = np.zeros((len(F), 3))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(F, y):
            lr = LogisticRegression(C=C, max_iter=2000).fit(F[tr], y[tr])
            meta_oof[va] = lr.predict_proba(F[va])[:, [list(lr.classes_).index(c) for c in CLASSES]]
        sc = f1(meta_oof)
        print(f"  stacking C={C}: OOF {sc:.4f}")
        if best is None or sc > best[0]:
            best = (sc, C, meta_oof)
    lr = LogisticRegression(C=best[1], max_iter=2000).fit(F, y)
    # méta-modèle appliqué à chaque pli (même calibration que les OOF), puis moyenne
    idx = [list(lr.classes_).index(c) for c in CLASSES]
    pt_meta = np.mean([lr.predict_proba(Ft)[:, idx] for Ft in Ft_folds], axis=0)
    cache = {"stack": (best[2], pt_meta)}
    models, w = ["stack"], np.ones(3)
    print("stacking retenu : C =", best[1], "| OOF", round(best[0], 4), "| modèles :", ms)
elif a.greedy:
    models, best = [], -1
    for _ in range(12):
        cand = {m: f1(np.mean([cache[x][0] for x in models + [m]], axis=0)) for m in cache}
        m, sc = max(cand.items(), key=lambda kv: kv[1])
        if sc <= best + 1e-4:
            break
        models.append(m); best = sc
    base = np.mean([cache[x][0] for x in models], axis=0)
    w, bw = np.ones(3), f1(base)
    for _ in range(2):
        for k in range(3):
            for v in np.arange(0.8, 1.21, 0.01):
                t = w.copy(); t[k] = v
                if f1(base * t) > bw + 1e-6:
                    bw, w = f1(base * t), t
    if bw - best < 0.002:
        w = np.ones(3)
    print("sélection :", models, "| OOF", round(best, 4), "| poids", w.round(2).tolist(), "->", round(f1(base * w), 4))
else:
    models = names
    w = np.array([float(x) for x in a.weights.split(",")])
oof = np.mean([cache[m][0] for m in models], axis=0) * w
pt = np.mean([cache[m][1] for m in models], axis=0) * w
np.savez(ROOT / "out" / f"blend_{a.tag or 'blend'}.npz", oof=oof, test=pt, classes=np.array(CLASSES))
pred_oof = np.array(CLASSES)[oof.argmax(1)]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, pred_oof, labels=CLASSES)
print(classification_report(y, pred_oof, labels=CLASSES, digits=3))
print("confusion (lignes = vrai, colonnes = prédit, ordre easy/moderate/hard) :")
print(cm)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, M, title, fmt in ((axes[0], cm, "Effectifs", "d"), (axes[1], cm / cm.sum(1, keepdims=True), "Normalisée par ligne (rappel)", ".2f")):
    ax.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("prédit"); ax.set_ylabel("vrai"); ax.set_title(title)
fig.suptitle(f"Matrice de confusion OOF ({a.tag or a.version}) - macro-F1 {f1_score(y, pred_oof, average='macro'):.3f}")
fig.tight_layout()
fig.savefig(ROOT / "out" / "figures" / f"confusion_{a.tag or 'blend'}.png", dpi=120)
print("macro-F1 OOF :", round(f1_score(train["intensity"], np.array(CLASSES)[oof.argmax(1)], average="macro"), 4))
pred = np.array(CLASSES)[pt.argmax(1)]
sub = pd.DataFrame({"workout_id": test["workout_id"], "intensity": pred})
sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample) and (sub.workout_id.values == sample.workout_id.values).all()
out = ROOT / "submissions" / f"sub_{datetime.now():%H%M}_{a.tag or a.version.split('-')[0]}-{'+'.join(sorted(set(x.rpartition(':')[2] for x in models)))}.csv"
sub.to_csv(out, index=False)
print("écrit :", out.name, "| distribution :", sub.intensity.value_counts(normalize=True).round(3).to_dict())
