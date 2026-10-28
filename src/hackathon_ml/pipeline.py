"""Pipeline de prédiction — Hackathon ML 2026.

Séance brute  ->  1. validation  ->  2. normalisation (par rapport aux habitudes de l'utilisateur)
              ->  3. modèle HistGradientBoosting  ->  4. réponse compréhensible

Utilisation :
    python pipeline.py train                      # entraîne et sauvegarde model.joblib
    python pipeline.py demo                       # prédit une séance d'exemple
    from pipeline import predire_seance           # dans un autre script / notebook
"""
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

MODEL_PATH = "model.joblib"
COLONNES_BRUTES = ["user_id", "sport", "gender", "year", "month", "weekday", "start_hour_utc",
                   "duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh",
                   "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points"]
COLS_RELATIVES = ["duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh",
                  "altitude_gain_m", "denivele_par_km", "vmax_sur_vmoy"]
GROUPES = [["user_id"], ["user_id", "sport"], ["sport"], ["user_id", "sport", "year"]]
TRAD = {"easy": "FACILE", "moderate": "MODÉRÉE", "hard": "DIFFICILE"}


# ---------- 2. Normalisation / feature engineering ----------
def construire_features(full: pd.DataFrame) -> pd.DataFrame:
    """Transforme les séances brutes en features normalisées par rapport à chaque utilisateur."""
    full = full.copy()
    full["denivele_par_km"] = full["altitude_gain_m"] / (full["distance_km"] + 1e-3)
    full["vmax_sur_vmoy"] = full["max_speed_kmh"] / (full["speed_kmh"] + 1e-3)
    new = {}
    for g in GROUPES:
        tag = "_".join(g)
        for c in COLS_RELATIVES:
            grp = full.groupby(g)[c]
            m, s = grp.transform("mean"), grp.transform("std")
            new[f"{c}_ratio_{tag}"] = full[c] / (m + 1e-6)       # x fois sa moyenne
            new[f"{c}_z_{tag}"] = (full[c] - m) / (s + 1e-6)     # z-score : écart normalisé
            new[f"{c}_rang_{tag}"] = grp.rank(pct=True)           # rang parmi ses séances (0-1)
        new[f"nb_seances_{tag}"] = full.groupby(g)["user_id"].transform("count")
    full = pd.concat([full, pd.DataFrame(new, index=full.index)], axis=1)
    full = pd.get_dummies(full.drop(columns=["user_id"]), columns=["sport", "gender"])
    return full.replace([np.inf, -np.inf], np.nan)


# ---------- Entraînement ----------
def entrainer(train_csv="train.csv", test_csv="test.csv"):
    train, test = pd.read_csv(train_csv), pd.read_csv(test_csv)
    historique = pd.concat([train[COLONNES_BRUTES], test[COLONNES_BRUTES]], ignore_index=True)
    X_all = construire_features(historique)
    X = X_all.iloc[:len(train)]
    model = HistGradientBoostingClassifier(random_state=0, learning_rate=0.05, max_iter=600,
                                           early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=30)
    model.fit(X, train["intensity"])
    joblib.dump({"model": model, "colonnes": list(X.columns), "historique": historique}, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


# ---------- 1. Validation de l'entrée ----------
def valider(seance: dict, historique: pd.DataFrame):
    manquants = [c for c in COLONNES_BRUTES if c not in seance]
    if manquants:
        raise ValueError(f"Champs manquants : {manquants}")
    if seance["user_id"] not in set(historique["user_id"]):
        raise ValueError("Utilisateur inconnu : impossible de comparer à ses habitudes.")
    if seance["duration_min"] <= 0 or seance["distance_km"] < 0:
        raise ValueError("Durée ou distance invalide.")


# ---------- Pipeline complet : 1 -> 2 -> 3 -> 4 ----------
def predire_seance(seance: dict) -> dict:
    art = joblib.load(MODEL_PATH)
    model, colonnes, historique = art["model"], art["colonnes"], art["historique"]

    valider(seance, historique)                                            # 1
    full = pd.concat([historique, pd.DataFrame([seance])[COLONNES_BRUTES]], ignore_index=True)
    X = construire_features(full).reindex(columns=colonnes).iloc[[-1]]     # 2
    proba = model.predict_proba(X)[0]                                      # 3
    classes = list(model.classes_)
    classe = classes[int(np.argmax(proba))]

    # 4. Réponse compréhensible
    tag = "user_id_sport"
    v = X.iloc[0]
    raisons = []
    for col, nom in [("speed_kmh", "vitesse"), ("duration_min", "durée"),
                     ("distance_km", "distance"), ("altitude_gain_m", "dénivelé")]:
        r = v[f"{col}_ratio_{tag}"]
        if pd.notna(r) and abs(r - 1) >= 0.15:
            sens = "au-dessus" if r > 1 else "en dessous"
            raisons.append(f"{nom} {abs(r - 1) * 100:.0f} % {sens} de sa moyenne en {seance['sport']}")
    message = (f"Cette séance de {seance['sport']} ({seance['duration_min']:.0f} min, "
               f"{seance['distance_km']:.1f} km) est probablement {TRAD[classe]} pour cette personne "
               f"(confiance {proba.max() * 100:.0f} %).")
    message += (" Raisons : " + " ; ".join(raisons) + ".") if raisons else \
        " Elle ressemble à ses séances habituelles."
    return {"classe": classe,
            "probabilites": {TRAD[c]: round(float(p), 3) for c, p in zip(classes, proba)},
            "message": message}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "train":
        entrainer()
    else:
        exemple = {"user_id": 1031, "sport": "running", "gender": "male", "year": 2015, "month": 10,
                   "weekday": 6, "start_hour_utc": 18, "duration_min": 55.0, "distance_km": 12.0,
                   "speed_kmh": 13.1, "max_speed_kmh": 15.5, "speed_std_kmh": 1.4,
                   "altitude_mean_m": 150.0, "altitude_gain_m": 120.0, "altitude_loss_m": 118.0,
                   "n_points": 500}
        res = predire_seance(exemple)
        print(res["message"])
        print(res["probabilites"])
