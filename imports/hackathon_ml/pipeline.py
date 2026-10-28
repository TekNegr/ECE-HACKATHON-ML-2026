"""Pipeline de prédiction — Hackathon ML 2026.

Utilisation :
    uv run python -m hackathon_ml.pipeline train
    uv run python -m hackathon_ml.pipeline demo
    uv run python -m hackathon_ml.pipeline submit
    uv run hackathon-submit
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
COLONNES_BRUTES = [
    "user_id",
    "sport",
    "gender",
    "year",
    "month",
    "weekday",
    "start_hour_utc",
    "duration_min",
    "distance_km",
    "speed_kmh",
    "max_speed_kmh",
    "speed_std_kmh",
    "altitude_mean_m",
    "altitude_gain_m",
    "altitude_loss_m",
    "n_points",
]
COLS_RELATIVES = [
    "duration_min",
    "distance_km",
    "speed_kmh",
    "max_speed_kmh",
    "speed_std_kmh",
    "altitude_gain_m",
    "denivele_par_km",
    "vmax_sur_vmoy",
]
GROUPES = [["user_id"], ["user_id", "sport"], ["sport"], ["user_id", "sport", "year"]]
TRAD = {"easy": "FACILE", "moderate": "MODÉRÉE", "hard": "DIFFICILE"}


def resolve_data_dir() -> Path:
    candidates = [
        Path.cwd() / "data",
        Path(__file__).resolve().parent / "data",
        ROOT_DIR / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_file(filename: str) -> Path:
    data_dir = resolve_data_dir()
    return data_dir / filename


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
            new[f"{c}_ratio_{tag}"] = full[c] / (m + 1e-6)
            new[f"{c}_z_{tag}"] = (full[c] - m) / (s + 1e-6)
            new[f"{c}_rang_{tag}"] = grp.rank(pct=True)
        new[f"nb_seances_{tag}"] = full.groupby(g)["user_id"].transform("count")
    full = pd.concat([full, pd.DataFrame(new, index=full.index)], axis=1)
    full = pd.get_dummies(full.drop(columns=["user_id"]), columns=["sport", "gender"])
    return full.replace([np.inf, -np.inf], np.nan)


def entrainer(train_csv: str | Path | None = None, test_csv: str | Path | None = None):
    train_path = Path(train_csv) if train_csv else resolve_file("train.csv")
    test_path = Path(test_csv) if test_csv else resolve_file("test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    historique = pd.concat([train[COLONNES_BRUTES], test[COLONNES_BRUTES]], ignore_index=True)
    X_all = construire_features(historique)
    X = X_all.iloc[: len(train)]
    model = HistGradientBoostingClassifier(
        random_state=0,
        learning_rate=0.05,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )
    model.fit(X, train["intensity"])
    joblib.dump({"model": model, "colonnes": list(X.columns), "historique": historique}, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")


def export_submission(output_csv: str | Path | None = None):
    train_path = resolve_file("train.csv")
    test_path = resolve_file("test.csv")
    sample_path = resolve_file("sample_submission.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_path)

    historique = pd.concat([train[COLONNES_BRUTES], test[COLONNES_BRUTES]], ignore_index=True)
    X_all = construire_features(historique)
    X = X_all.iloc[: len(train)]
    X_test = X_all.iloc[len(train) :]

    model = HistGradientBoostingClassifier(
        random_state=0,
        learning_rate=0.05,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )
    model.fit(X, train["intensity"])
    sub["intensity"] = model.predict(X_test)

    destination = Path(output_csv) if output_csv else resolve_file("submission.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(destination, index=False)
    print(f"Submission exportée vers : {destination}")


def valider(seance: dict, historique: pd.DataFrame):
    manquants = [c for c in COLONNES_BRUTES if c not in seance]
    if manquants:
        raise ValueError(f"Champs manquants : {manquants}")
    if seance["user_id"] not in set(historique["user_id"]):
        raise ValueError("Utilisateur inconnu : impossible de comparer à ses habitudes.")
    if seance["duration_min"] <= 0 or seance["distance_km"] < 0:
        raise ValueError("Durée ou distance invalide.")


def predire_seance(seance: dict) -> dict:
    art = joblib.load(MODEL_PATH)
    model, colonnes, historique = art["model"], art["colonnes"], art["historique"]

    valider(seance, historique)
    full = pd.concat([historique, pd.DataFrame([seance])[COLONNES_BRUTES]], ignore_index=True)
    X = construire_features(full).reindex(columns=colonnes).iloc[[-1]]
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    classe = classes[int(np.argmax(proba))]

    tag = "user_id_sport"
    v = X.iloc[0]
    raisons = []
    for col, nom in [("speed_kmh", "vitesse"), ("duration_min", "durée"), ("distance_km", "distance"), ("altitude_gain_m", "dénivelé")]:
        r = v[f"{col}_ratio_{tag}"]
        if pd.notna(r) and abs(r - 1) >= 0.15:
            sens = "au-dessus" if r > 1 else "en dessous"
            raisons.append(f"{nom} {abs(r - 1) * 100:.0f} % {sens} de sa moyenne en {seance['sport']}")
    message = (f"Cette séance de {seance['sport']} ({seance['duration_min']:.0f} min, "
               f"{seance['distance_km']:.1f} km) est probablement {TRAD[classe]} pour cette personne "
               f"(confiance {proba.max() * 100:.0f} %).")
    message += (" Raisons : " + " ; ".join(raisons) + ".") if raisons else " Elle ressemble à ses séances habituelles."
    return {
        "classe": classe,
        "probabilites": {TRAD[c]: round(float(p), 3) for c, p in zip(classes, proba)},
        "message": message,
    }


def demo_main() -> None:
    exemple = {
        "user_id": 1031,
        "sport": "running",
        "gender": "male",
        "year": 2015,
        "month": 10,
        "weekday": 6,
        "start_hour_utc": 18,
        "duration_min": 55.0,
        "distance_km": 12.0,
        "speed_kmh": 13.1,
        "max_speed_kmh": 15.5,
        "speed_std_kmh": 1.4,
        "altitude_mean_m": 150.0,
        "altitude_gain_m": 120.0,
        "altitude_loss_m": 118.0,
        "n_points": 500,
    }
    res = predire_seance(exemple)
    print(res["message"])
    print(res["probabilites"])


def submit_main() -> None:
    export_submission()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackathon pipeline commands")
    parser.add_argument("command", nargs="?", choices=["train", "demo", "submit"], default="demo")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    if args.command == "train":
        entrainer()
    elif args.command == "submit":
        export_submission(args.output)
    else:
        demo_main()


if __name__ == "__main__":
    main()
