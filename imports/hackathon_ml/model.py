"""Hackathon ML 2026 - validation and Kaggle export CLI.

Usage:
    uv run python -m hackathon_ml.model validate
    uv run python -m hackathon_ml.model submit --output submission.csv
    uv run hackathon-test
    uv run hackathon-submit
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def resolve_data_dir() -> Path:
    candidates = [
        Path.cwd() / "data",
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parents[2] / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_data_files() -> tuple[Path, Path, Path]:
    data_dir = resolve_data_dir()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    for required in (train_path, test_path, sample_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")
    return train_path, test_path, sample_path


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.concat([train.drop(columns=["intensity"]), test], ignore_index=True)
    full["denivele_par_km"] = full["altitude_gain_m"] / (full["distance_km"] + 1e-3)
    full["vmax_sur_vmoy"] = full["max_speed_kmh"] / (full["speed_kmh"] + 1e-3)

    cols = [
        "duration_min",
        "distance_km",
        "speed_kmh",
        "max_speed_kmh",
        "speed_std_kmh",
        "altitude_gain_m",
        "denivele_par_km",
        "vmax_sur_vmoy",
    ]
    groups = [["user_id"], ["user_id", "sport"], ["sport"], ["user_id", "sport", "year"]]
    new = {}
    for g in groups:
        tag = "_".join(g)
        for c in cols:
            grp = full.groupby(g)[c]
            m, s = grp.transform("mean"), grp.transform("std")
            new[f"{c}_ratio_{tag}"] = full[c] / (m + 1e-6)
            new[f"{c}_z_{tag}"] = (full[c] - m) / (s + 1e-6)
            new[f"{c}_rang_{tag}"] = grp.rank(pct=True)
        new[f"nb_seances_{tag}"] = full.groupby(g)["workout_id"].transform("count")
    full = pd.concat([full, pd.DataFrame(new)], axis=1)

    full = full.drop(columns=["user_id"])
    full = pd.get_dummies(full, columns=["sport", "gender"])
    full = full.replace([np.inf, -np.inf], np.nan)
    X = full.iloc[: len(train)].drop(columns=["workout_id"])
    X_test = full.iloc[len(train) :].drop(columns=["workout_id"])
    return X, X_test


def build_models() -> dict[str, object]:
    return {
        "LogReg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
        ),
        "HistGB": HistGradientBoostingClassifier(
            random_state=0,
            learning_rate=0.05,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
        ),
    }


def validate_model() -> dict[str, float]:
    train_path, test_path, _ = load_data_files()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = train["intensity"]
    X, _ = build_features(train, test)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores: dict[str, float] = {}
    for name, model in build_models().items():
        scores[name] = float(cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1).mean())
        print(f"{name}: macro-F1 CV = {scores[name]:.4f}")

    best_model_name = max(scores, key=scores.get)
    best_model = build_models()[best_model_name]
    preds = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)
    labels = ["easy", "moderate", "hard"]
    print(pd.DataFrame(confusion_matrix(y, preds, labels=labels), index=labels, columns=labels))
    print(classification_report(y, preds))
    return scores


def submit_model(output_path: str | Path = "submission.csv") -> Path:
    train_path, test_path, sample_path = load_data_files()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sample_path)

    X, X_test = build_features(train, test)
    y = train["intensity"]
    model = build_models()["HistGB"]
    model.fit(X, y)
    submission["intensity"] = model.predict(X_test)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"submission exportée : {output}")
    return output


def validate_main() -> None:
    validate_model()


def submit_main() -> None:
    submit_model()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackathon model validation and Kaggle export")
    parser.add_argument("command", nargs="?", choices=["validate", "submit"], default="validate")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    if args.command == "validate":
        validate_main()
    else:
        submit_main()
        # compatibility with custom output path if provided
        if args.output != "submission.csv":
            submit_model(args.output)


if __name__ == "__main__":
    main()
