from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from hackathon_ml.config import ProjectConfig
from hackathon_ml.logging_utils import configure_logging


RAW_COLUMNS = [
    "user_id", "sport", "gender", "year", "month", "weekday", "start_hour_utc",
    "duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh",
    "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points",
]
RELATIVE_COLUMNS = [
    "duration_min", "distance_km", "speed_kmh", "max_speed_kmh",
    "speed_std_kmh", "altitude_gain_m", "denivele_par_km", "vmax_sur_vmoy",
]
GROUPS = [["user_id"], ["user_id", "sport"], ["sport"], ["user_id", "sport", "year"]]
CLASS_ORDER = np.array(["easy", "moderate", "hard"])


def build_features(full: pd.DataFrame) -> pd.DataFrame:
    data = full[RAW_COLUMNS].copy()
    data["denivele_par_km"] = data["altitude_gain_m"] / (data["distance_km"] + 1e-3)
    data["vmax_sur_vmoy"] = data["max_speed_kmh"] / (data["speed_kmh"] + 1e-3)
    data["speed_calc_kmh"] = data["distance_km"] / (data["duration_min"] / 60 + 1e-6)
    data["denivele_par_min"] = data["altitude_gain_m"] / (data["duration_min"] + 1e-3)
    data["vmax_sur_speed"] = data["max_speed_kmh"] / (data["speed_kmh"] + 1e-3)
    data["effort_speed"] = (
        data["distance_km"] + data["altitude_gain_m"].fillna(0) / 100
    ) / (data["duration_min"] / 60 + 1e-6)
    data["month_index"] = data["year"] * 12 + data["month"]
    data["is_weekend"] = (data["weekday"] >= 5).astype(int)
    hour_angle = 2 * np.pi * data["start_hour_utc"] / 24
    data["hour_sin"] = np.sin(hour_angle)
    data["hour_cos"] = np.cos(hour_angle)

    for group, tag in (
        (["user_id", "month_index"], "user_month"),
        (["user_id", "sport", "month_index"], "user_sport_month"),
    ):
        grouped = data.groupby(group, observed=True)
        data[f"sessions_{tag}"] = grouped["user_id"].transform("size")
        data[f"duration_sum_{tag}"] = grouped["duration_min"].transform("sum")
        data[f"distance_sum_{tag}"] = grouped["distance_km"].transform("sum")

    generated: dict[str, pd.Series] = {}
    for group in GROUPS:
        tag = "_".join(group)
        for column in RELATIVE_COLUMNS:
            grouped = data.groupby(group, observed=True)[column]
            mean = grouped.transform("mean")
            std = grouped.transform("std")
            generated[f"{column}_ratio_{tag}"] = data[column] / (mean + 1e-6)
            generated[f"{column}_z_{tag}"] = (data[column] - mean) / (std + 1e-6)
            generated[f"{column}_rank_{tag}"] = grouped.rank(pct=True)
        generated[f"nb_seances_{tag}"] = data.groupby(group, observed=True)["user_id"].transform("size")
    data = pd.concat([data, pd.DataFrame(generated, index=data.index)], axis=1)
    data = pd.get_dummies(data.drop(columns=["user_id"]), columns=["sport", "gender"])
    return data.replace([np.inf, -np.inf], np.nan)


def make_model() -> VotingClassifier:
    """Build the three-model soft-voting ensemble used by every command."""
    random_forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    hist_gradient_boosting = HistGradientBoostingClassifier(
        random_state=42,
        learning_rate=0.05,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    )
    xgboost = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[
            ("random_forest", random_forest),
            ("hist_gradient_boosting", hist_gradient_boosting),
            ("xgboost", xgboost),
        ],
        voting="soft",
        n_jobs=-1,
    )


def load_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (config.train_path, config.test_path, config.sample_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")
    return pd.read_csv(config.train_path), pd.read_csv(config.test_path), pd.read_csv(config.sample_path)


def probabilities_in_class_order(model: VotingClassifier, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    columns = [list(model.classes_).index(label) for label in CLASS_ORDER]
    return probabilities[:, columns]


def expected_user_counts(unknown_count: int, total_count: int, known_counts: np.ndarray) -> np.ndarray:
    """Estimate the remaining per-user class counts under the measured 1/3-label rule."""
    expected = np.maximum(total_count / 3 - known_counts, 0)
    expected *= unknown_count / max(expected.sum(), 1e-9)
    counts = np.floor(expected).astype(int)
    remainder = unknown_count - counts.sum()
    if remainder:
        counts[np.argsort(-(expected - counts))[:remainder]] += 1
    return counts.astype(float)


def soft_balance(probabilities: np.ndarray, target_counts: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Adjust class priors softly, preserving confidence better than hard quotas."""
    weights = np.ones(len(CLASS_ORDER))
    for _ in range(iterations):
        adjusted = probabilities * weights
        adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
        weights *= (target_counts / np.maximum(adjusted.sum(axis=0), 1e-12)) ** 0.7
    decision_weights = np.array([1.0, 1.2, 1.0])
    return (probabilities * weights * decision_weights).argmax(axis=1)


def balance_user_predictions(
    probabilities: np.ndarray,
    users: pd.Series | np.ndarray,
    known_users: pd.Series | np.ndarray,
    known_labels: pd.Series | np.ndarray,
    total_user_counts: pd.Series,
) -> np.ndarray:
    """Apply the soft per-user balancing used by the strongest imported experiment."""
    users = np.asarray(users)
    known_users = np.asarray(known_users)
    known_labels = np.asarray(known_labels)
    label_codes = {label: index for index, label in enumerate(CLASS_ORDER)}
    known_codes = np.array([label_codes[label] for label in known_labels])
    known_table = pd.crosstab(pd.Series(known_users), pd.Series(known_codes)).reindex(
        columns=range(len(CLASS_ORDER)), fill_value=0
    )
    predictions = probabilities.argmax(axis=1)
    for user, indices in pd.Series(np.arange(len(users))).groupby(users).indices.items():
        known_counts = known_table.reindex([user], fill_value=0).iloc[0].to_numpy(dtype=float)
        target = expected_user_counts(
            len(indices),
            int(total_user_counts.get(user, len(indices))),
            known_counts,
        )
        predictions[indices] = soft_balance(probabilities[indices], target)
    return predictions


def train(config: ProjectConfig | None = None) -> Path:
    config = config or ProjectConfig()
    logger = configure_logging(config.log_path)
    logger.info("TRAIN: début de l'entraînement")
    train_df, test_df, _ = load_data(config)
    logger.info("TRAIN: données chargées (%d lignes train, %d lignes test)", len(train_df), len(test_df))
    history = pd.concat([train_df[RAW_COLUMNS], test_df[RAW_COLUMNS]], ignore_index=True)
    logger.info("TRAIN: construction des features")
    features = build_features(history)
    logger.info("TRAIN: %d features construites", features.shape[1])
    logger.info("TRAIN: entraînement du VotingClassifier (Random Forest + HGB + XGBoost)")
    model = make_model().fit(features.iloc[:len(train_df)], train_df[config.target_col])
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(features.columns)}, config.model_path)
    logger.info("TRAIN: modèle sauvegardé dans %s", config.model_path)
    return config.model_path


def test(config: ProjectConfig | None = None) -> float:
    config = config or ProjectConfig()
    logger = configure_logging(config.log_path)
    logger.info("TEST: début de la validation croisée (5 folds)")
    train_df, test_df, _ = load_data(config)
    history = pd.concat([train_df[RAW_COLUMNS], test_df[RAW_COLUMNS]], ignore_index=True)
    features = build_features(history).iloc[:len(train_df)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = train_df[config.target_col].to_numpy()
    predictions = np.empty(len(labels), dtype=labels.dtype)
    total_user_counts = pd.concat([train_df["user_id"], test_df["user_id"]]).value_counts()
    for fold_number, (train_indices, validation_indices) in enumerate(cv.split(features, labels), start=1):
        logger.info(
            "TEST: fold %d/5 démarré (%d entraînement, %d validation)",
            fold_number,
            len(train_indices),
            len(validation_indices),
        )
        fold_model = make_model().fit(features.iloc[train_indices], labels[train_indices])
        validation_probabilities = probabilities_in_class_order(fold_model, features.iloc[validation_indices])
        test_probabilities = probabilities_in_class_order(fold_model, features.iloc[len(train_df):])
        unknown_probabilities = np.vstack([validation_probabilities, test_probabilities])
        unknown_users = pd.concat(
            [train_df.iloc[validation_indices]["user_id"], test_df["user_id"]],
            ignore_index=True,
        )
        predictions[validation_indices] = np.array(CLASS_ORDER)[balance_user_predictions(
            unknown_probabilities,
            unknown_users,
            train_df.iloc[train_indices]["user_id"],
            labels[train_indices],
            total_user_counts,
        )[:len(validation_indices)]]
        fold_score = f1_score(labels[validation_indices], predictions[validation_indices], average="macro")
        logger.info("TEST: fold %d/5 terminé | Macro-F1 = %.4f", fold_number, fold_score)
    score = float(f1_score(train_df[config.target_col], predictions, average="macro"))
    metrics = {"macro_f1": score, "report": classification_report(train_df[config.target_col], predictions, output_dict=True)}
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("TEST: validation terminée | Macro-F1 global = %.4f", score)
    logger.info("TEST: métriques sauvegardées dans %s", config.metrics_path)
    print(f"macro-F1: {score:.4f}")
    print(classification_report(train_df[config.target_col], predictions))
    return score


def export(config: ProjectConfig | None = None, output: str | Path | None = None) -> Path:
    config = config or ProjectConfig()
    logger = configure_logging(config.log_path)
    logger.info("EXPORT: début de la génération de la soumission")
    if not config.model_path.exists():
        raise FileNotFoundError(f"No trained model found at {config.model_path}. Run 'train' first.")
    train_df, test_df, sample = load_data(config)
    history = pd.concat([train_df[RAW_COLUMNS], test_df[RAW_COLUMNS]], ignore_index=True)
    logger.info("EXPORT: construction des features de test")
    features = build_features(history)
    artifact = joblib.load(config.model_path)
    test_features = features.iloc[len(train_df):].reindex(columns=artifact["columns"], fill_value=0)
    probabilities = probabilities_in_class_order(artifact["model"], test_features)
    total_user_counts = pd.concat([train_df["user_id"], test_df["user_id"]]).value_counts()
    predictions = np.array(CLASS_ORDER)[balance_user_predictions(
        probabilities,
        test_df["user_id"],
        train_df["user_id"],
        train_df[config.target_col],
        total_user_counts,
    )]
    submission = sample.copy()
    submission[config.target_col] = predictions
    destination = Path(output) if output else config.artifact_dir / "submission.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    if list(submission.columns) != list(sample.columns) or len(submission) != len(sample):
        raise ValueError("Generated submission does not match sample_submission.csv")
    logger.info("EXPORT: soumission sauvegardée dans %s (%d lignes)", destination, len(submission))
    return destination
