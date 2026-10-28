from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from hackathon_ml.config import ProjectConfig


def build_model_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def generate_submission(config: ProjectConfig | None = None) -> pd.DataFrame:
    config = config or ProjectConfig()

    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)

    if config.id_col in train_df.columns:
        train_df = train_df.drop(columns=[config.id_col])

    feature_cols = [col for col in train_df.columns if col != config.target_col]
    X_train = train_df[feature_cols]
    y_train = train_df[config.target_col]

    pipeline = build_model_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    X_test = test_df[feature_cols]
    predictions = pipeline.predict(X_test)

    submission = pd.DataFrame({config.id_col: test_df[config.id_col], config.target_col: predictions})
    submission.to_csv(config.sample_submission_path, index=False)
    return submission


def main() -> None:
    generate_submission()
    print(f"Submission saved to: {ProjectConfig().sample_submission_path}")


if __name__ == "__main__":
    main()
