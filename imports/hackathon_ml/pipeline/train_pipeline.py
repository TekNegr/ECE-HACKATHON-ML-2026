from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from hackathon_ml.models.training import build_baseline_model
from hackathon_ml.pipeline.preprocessing import build_preprocessor


def make_training_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Create a minimal sklearn pipeline for training."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_baseline_model()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def fit_training_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    pipeline = make_training_pipeline(numeric_features, categorical_features)
    pipeline.fit(X, y)
    return pipeline
