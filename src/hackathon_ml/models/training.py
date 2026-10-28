from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def build_baseline_model() -> RandomForestClassifier:
    """Create a simple baseline model placeholder."""
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )


def build_voting_model() -> VotingClassifier:
    """Create a soft-voting ensemble scaffold."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    hgb = HistGradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=2000)
    return VotingClassifier(
        estimators=[
            ("rf", rf),
            ("hgb", hgb),
            ("lr", lr),
        ],
        voting="soft",
    )


def train_model(model, X: pd.DataFrame, y: pd.Series):
    """Fit a model and return it."""
    return model.fit(X, y)
