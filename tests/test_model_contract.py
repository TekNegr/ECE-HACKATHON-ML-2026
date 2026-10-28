from __future__ import annotations

import pandas as pd

from hackathon_ml.models.training import build_baseline_model, build_voting_model


def test_baseline_model_can_fit_and_predict():
    X = pd.DataFrame(
        {
            "hour": [9, 18, 20, 7],
            "duration": [30, 45, 50, 25],
            "distance": [2.0, 4.0, 5.0, 1.5],
        }
    )
    y = pd.Series(["easy", "moderate", "hard", "easy"])

    model = build_baseline_model()
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    set(preds).issubset({"easy", "moderate", "hard"})


def test_voting_model_can_be_built():
    model = build_voting_model()
    assert model.voting == "soft"
    assert len(model.estimators) == 3
