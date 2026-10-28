from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score


def macro_f1_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute the macro F1 score used in the hackathon."""
    return float(f1_score(y_true, y_pred, average="macro"))
