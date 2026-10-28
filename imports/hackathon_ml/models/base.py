from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Abstract base class for hackathon models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
