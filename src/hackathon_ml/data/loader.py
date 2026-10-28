from __future__ import annotations

from pathlib import Path

import pandas as pd

from hackathon_ml.config import ProjectConfig
from hackathon_ml.data.schema import validate_columns


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and validate its expected columns."""
    df = pd.read_csv(path)
    missing = validate_columns(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def load_training_data(config: ProjectConfig | None = None) -> pd.DataFrame:
    config = config or ProjectConfig()
    return load_dataset(config.train_path)


def load_test_data(config: ProjectConfig | None = None) -> pd.DataFrame:
    config = config or ProjectConfig()
    return load_dataset(config.test_path)
