from __future__ import annotations

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple time-based feature placeholders.

    This function is intentionally lightweight and meant as a scaffold for future
    feature engineering work.
    """
    out = df.copy()
    if "hour" in out.columns:
        out["hour_sin"] = (out["hour"] / 24.0) * 2 * 3.141592653589793
        out["hour_cos"] = out["hour_sin"].map(lambda x: __import__("math").sin(x))
    return out


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic ratio-based features."""
    out = df.copy()
    if {"distance", "duration"}.issubset(out.columns):
        out["distance_per_duration"] = out["distance"] / out["duration"].replace(0, 1)
    if {"speed", "max_speed"}.issubset(out.columns):
        out["speed_ratio"] = out["speed"] / out["max_speed"].replace(0, 1)
    return out
