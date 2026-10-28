from __future__ import annotations

from typing import Iterable

REQUIRED_COLUMNS = {
    "user_id",
    "sport",
    "gender",
    "hour",
    "duration",
    "distance",
    "speed",
    "max_speed",
    "elevation_gain",
    "intensity",
}


def validate_columns(columns: Iterable[str]) -> list[str]:
    """Return missing required columns for a dataset."""
    existing = set(columns)
    missing = sorted(REQUIRED_COLUMNS - existing)
    return missing
