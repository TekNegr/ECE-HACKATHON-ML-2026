from __future__ import annotations

from hackathon_ml.data.schema import REQUIRED_COLUMNS, validate_columns


def test_required_columns_are_defined():
    assert "user_id" in REQUIRED_COLUMNS
    assert "sport" in REQUIRED_COLUMNS
    assert "intensity" in REQUIRED_COLUMNS


def test_validate_columns_detects_missing_fields():
    columns = ["user_id", "sport", "gender"]
    missing = validate_columns(columns)
    assert "hour" in missing
    assert "intensity" in missing
