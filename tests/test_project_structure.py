from __future__ import annotations

from pathlib import Path

from hackathon_ml.config import ProjectConfig
from hackathon_ml.pipeline import RAW_COLUMNS, build_features, make_model
import pandas as pd


def test_project_has_expected_top_level_folders():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src").exists()
    assert (root / "tests").exists()
    assert (root / "README.md").exists()


def test_project_config_points_to_existing_data_files():
    cfg = ProjectConfig()
    assert cfg.train_path.exists(), "Train data file is missing"
    assert cfg.test_path.exists(), "Test data file is missing"
    assert cfg.sample_path.exists(), "Sample submission file is missing"


def test_unified_pipeline_modules_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src" / "hackathon_ml" / "__init__.py",
        root / "src" / "hackathon_ml" / "cli.py",
        root / "src" / "hackathon_ml" / "config.py",
        root / "src" / "hackathon_ml" / "pipeline.py",
    ]

    for path in required:
        assert path.exists(), f"Missing expected module: {path}"


def test_pipeline_builds_expected_features_and_model():
    config = ProjectConfig()
    train = pd.read_csv(config.train_path, nrows=4)
    features = build_features(train[RAW_COLUMNS])
    assert len(features) == len(train)
    assert "denivele_par_km" in features.columns
    assert "vmax_sur_vmoy" in features.columns
    assert "effort_speed" in features.columns
    assert "hour_sin" in features.columns
    assert "sessions_user_month" in features.columns
    model = make_model()
    assert model.voting == "soft"
    assert [name for name, _ in model.estimators] == [
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    ]
