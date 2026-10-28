from __future__ import annotations

from pathlib import Path

from hackathon_ml.config import ProjectConfig


def test_project_has_expected_top_level_folders():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src").exists()
    assert (root / "tests").exists()
    assert (root / "README.md").exists()


def test_project_config_points_to_existing_data_files():
    cfg = ProjectConfig()
    assert cfg.train_path.exists(), "Train data file is missing"
    assert cfg.test_path.exists(), "Test data file is missing"


def test_package_modules_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src" / "hackathon_ml" / "config.py",
        root / "src" / "hackathon_ml" / "data" / "schema.py",
        root / "src" / "hackathon_ml" / "features" / "engineering.py",
        root / "src" / "hackathon_ml" / "models" / "training.py",
        root / "src" / "hackathon_ml" / "pipeline" / "train_pipeline.py",
        root / "src" / "hackathon_ml" / "evaluation" / "metrics.py",
    ]

    for path in required:
        assert path.exists(), f"Missing expected module: {path}"
