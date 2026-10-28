from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    candidates = (
        ROOT_DIR / "data",
        ROOT_DIR / "imports" / "hackathon_ml" / "data",
        ROOT_DIR / "imports" / "Pipeline 16h28" / "data",
    )
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: Path = resolve_data_dir()
    artifact_dir: Path = ROOT_DIR / "artifacts"
    train_name: str = "train.csv"
    test_name: str = "test.csv"
    sample_name: str = "sample_submission.csv"
    target_col: str = "intensity"
    id_col: str = "workout_id"

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_name

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_name

    @property
    def sample_path(self) -> Path:
        return self.data_dir / self.sample_name

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.artifact_dir / "test_metrics.json"

    @property
    def log_path(self) -> Path:
        return self.artifact_dir / "hackathon.log"
