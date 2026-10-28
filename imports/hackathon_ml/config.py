from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    candidates = [
        ROOT_DIR / "data",
        ROOT_DIR / "src" / "hackathon_ml" / "data",
        Path(__file__).resolve().parent / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_DIR = resolve_data_dir()


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: Path = DATA_DIR
    train_path: Path = DATA_DIR / "train.csv"
    test_path: Path = DATA_DIR / "test.csv"
    sample_submission_path: Path = DATA_DIR / "sample_submission.csv"
    target_col: str = "intensity"
    id_col: str = "workout_id"
