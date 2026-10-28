from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parents[2]
