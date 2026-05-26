"""Resolve project root whether the caller runs from repo root or notebooks/."""
from pathlib import Path


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "main.py").is_file():
        return cwd
    if (cwd.parent / "main.py").is_file():
        return cwd.parent
    raise FileNotFoundError(
        "Cannot find project root (expected main.py in cwd or parent). "
        "Run commands from the repository root."
    )


ROOT = project_root()
DATASETS_DIR = ROOT / "datasets"
RESULT_DIR = ROOT / "result"
WEB_DIR = ROOT / "web"
