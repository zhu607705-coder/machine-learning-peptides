from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"


def ensure_project_paths() -> None:
    root = str(PROJECT_ROOT)
    python_dir = str(PYTHON_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
