from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = PROJECT_ROOT / "archive" / "tooling" / ".venv-pymc" / "bin" / "python"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "python"
    env["AUTORESEARCH_IMPORT_ONLY"] = "1"
    return env


def test_autoresearch_train_entrypoint_imports_cleanly() -> None:
    result = subprocess.run(
        [str(PYTHON_BIN), "autoresearch/train.py"],
        cwd=PROJECT_ROOT,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_autoresearch_wrapper_stays_compatible() -> None:
    result = subprocess.run(
        [str(PYTHON_BIN), "python/autoresearch_train.py"],
        cwd=PROJECT_ROOT,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
