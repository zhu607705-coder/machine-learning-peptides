from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoresearch.bootstrap import ensure_project_paths

ensure_project_paths()

import comprehensive_ml_workflow as workflow

from autoresearch.prepare import run_baseline

# Editable experiment knobs. This file is the only place autoresearch should modify.
workflow.SOURCE_DRIFT_Z_THRESHOLD = 2.0
workflow.SOURCE_DRIFT_MIN_ROWS = 3

RANDOM_SEARCH_ITERATIONS = 2
ENABLE_BAYESIAN_CALIBRATION = False
EXPERIMENT_DESCRIPTION = "baseline organic autoresearch run"


def main() -> None:
    if os.environ.get("AUTORESEARCH_IMPORT_ONLY") == "1":
        return
    summary = run_baseline(
        description=EXPERIMENT_DESCRIPTION,
        random_search_iterations=RANDOM_SEARCH_ITERATIONS,
        enable_bayesian_calibration=ENABLE_BAYESIAN_CALIBRATION,
    )
    print("---")
    print(f"primary_head:                 {summary['primaryHead']}")
    print(f"primary_r2:                   {summary['primaryR2']:.6f}")
    print(f"primary_rmse:                 {summary['primaryRmse']:.6f}")
    print(f"primary_accuracy_within_tol:  {summary['primaryAccuracyWithinTolerance']:.6f}")
    print(f"primary_decision:             {summary['primaryDecision']}")
    print(f"accepted_high_conf_heads:     {summary['acceptedHighConfidenceHeads']}")
    print(f"accepted_warning_heads:       {summary['acceptedWarningHeads']}")
    print(f"rejected_heads:               {summary['rejectedHeads']}")
    print(f"weighted_r2:                  {summary['weightedR2']:.6f}")
    print(f"weighted_rmse:                {summary['weightedRmse']:.6f}")
    print(f"total_seconds:                {summary['totalSeconds']:.1f}")
    print("prediction_policy_json:", json.dumps(summary["predictionPolicy"], ensure_ascii=False))


if __name__ == "__main__":
    main()
