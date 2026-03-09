from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from autoresearch.bootstrap import PROJECT_ROOT, ensure_project_paths

ensure_project_paths()

from comprehensive_ml_workflow import (
    apply_cleaning_pipeline,
    build_modeling_dataframe,
    parse_args as parse_workflow_args,
    run_strict_semantic_head_loso,
    stage_definitions,
)
RESULTS_TSV_PATH = PROJECT_ROOT / "results.tsv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "artifacts" / "autoresearch" / "latest-run.json"
TIME_BUDGET = 300


def make_workflow_args(
    *,
    random_search_iterations: int = 2,
    enable_bayesian_calibration: bool = False,
) -> Any:
    args = parse_workflow_args()
    args.random_search_iterations = int(random_search_iterations)
    args.enable_bayesian_calibration = bool(enable_bayesian_calibration)
    return args


def current_commit_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def ensure_results_header(path: Path = RESULTS_TSV_PATH) -> None:
    if path.exists():
        return
    path.write_text(
        "commit\tprimary_r2\tprimary_rmse\thigh_conf_heads\tstatus\tdescription\n",
        encoding="utf-8",
    )


def summarize_report(strict_loso: dict[str, Any], cleaning: dict[str, Any], total_seconds: float) -> dict[str, Any]:
    primary_head = next(
        (
            row
            for row in strict_loso["headLevelMetrics"]
            if row["target"] == "yield_target" and row["head"] == "isolated|isolated"
        ),
        None,
    )
    return {
        "primaryHead": "yield_target/isolated|isolated",
        "primaryRmse": None if primary_head is None else float(primary_head["metrics"]["rmse"]),
        "primaryR2": None if primary_head is None else float(primary_head["metrics"]["r2"]),
        "primaryAccuracyWithinTolerance": None if primary_head is None else float(primary_head["metrics"]["accuracyWithinTolerance"]),
        "primaryDecision": None if primary_head is None else primary_head["decision"]["confidenceTier"],
        "acceptedHighConfidenceHeads": int(len(strict_loso["predictionPolicy"]["acceptedHighConfidenceHeads"])),
        "acceptedWarningHeads": int(len(strict_loso["predictionPolicy"]["acceptedWarningHeads"])),
        "rejectedHeads": int(len(strict_loso["predictionPolicy"]["rejectedHeads"])),
        "weightedR2": float(strict_loso["weightedHeadSummary"]["weightedR2"]),
        "weightedRmse": float(strict_loso["weightedHeadSummary"]["weightedRmse"]),
        "strictCleaning": cleaning,
        "strictSemanticFilter": strict_loso["strictSemanticFilter"],
        "predictionPolicy": strict_loso["predictionPolicy"],
        "headLevelMetrics": strict_loso["headLevelMetrics"],
        "sourceDiagnostics": strict_loso.get("sourceDiagnostics", []),
        "totalSeconds": float(total_seconds),
    }


def append_results_row(summary: dict[str, Any], description: str, path: Path = RESULTS_TSV_PATH) -> None:
    ensure_results_header(path)
    status = "keep" if summary["acceptedHighConfidenceHeads"] > 0 else "warning"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{current_commit_short()}\t"
            f"{0.0 if summary['primaryR2'] is None else summary['primaryR2']:.6f}\t"
            f"{0.0 if summary['primaryRmse'] is None else summary['primaryRmse']:.6f}\t"
            f"{summary['acceptedHighConfidenceHeads']}\t"
            f"{status}\t"
            f"{description}\n"
        )


def save_summary(summary: dict[str, Any], output_path: Path = OUTPUT_JSON_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run_baseline(
    *,
    description: str = "baseline strict-head run",
    random_search_iterations: int = 2,
    enable_bayesian_calibration: bool = False,
) -> dict[str, Any]:
    args = make_workflow_args(
        random_search_iterations=random_search_iterations,
        enable_bayesian_calibration=enable_bayesian_calibration,
    )
    start = time.perf_counter()
    base_df = build_modeling_dataframe()
    strict_stage = next(item for item in stage_definitions() if item.name == "final_tuned_ensemble")
    strict_df, strict_cleaning = apply_cleaning_pipeline(
        base_df,
        strict_stage,
        similarity_threshold=args.sequence_similarity_threshold,
    )
    strict_loso = run_strict_semantic_head_loso(strict_df, strict_stage, args)
    summary = summarize_report(strict_loso, strict_cleaning.__dict__, time.perf_counter() - start)
    save_summary(summary)
    append_results_row(summary, description)
    return summary
