from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from meta_isolated_yield_analysis import build_features, normalize_sequence, parse_numeric_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "real" / "final_purity_yield_literature.csv"
JSON_REPORT_PATH = PROJECT_ROOT / "artifacts" / "fmoc-sop-subset-analysis.json"
MARKDOWN_REPORT_PATH = PROJECT_ROOT / "docs" / "fmoc-sop-subset-analysis.md"

FEATURE_COLUMNS = [
    "length",
    "avg_hydrophobicity",
    "total_charge",
    "max_coupling_difficulty",
    "hydrophobic_ratio",
    "aromatic_ratio",
    "molecular_weight",
    "longest_hydrophobic_run_norm",
]

EXCLUDED_TERMS = [
    "boc",
    "afps",
    "stapled",
    "sulfotyrosine",
    "sulfated",
    "peg23",
    "pegyl",
    "palmitoyl",
    "hopo",
    "solvent-less",
    "ram ",
    "ligation",
    "fragment synthesis",
    "microwave",
]


def load_rows() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(DATASET_PATH.open()))
    subset: list[dict[str, Any]] = []
    for row in rows:
        condition = str(row.get("condition_summary") or "").lower()
        topology = str(row.get("topology") or "").strip().lower()
        source_id = str(row.get("source_id") or "")
        sequence_norm = normalize_sequence(row.get("sequence", ""))
        is_fmoc = "fmoc" in condition
        is_linear = topology in {"linear", ""}
        is_excluded = any(term in condition or term in source_id for term in EXCLUDED_TERMS)
        if not (is_fmoc and is_linear and not is_excluded and sequence_norm):
            continue
        features = build_features(sequence_norm)
        payload = dict(row)
        payload.update(features)
        payload["sequence_norm"] = sequence_norm
        payload["purity_value"] = parse_numeric_label(row.get("purity_pct"))
        payload["yield_value"] = parse_numeric_label(row.get("yield_pct"))
        subset.append(payload)
    return subset


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_grouped_ridge(rows: list[dict[str, Any]], target_key: str) -> dict[str, Any]:
    usable = [row for row in rows if row[target_key] is not None]
    sources = sorted({row["source_id"] for row in usable})
    predictions: list[float] = []
    truths: list[float] = []
    baseline_predictions: list[float] = []
    per_source: list[dict[str, Any]] = []

    for source in sources:
        train = [row for row in usable if row["source_id"] != source]
        test = [row for row in usable if row["source_id"] == source]
        x_train = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in train], dtype=float)
        x_test = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in test], dtype=float)
        y_train = np.asarray([row[target_key] for row in train], dtype=float)
        y_test = np.asarray([row[target_key] for row in test], dtype=float)
        scaler = StandardScaler().fit(x_train)
        model = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
        y_pred = model.predict(scaler.transform(x_test))
        source_baseline = np.full_like(y_test, float(np.mean(y_train)), dtype=float)

        predictions.extend(y_pred.tolist())
        truths.extend(y_test.tolist())
        baseline_predictions.extend(source_baseline.tolist())
        per_source.append(
            {
                "source_id": source,
                "n_test": int(len(test)),
                "ridge_rmse": rmse(y_test, y_pred),
                "baseline_rmse": rmse(y_test, source_baseline),
                "ridge_mae": float(mean_absolute_error(y_test, y_pred)),
                "baseline_mae": float(mean_absolute_error(y_test, source_baseline)),
            }
        )

    y_true = np.asarray(truths, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    y_base = np.asarray(baseline_predictions, dtype=float)
    return {
        "n": int(len(y_true)),
        "sources": sources,
        "ridge": {
            "rmse": rmse(y_true, y_pred),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        },
        "mean_baseline": {
            "rmse": rmse(y_true, y_base),
            "mae": float(mean_absolute_error(y_true, y_base)),
            "r2": float(r2_score(y_true, y_base)) if len(y_true) > 1 else float("nan"),
        },
        "per_source": per_source,
    }


def centered_correlations(rows: list[dict[str, Any]], target_key: str) -> dict[str, float]:
    usable = [dict(row) for row in rows if row[target_key] is not None]
    source_means = {
        source: float(np.mean([row[target_key] for row in usable if row["source_id"] == source]))
        for source in sorted({row["source_id"] for row in usable})
    }
    for row in usable:
        row[f"{target_key}_centered"] = float(row[target_key]) - source_means[row["source_id"]]
    correlations: dict[str, float] = {}
    y = np.asarray([row[f"{target_key}_centered"] for row in usable], dtype=float)
    for column in FEATURE_COLUMNS:
        x = np.asarray([row[column] for row in usable], dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            correlations[column] = 0.0
        else:
            correlations[column] = float(np.corrcoef(x, y)[0, 1])
    return dict(sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True))


def source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for source in sorted({row["source_id"] for row in rows}):
        group = [row for row in rows if row["source_id"] == source]
        summary.append(
            {
                "source_id": source,
                "n": int(len(group)),
                "mean_length": float(np.mean([row["length"] for row in group])),
                "mean_purity": float(np.mean([row["purity_value"] for row in group if row["purity_value"] is not None])),
                "mean_yield": float(np.mean([row["yield_value"] for row in group if row["yield_value"] is not None])),
                "yield_stage_counts": dict(Counter(row["yield_stage"] for row in group)),
                "purity_stage_counts": dict(Counter(row["purity_stage"] for row in group)),
            }
        )
    return summary


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    purity_rows = sum(row["purity_value"] is not None for row in rows)
    yield_rows = sum(row["yield_value"] is not None for row in rows)
    report = {
        "filter_definition": {
            "include": [
                "condition_summary contains 'Fmoc'",
                "topology is linear or empty",
                "sequence can be normalized into a standard amino-acid sequence",
            ],
            "exclude_terms": EXCLUDED_TERMS,
        },
        "dataset_summary": {
            "rows": int(len(rows)),
            "sources": sorted({row["source_id"] for row in rows}),
            "source_count": int(len({row["source_id"] for row in rows})),
            "purity_rows": int(purity_rows),
            "yield_rows": int(yield_rows),
            "yield_stage_counts": dict(Counter(row["yield_stage"] for row in rows)),
            "purity_stage_counts": dict(Counter(row["purity_stage"] for row in rows)),
        },
        "source_summary": source_summary(rows),
        "grouped_purity": evaluate_grouped_ridge(rows, "purity_value"),
        "grouped_yield": evaluate_grouped_ridge(rows, "yield_value"),
        "centered_correlations": {
            "purity": centered_correlations(rows, "purity_value"),
            "yield": centered_correlations(rows, "yield_value"),
        },
    }
    return report


def write_markdown(report: dict[str, Any]) -> None:
    summary = report["dataset_summary"]
    lines = [
        "# Fmoc SOP Subset Analysis",
        "",
        "## Filter definition",
        "- Keep rows whose `condition_summary` explicitly contains `Fmoc`.",
        "- Keep linear/blank-topology rows only.",
        "- Exclude rows and sources containing: " + ", ".join(report["filter_definition"]["exclude_terms"]),
        "- Keep only sequence-resolved rows after sequence normalization.",
        "",
        "## Dataset summary",
        f"- Rows: {summary['rows']}",
        f"- Sources: {summary['source_count']} -> {', '.join(summary['sources'])}",
        f"- Yield stages: {summary['yield_stage_counts']}",
        f"- Purity stages: {summary['purity_stage_counts']}",
        "",
        "## Source summary",
        "| Source | n | Mean length | Mean purity | Mean yield |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["source_summary"]:
        lines.append(
            f"| {row['source_id']} | {row['n']} | {row['mean_length']:.2f} | {row['mean_purity']:.2f} | {row['mean_yield']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Grouped evaluation",
            f"- Purity grouped Ridge RMSE: {report['grouped_purity']['ridge']['rmse']:.3f}",
            f"- Purity mean-baseline RMSE: {report['grouped_purity']['mean_baseline']['rmse']:.3f}",
            f"- Yield grouped Ridge RMSE: {report['grouped_yield']['ridge']['rmse']:.3f}",
            f"- Yield mean-baseline RMSE: {report['grouped_yield']['mean_baseline']['rmse']:.3f}",
            "",
            "## Within-source centered correlations",
            "### Yield",
        ]
    )
    for feature, value in report["centered_correlations"]["yield"].items():
        lines.append(f"- {feature}: {value:.3f}")
    lines.extend(["", "### Purity"])
    for feature, value in report["centered_correlations"]["purity"].items():
        lines.append(f"- {feature}: {value:.3f}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Relative to the SOP-like mean baseline, grouped Ridge improves yield prediction noticeably, indicating that shrinking the chemistry space toward the current Fmoc SOP does recover some stable signal.",
            "- Purity remains difficult because this subset still mixes `crude_hplc` and `final_product` semantics across different studies.",
            "- After centering by source mean, no single sequence feature dominates; this suggests the current SOP-aligned literature subset is directionally useful but still too small for strong mechanistic claims.",
        ]
    )
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    report = build_report(rows)
    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report)


if __name__ == "__main__":
    main()
