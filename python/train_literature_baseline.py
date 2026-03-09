from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "real" / "final_purity_yield_literature.csv"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "literature-baseline-report.json"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
HYDROPHOBIC = set("AVILFWYMP")
CHARGED = set("DEKRH")
AROMATIC = set("FWYH")
POLAR = set("NQSTYC")
BREAKERS = set("GP")
SULFUR = set("CM")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_column: str
    filters: dict[str, str]
    include_stage_features: bool


def load_rows() -> list[dict[str, str]]:
    with DATASET_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_numeric_label(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None

    cleaned = raw.replace("%", "").replace(",", ".").strip()
    cleaned = cleaned.lstrip("<>~= ")
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_sequence(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""

    if set(raw).issubset(AA_SET):
        return raw

    residues: list[str] = []
    for token in raw.split("-"):
        token = token.strip()
        if not token or token in {"H", "NH2", "OH", "OMe", "SBzl"}:
            continue
        key = token[:3]
        if key in THREE_TO_ONE:
            residues.append(THREE_TO_ONE[key])
    return "".join(residues)


def fraction(sequence: str, alphabet: Iterable[str]) -> float:
    if not sequence:
        return 0.0
    alphabet_set = set(alphabet)
    return sum(1 for residue in sequence if residue in alphabet_set) / len(sequence)


def build_feature_vector(row: dict[str, str], include_stage_features: bool) -> np.ndarray:
    sequence = row.get("_sequence_norm", "") or normalize_sequence(row["sequence"])
    counts = np.array([sequence.count(residue) for residue in AA_ORDER], dtype=float)
    length = max(len(sequence), 1)
    fractions = counts / length

    features = [
        float(len(sequence)),
        float(len(set(sequence))) / length,
        fraction(sequence, HYDROPHOBIC),
        fraction(sequence, CHARGED),
        fraction(sequence, AROMATIC),
        fraction(sequence, POLAR),
        fraction(sequence, BREAKERS),
        fraction(sequence, SULFUR),
    ]
    features.extend(fractions.tolist())

    topology = row["topology"].strip().lower()
    features.extend([
        1.0 if topology == "linear" else 0.0,
        1.0 if "cyclic" in topology or "disulfide" in topology else 0.0,
    ])

    if include_stage_features:
        purity_stage = row["purity_stage"].strip().lower()
        yield_stage = row["yield_stage"].strip().lower()
        features.extend([
            1.0 if purity_stage == "final_product" else 0.0,
            1.0 if purity_stage.startswith("crude") else 0.0,
            1.0 if purity_stage.startswith("purified") else 0.0,
            1.0 if yield_stage == "isolated" else 0.0,
            1.0 if yield_stage == "crude" else 0.0,
            1.0 if yield_stage == "recovery" else 0.0,
        ])

    return np.array(features, dtype=float)


def filter_rows(rows: list[dict[str, str]], spec: TaskSpec) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        sequence = normalize_sequence(row["sequence"])
        if not sequence:
            continue
        target_value = parse_numeric_label(row.get(spec.target_column, ""))
        if target_value is None:
            continue
        if any(row.get(column) != value for column, value in spec.filters.items()):
            continue
        filtered.append({**row, "_sequence_norm": sequence, "_target_value": target_value})
    return filtered


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> float:
    x_train, x_test = standardize(train_x, test_x)
    x_train_aug = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    x_test_aug = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    penalty = np.eye(x_train_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train_aug.T @ x_train_aug + penalty, x_train_aug.T @ train_y)
    return float((x_test_aug @ beta)[0])


def mean_baseline(train_y: np.ndarray) -> float:
    return float(np.mean(train_y))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean(np.square(y_pred - y_true))))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 if denom == 0 else 1.0 - (float(np.sum(np.square(y_pred - y_true))) / denom)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def leave_one_source_out(
    filtered: list[dict[str, str]],
    x: np.ndarray,
    y: np.ndarray,
    alpha_grid: list[float],
) -> dict[str, object]:
    sources = sorted({row["source_id"] for row in filtered})
    if len(sources) < 2:
        return {"status": "single_source_only", "sources": sources}

    best_result: dict[str, object] | None = None
    for alpha in alpha_grid:
        preds = []
        trues = []
        for source in sources:
            test_idx = [index for index, row in enumerate(filtered) if row["source_id"] == source]
            train_idx = [index for index, row in enumerate(filtered) if row["source_id"] != source]
            if not train_idx or not test_idx:
                continue
            for index in test_idx:
                preds.append(ridge_predict(x[train_idx], y[train_idx], x[index:index + 1], alpha))
                trues.append(y[index])
        metrics = regression_metrics(np.array(trues), np.array(preds))
        candidate = {"bestAlpha": alpha, "sources": sources, "metrics": metrics}
        if best_result is None or metrics["rmse"] < best_result["metrics"]["rmse"]:
            best_result = candidate

    assert best_result is not None
    return best_result


def loocv_task(rows: list[dict[str, str]], spec: TaskSpec, alpha_grid: list[float]) -> dict[str, object]:
    filtered = filter_rows(rows, spec)
    if len(filtered) < 8:
        return {
            "task": spec.name,
            "rows": len(filtered),
            "status": "too_few_rows",
        }

    x = np.stack([build_feature_vector(row, spec.include_stage_features) for row in filtered], axis=0)
    y = np.array([float(row["_target_value"]) for row in filtered], dtype=float)

    alpha_scores = []
    for alpha in alpha_grid:
        preds = []
        baseline_preds = []
        for index in range(len(filtered)):
            train_mask = np.ones(len(filtered), dtype=bool)
            train_mask[index] = False
            preds.append(ridge_predict(x[train_mask], y[train_mask], x[index:index + 1], alpha))
            baseline_preds.append(mean_baseline(y[train_mask]))
        alpha_scores.append((alpha, regression_metrics(y, np.array(preds))["rmse"], preds, baseline_preds))

    alpha_scores.sort(key=lambda item: item[1])
    best_alpha, _, best_preds, baseline_preds = alpha_scores[0]

    model_metrics = regression_metrics(y, np.array(best_preds))
    baseline_metrics = regression_metrics(y, np.array(baseline_preds))

    return {
        "task": spec.name,
        "rows": len(filtered),
        "bestAlpha": best_alpha,
        "modelMetrics": model_metrics,
        "baselineMetrics": baseline_metrics,
        "leaveOneSourceOut": leave_one_source_out(filtered, x, y, alpha_grid),
        "improvement": {
            "rmseDelta": baseline_metrics["rmse"] - model_metrics["rmse"],
            "maeDelta": baseline_metrics["mae"] - model_metrics["mae"],
        },
        "targetMean": float(np.mean(y)),
        "targetStd": float(np.std(y)),
        "sources": sorted({row["source_id"] for row in filtered}),
    }


def main() -> None:
    rows = load_rows()
    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    tasks = [
        TaskSpec(
            name="isolated_yield_sequence_only",
            target_column="yield_pct",
            filters={"yield_stage": "isolated"},
            include_stage_features=False,
        ),
        TaskSpec(
            name="crude_yield_sequence_only",
            target_column="yield_pct",
            filters={"yield_stage": "crude"},
            include_stage_features=False,
        ),
        TaskSpec(
            name="recovery_yield_sequence_only",
            target_column="yield_pct",
            filters={"yield_stage": "recovery"},
            include_stage_features=False,
        ),
        TaskSpec(
            name="purified_purity_sequence_only",
            target_column="purity_pct",
            filters={"purity_stage": "purified_hplc"},
            include_stage_features=False,
        ),
        TaskSpec(
            name="purity_mixed_sequence_plus_stage",
            target_column="purity_pct",
            filters={},
            include_stage_features=True,
        ),
    ]

    report = {
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "tasks": [loocv_task(rows, spec, alpha_grid) for spec in tasks],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
