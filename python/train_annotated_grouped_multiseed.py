from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

from annotated_training_utils import (
    FEATURE_COLUMNS,
    build_group_labels,
    build_regression_strata,
    drop_low_variance_sources,
    winsorize_target,
)
from annotated_source_router import SourceRoutedRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "annotated_training_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "annotated_grouped_review"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "annotated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped multi-seed review for annotated peptide training data.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260306, 20260316, 20260326, 20260336, 20260346])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-group-bins", type=int, default=8)
    parser.add_argument("--yield-strata-bins", type=int, default=3)
    parser.add_argument("--winsor-lower", type=float, default=0.02)
    parser.add_argument("--winsor-upper", type=float, default=0.98)
    return parser.parse_args()


def model_factories(seed: int) -> dict[str, Callable[[], object]]:
    return {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        ),
        "SourceRouted": lambda: SourceRoutedRegressor(
            feature_columns=FEATURE_COLUMNS,
            model_factory=lambda: GradientBoostingRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                random_state=seed,
            ),
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    accuracy_5 = float(np.mean(np.abs(y_true - y_pred) <= 5.0))
    accuracy_10 = float(np.mean(np.abs(y_true - y_pred) <= 10.0))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "accuracy_5": accuracy_5,
        "accuracy_10": accuracy_10,
    }


def summarize_runs(run_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ["rmse", "mae", "r2", "accuracy_5", "accuracy_10"]:
        values = np.array([item[metric_name] for item in run_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
    return summary


def evaluate_multiseed(
    frame: pd.DataFrame,
    target_column: str,
    seeds: list[int],
    n_splits: int,
    n_group_bins: int,
    strata_bins: int,
) -> dict[str, object]:
    working = frame.dropna(subset=[target_column]).copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
    working = working.dropna(subset=[target_column]).copy()
    working["group_id"] = build_group_labels(working, n_bins=n_group_bins)
    working["stratum"] = build_regression_strata(working[target_column], n_bins=strata_bins)

    x = working[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = working[target_column].to_numpy(dtype=float)
    groups = working["group_id"].to_numpy()
    strata = working["stratum"].to_numpy()

    effective_splits = min(n_splits, int(pd.Series(groups).nunique()))
    if effective_splits < 2:
        raise ValueError(f"Not enough unique groups for grouped CV: {pd.Series(groups).nunique()}")

    model_names = tuple(model_factories(seeds[0]).keys())
    aggregate: dict[str, list[dict[str, float]]] = {name: [] for name in model_names}
    seed_summaries: list[dict[str, object]] = []

    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        per_model: dict[str, list[dict[str, float]]] = {name: [] for name in model_names}
        for train_idx, test_idx in splitter.split(x, strata, groups):
            x_train = x[train_idx]
            y_train = y[train_idx]
            x_test = x[test_idx]
            y_test = y[test_idx]
            train_frame = working.iloc[train_idx].copy()
            test_frame = working.iloc[test_idx].copy()

            for model_name, factory in model_factories(seed).items():
                model = factory()
                if model_name == "SourceRouted":
                    model.fit(train_frame, pd.Series(y_train, index=train_frame.index))
                    predictions = model.predict(test_frame)
                else:
                    model.fit(x_train, y_train)
                    predictions = model.predict(x_test)
                metrics = regression_metrics(y_test, predictions)
                per_model[model_name].append(metrics)
                aggregate[model_name].append(metrics)

        seed_summaries.append(
            {
                "seed": seed,
                "models": {model_name: summarize_runs(metrics) for model_name, metrics in per_model.items()},
            }
        )

    overall = {model_name: summarize_runs(metrics) for model_name, metrics in aggregate.items()}
    best_model = min(overall, key=lambda name: overall[name]["rmse_mean"])
    return {
        "rows": int(len(working)),
        "groups": int(working["group_id"].nunique()),
        "strataCounts": working["stratum"].value_counts().sort_index().to_dict(),
        "groupCountsTop10": working["group_id"].value_counts().head(10).to_dict(),
        "seedSummaries": seed_summaries,
        "overallModels": overall,
        "bestModel": best_model,
    }


def fit_final_model(frame: pd.DataFrame, target_column: str, model_name: str) -> object:
    seed = 20260306
    model = model_factories(seed)[model_name]()
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame[target_column].to_numpy(dtype=float)
    if model_name == "SourceRouted":
        model.fit(frame, pd.Series(y, index=frame.index))
        return model
    model.fit(x, y)
    return model


def build_cleaned_target_review(
    frame: pd.DataFrame,
    target_column: str,
    strata_bins: int,
    seeds: list[int],
    n_splits: int,
    n_group_bins: int,
    winsor_lower: float,
    winsor_upper: float,
    model_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    cleaned, variance_stats = drop_low_variance_sources(frame, target_column=target_column)
    cleaned = cleaned.copy()
    clipped_target, winsor_stats = winsorize_target(
        cleaned[target_column],
        lower_quantile=winsor_lower,
        upper_quantile=winsor_upper,
    )
    cleaned[target_column] = clipped_target
    cleaned[f"{target_column}_stratum"] = build_regression_strata(cleaned[target_column], n_bins=strata_bins)

    cleaned_review = evaluate_multiseed(
        frame=cleaned,
        target_column=target_column,
        seeds=seeds,
        n_splits=n_splits,
        n_group_bins=n_group_bins,
        strata_bins=strata_bins,
    )

    final_model_name = cleaned_review["bestModel"]
    final_model = fit_final_model(cleaned, target_column=target_column, model_name=final_model_name)
    final_model_path = model_dir / f"{target_column}_cleaned_{final_model_name.lower()}.joblib"
    joblib.dump(final_model, final_model_path)

    cleaned_dataset_path = output_dir / f"{target_column}_cleaned_training_data.csv"
    cleaned.to_csv(cleaned_dataset_path, index=False)

    return {
        "cleaning": {
            "inputRows": int(frame[target_column].notna().sum()),
            "outputRows": int(len(cleaned)),
            "varianceFilter": variance_stats,
            "winsorization": winsor_stats,
            "strataCounts": cleaned[f"{target_column}_stratum"].value_counts().sort_index().to_dict(),
        },
        "retrain": {
            "review": cleaned_review,
            "finalModel": final_model_name,
            "finalModelPath": str(final_model_path),
            "cleanedDatasetPath": str(cleaned_dataset_path),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.dataset)

    purity_review = evaluate_multiseed(
        frame=frame,
        target_column="purity",
        seeds=args.seeds,
        n_splits=args.n_splits,
        n_group_bins=args.n_group_bins,
        strata_bins=args.yield_strata_bins,
    )
    yield_review = evaluate_multiseed(
        frame=frame,
        target_column="yield_val",
        seeds=args.seeds,
        n_splits=args.n_splits,
        n_group_bins=args.n_group_bins,
        strata_bins=args.yield_strata_bins,
    )

    final_purity_model_name = purity_review["bestModel"]
    final_purity_model = fit_final_model(frame.dropna(subset=["purity"]).copy(), target_column="purity", model_name=final_purity_model_name)
    final_purity_model_path = model_dir / f"purity_full_{final_purity_model_name.lower()}.joblib"
    joblib.dump(final_purity_model, final_purity_model_path)

    final_yield_full_model_name = yield_review["bestModel"]
    final_yield_full_model = fit_final_model(frame.dropna(subset=["yield_val"]).copy(), target_column="yield_val", model_name=final_yield_full_model_name)
    final_yield_full_model_path = model_dir / f"yield_val_full_{final_yield_full_model_name.lower()}.joblib"
    joblib.dump(final_yield_full_model, final_yield_full_model_path)

    purity_cleaned = build_cleaned_target_review(
        frame=frame,
        target_column="purity",
        strata_bins=args.yield_strata_bins,
        seeds=args.seeds,
        n_splits=args.n_splits,
        n_group_bins=args.n_group_bins,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        model_dir=model_dir,
        output_dir=args.output_dir,
    )
    yield_cleaned = build_cleaned_target_review(
        frame=frame,
        target_column="yield_val",
        strata_bins=args.yield_strata_bins,
        seeds=args.seeds,
        n_splits=args.n_splits,
        n_group_bins=args.n_group_bins,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        model_dir=model_dir,
        output_dir=args.output_dir,
    )

    report = {
        "dataset": str(args.dataset),
        "seeds": args.seeds,
        "nSplits": args.n_splits,
        "nGroupBins": args.n_group_bins,
        "purityGroupedReview": purity_review,
        "yieldGroupedReview": yield_review,
        "fullDatasetBestModels": {
            "purity": {
                "model": final_purity_model_name,
                "path": str(final_purity_model_path),
            },
            "yield_val": {
                "model": final_yield_full_model_name,
                "path": str(final_yield_full_model_path),
            },
        },
        "purityCleaning": purity_cleaned["cleaning"],
        "purityRetrain": purity_cleaned["retrain"],
        "yieldCleaning": yield_cleaned["cleaning"],
        "yieldRetrain": yield_cleaned["retrain"],
    }

    report_path = args.output_dir / "annotated_grouped_multiseed_review.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
