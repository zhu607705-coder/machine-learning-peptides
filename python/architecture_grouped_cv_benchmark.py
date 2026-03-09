from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from train_real_model import (
    ModelConfig,
    build_candidate_configs,
    build_step_datasets_for_config,
    candidate_seed_offset,
    configure_runtime,
    load_real_dataset,
    project_root,
    train_candidate,
)


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def split_train_val_by_group(
    train_df: pd.DataFrame,
    *,
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    serials = sorted(train_df["serial"].astype(str).unique().tolist())
    if len(serials) < 2:
        raise ValueError("Need at least two serial groups to split train/validation")
    rng = np.random.default_rng(seed)
    rng.shuffle(serials)
    val_count = int(round(len(serials) * val_ratio))
    val_count = max(1, min(len(serials) - 1, val_count))
    val_serials = set(serials[:val_count])
    is_val = train_df["serial"].astype(str).isin(val_serials)
    val_df = train_df[is_val].copy()
    fit_df = train_df[~is_val].copy()
    return fit_df, val_df


def select_architecture_configs(
    *,
    architectures: list[str],
    huber_delta: float,
    epoch_scale: float,
) -> dict[str, ModelConfig]:
    selected: dict[str, ModelConfig] = {}
    for config in build_candidate_configs([huber_delta]):
        if config.architecture not in architectures:
            continue
        if config.architecture in selected:
            continue
        if epoch_scale != 1.0:
            max_epochs = max(6, int(round(config.max_epochs * epoch_scale)))
            patience = max(3, int(round(config.patience * epoch_scale)))
            config = replace(config, max_epochs=max_epochs, patience=patience)
        selected[config.architecture] = config
    missing = [name for name in architectures if name not in selected]
    if missing:
        raise ValueError(f"Unknown architecture(s): {', '.join(missing)}")
    return selected


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for architecture, group in frame.groupby("architecture"):
        summary[architecture] = {
            "nRuns": int(len(group)),
            "validationCombinedRmseMean": float(group["validationCombinedRmse"].mean()),
            "validationCombinedRmseStd": float(group["validationCombinedRmse"].std(ddof=1)) if len(group) > 1 else 0.0,
            "testCombinedRmseMean": float(group["testCombinedRmse"].mean()),
            "testCombinedRmseStd": float(group["testCombinedRmse"].std(ddof=1)) if len(group) > 1 else 0.0,
        }

    if "gru" in summary:
        gru_stats = summary["gru"]
        gru_frame = frame[frame["architecture"] == "gru"].set_index(["seed", "fold"])
        for architecture, group in frame.groupby("architecture"):
            if architecture == "gru":
                summary[architecture]["vsGru"] = {
                    "testCombinedRmseMeanDelta": 0.0,
                    "testCombinedRmseStdDelta": 0.0,
                    "pairwiseWinRate": 0.0,
                    "stableBetterThanGru": False,
                }
                continue
            merged = group.set_index(["seed", "fold"]).join(
                gru_frame[["testCombinedRmse"]],
                how="inner",
                rsuffix="_gru",
            )
            pairwise_win_rate = float((merged["testCombinedRmse"] < merged["testCombinedRmse_gru"]).mean()) if len(merged) else 0.0
            mean_delta = float(summary[architecture]["testCombinedRmseMean"] - gru_stats["testCombinedRmseMean"])
            std_delta = float(summary[architecture]["testCombinedRmseStd"] - gru_stats["testCombinedRmseStd"])
            stable_better = (
                summary[architecture]["testCombinedRmseMean"] < gru_stats["testCombinedRmseMean"]
                and summary[architecture]["testCombinedRmseStd"] <= gru_stats["testCombinedRmseStd"]
                and pairwise_win_rate >= 0.60
            )
            summary[architecture]["vsGru"] = {
                "testCombinedRmseMeanDelta": mean_delta,
                "testCombinedRmseStdDelta": std_delta,
                "pairwiseWinRate": pairwise_win_rate,
                "stableBetterThanGru": bool(stable_better),
            }
    return summary


def write_markdown_report(payload: dict[str, Any], output_path: Path) -> None:
    summary = payload["summaryByArchitecture"]
    ordered_arch = sorted(summary.keys(), key=lambda key: (0 if key == "gru" else 1, key))
    lines = [
        "# Grouped CV + Multi-Seed Architecture Benchmark",
        "",
        f"- Seeds: `{payload['seeds']}`",
        f"- Folds: `{payload['nFolds']}`",
        f"- Validation ratio inside training fold: `{payload['valRatio']}`",
        f"- Huber delta: `{payload['huberDelta']}`",
        f"- Epoch scale: `{payload['epochScale']}`",
        "",
        "| Architecture | Runs | Val RMSE mean | Val RMSE std | Test RMSE mean | Test RMSE std | vs GRU mean Δ | vs GRU std Δ | Win rate vs GRU | Stable better than GRU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for architecture in ordered_arch:
        item = summary[architecture]
        vs = item.get("vsGru", {})
        lines.append(
            "| "
            f"{architecture} | {item['nRuns']} | "
            f"{item['validationCombinedRmseMean']:.6f} | {item['validationCombinedRmseStd']:.6f} | "
            f"{item['testCombinedRmseMean']:.6f} | {item['testCombinedRmseStd']:.6f} | "
            f"{float(vs.get('testCombinedRmseMeanDelta', 0.0)):+.6f} | "
            f"{float(vs.get('testCombinedRmseStdDelta', 0.0)):+.6f} | "
            f"{float(vs.get('pairwiseWinRate', 0.0)):.3f} | "
            f"{'yes' if bool(vs.get('stableBetterThanGru', False)) else 'no'} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped CV + multi-seed architecture benchmark.")
    parser.add_argument("--architectures", default="gru,rnn_attention")
    parser.add_argument("--seeds", default="20260305,20260315,20260325")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--epoch-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    architectures = parse_str_list(args.architectures)
    seeds = parse_int_list(args.seeds)
    if not architectures:
        raise ValueError("architectures cannot be empty")
    if not seeds:
        raise ValueError("seeds cannot be empty")
    if args.folds < 2:
        raise ValueError("folds must be >= 2")

    runtime = configure_runtime()
    raw_df = load_real_dataset(force_download=False)
    configs = select_architecture_configs(
        architectures=architectures,
        huber_delta=float(args.huber_delta),
        epoch_scale=float(args.epoch_scale),
    )

    splitter = GroupKFold(n_splits=args.folds)
    run_rows: list[dict[str, Any]] = []

    for seed in seeds:
        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(raw_df, groups=raw_df["serial"].astype(str)),
            start=1,
        ):
            fold_train = raw_df.iloc[train_idx].copy()
            fold_test = raw_df.iloc[test_idx].copy()
            fit_df, val_df = split_train_val_by_group(
                fold_train,
                seed=seed + (fold_index * 997),
                val_ratio=float(args.val_ratio),
            )

            for architecture in architectures:
                base_config = configs[architecture]
                run_config = replace(
                    base_config,
                    name=f"{base_config.name}_seed_{seed}_fold_{fold_index}",
                )
                train_dataset, val_dataset, test_dataset = build_step_datasets_for_config(
                    fit_df,
                    val_df,
                    fold_test,
                    run_config,
                )

                result = train_candidate(
                    run_config,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    runtime,
                    seed_offset=candidate_seed_offset(run_config),
                    seed_base=seed,
                )
                val_rmse = float(result["validation"]["combinedRmse"])
                test_rmse = float(result["test"]["combinedRmse"])
                run_rows.append(
                    {
                        "architecture": architecture,
                        "seed": int(seed),
                        "fold": int(fold_index),
                        "validationCombinedRmse": val_rmse,
                        "testCombinedRmse": test_rmse,
                        "config": asdict(run_config),
                    }
                )
                print(
                    f"[seed={seed} fold={fold_index} arch={architecture}] "
                    f"val={val_rmse:.4f} test={test_rmse:.4f}"
                )

    summary = aggregate_metrics(run_rows)
    payload = {
        "dataset": "peptimizer-fastflow",
        "splitStrategy": "GroupKFold(serial) + train-inner-grouped-validation",
        "nFolds": int(args.folds),
        "valRatio": float(args.val_ratio),
        "architectures": architectures,
        "seeds": seeds,
        "huberDelta": float(args.huber_delta),
        "epochScale": float(args.epoch_scale),
        "runtime": asdict(runtime),
        "runs": run_rows,
        "summaryByArchitecture": summary,
    }

    root = project_root()
    json_path = root / "artifacts" / "grouped-cv-architecture-benchmark.json"
    md_path = root / "docs" / "grouped-cv-architecture-benchmark.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(payload, md_path)
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
