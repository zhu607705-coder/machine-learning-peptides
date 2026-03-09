from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold

from architecture_grouped_cv_benchmark import split_train_val_by_group
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
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_gru_baseline(path: Path) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    baseline: dict[tuple[int, int], float] = {}
    for row in payload.get("runs", []):
        if row.get("architecture") != "gru":
            continue
        key = (int(row["seed"]), int(row["fold"]))
        baseline[key] = float(row["testCombinedRmse"])
    return baseline, payload


def build_rnn_base_config(delta: float) -> ModelConfig:
    for config in build_candidate_configs([delta]):
        if config.architecture == "rnn_attention":
            return config
    raise ValueError("rnn_attention base config not found")


def combo_name(config: ModelConfig) -> str:
    return (
        f"rnn_attn_h{config.sequence_hidden}_do{config.dropout:.2f}_"
        f"delta{config.huber_delta:.2f}"
    )


def summarize_grid(
    runs: list[dict[str, Any]],
    baseline_by_fold_seed: dict[tuple[int, int], float],
) -> dict[str, Any]:
    frame = pd.DataFrame(runs)
    results: dict[str, Any] = {}
    for key, group in frame.groupby("combo"):
        test_values = group["testCombinedRmse"].astype(float)
        val_values = group["validationCombinedRmse"].astype(float)
        pairwise_wins = []
        for _, row in group.iterrows():
            baseline = baseline_by_fold_seed.get((int(row["seed"]), int(row["fold"])))
            if baseline is None:
                continue
            pairwise_wins.append(float(row["testCombinedRmse"]) < baseline)
        win_rate = float(sum(pairwise_wins) / len(pairwise_wins)) if pairwise_wins else 0.0
        results[str(key)] = {
            "nRuns": int(len(group)),
            "validationCombinedRmseMean": float(val_values.mean()),
            "validationCombinedRmseStd": float(val_values.std(ddof=1)) if len(group) > 1 else 0.0,
            "testCombinedRmseMean": float(test_values.mean()),
            "testCombinedRmseStd": float(test_values.std(ddof=1)) if len(group) > 1 else 0.0,
            "pairwiseWinRateVsGru": win_rate,
        }
    return results


def stable_better_than_gru(
    summary_item: dict[str, Any],
    gru_mean: float,
    gru_std: float,
    *,
    target_win_rate: float,
) -> bool:
    return (
        float(summary_item["testCombinedRmseMean"]) < float(gru_mean)
        and float(summary_item["testCombinedRmseStd"]) <= float(gru_std)
        and float(summary_item["pairwiseWinRateVsGru"]) >= float(target_win_rate)
    )


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# RNN Attention Grid Benchmark (3x3 Stability)",
        "",
        f"- Seeds: `{payload['seeds']}`",
        f"- Folds: `{payload['folds']}`",
        f"- Validation ratio: `{payload['valRatio']}`",
        f"- Hidden grid: `{payload['hiddenGrid']}`",
        f"- Dropout grid: `{payload['dropoutGrid']}`",
        f"- Delta grid: `{payload['deltaGrid']}`",
        f"- Target win-rate threshold: `{payload['targetWinRate']}`",
        "",
        f"- GRU baseline mean/std: `{payload['gruBaseline']['testCombinedRmseMean']:.6f}` / "
        f"`{payload['gruBaseline']['testCombinedRmseStd']:.6f}`",
        "",
        "| Combo | Runs | Val RMSE mean | Val RMSE std | Test RMSE mean | Test RMSE std | Pairwise win rate vs GRU | Stable better than GRU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for combo, item in sorted(payload["summaryByCombo"].items()):
        lines.append(
            f"| {combo} | {item['nRuns']} | "
            f"{item['validationCombinedRmseMean']:.6f} | {item['validationCombinedRmseStd']:.6f} | "
            f"{item['testCombinedRmseMean']:.6f} | {item['testCombinedRmseStd']:.6f} | "
            f"{item['pairwiseWinRateVsGru']:.3f} | "
            f"{'yes' if item['stableBetterThanGru'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"## Best by win rate: `{payload['bestByWinRate']}`")
    lines.append(f"## Best by test RMSE mean: `{payload['bestByTestRmseMean']}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid benchmark for rnn_attention against GRU baseline.")
    parser.add_argument("--hidden-grid", default="32,40,48")
    parser.add_argument("--dropout-grid", default="0.10,0.15,0.20")
    parser.add_argument("--delta-grid", default="0.75,1.00")
    parser.add_argument("--seeds", default="20260305,20260315,20260325")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epoch-scale", type=float, default=0.75)
    parser.add_argument("--target-win-rate", type=float, default=0.6)
    parser.add_argument(
        "--gru-baseline-json",
        default=str(project_root() / "artifacts" / "grouped-cv-architecture-benchmark.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_grid = parse_int_list(args.hidden_grid)
    dropout_grid = parse_float_list(args.dropout_grid)
    delta_grid = parse_float_list(args.delta_grid)
    seeds = parse_int_list(args.seeds)
    if args.folds < 2:
        raise ValueError("folds must be >= 2")

    baseline_path = Path(args.gru_baseline_json)
    baseline_by_fold_seed, baseline_payload = load_gru_baseline(baseline_path)
    if not baseline_by_fold_seed:
        raise ValueError(f"No GRU baseline runs found in {baseline_path}")

    baseline_summary = baseline_payload.get("summaryByArchitecture", {}).get("gru")
    if not baseline_summary:
        raise ValueError(f"No GRU summary found in {baseline_path}")
    gru_mean = float(baseline_summary["testCombinedRmseMean"])
    gru_std = float(baseline_summary["testCombinedRmseStd"])

    runtime = configure_runtime()
    raw_df = load_real_dataset(force_download=False)
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

            for hidden in hidden_grid:
                for dropout in dropout_grid:
                    for delta in delta_grid:
                        base = build_rnn_base_config(delta)
                        max_epochs = max(6, int(round(base.max_epochs * float(args.epoch_scale))))
                        patience = max(3, int(round(base.patience * float(args.epoch_scale))))
                        config = replace(
                            base,
                            name=f"{base.name}_h{hidden}_do{dropout:.2f}_seed{seed}_fold{fold_index}",
                            sequence_hidden=int(hidden),
                            dropout=float(dropout),
                            huber_delta=float(delta),
                            max_epochs=max_epochs,
                            patience=patience,
                        )
                        train_dataset, val_dataset, test_dataset = build_step_datasets_for_config(
                            fit_df,
                            val_df,
                            fold_test,
                            config,
                        )
                        result = train_candidate(
                            config,
                            train_dataset,
                            val_dataset,
                            test_dataset,
                            runtime,
                            seed_offset=candidate_seed_offset(config),
                            seed_base=seed,
                        )
                        val_rmse = float(result["validation"]["combinedRmse"])
                        test_rmse = float(result["test"]["combinedRmse"])
                        combo = combo_name(config)
                        run_rows.append(
                            {
                                "combo": combo,
                                "seed": int(seed),
                                "fold": int(fold_index),
                                "validationCombinedRmse": val_rmse,
                                "testCombinedRmse": test_rmse,
                                "config": asdict(config),
                            }
                        )
                        print(
                            f"[seed={seed} fold={fold_index} {combo}] "
                            f"val={val_rmse:.4f} test={test_rmse:.4f}"
                        )

    summary_by_combo = summarize_grid(run_rows, baseline_by_fold_seed)
    for combo, item in summary_by_combo.items():
        item["stableBetterThanGru"] = stable_better_than_gru(
            item,
            gru_mean=gru_mean,
            gru_std=gru_std,
            target_win_rate=float(args.target_win_rate),
        )

    best_by_win_rate = max(
        summary_by_combo.items(),
        key=lambda kv: (kv[1]["pairwiseWinRateVsGru"], -kv[1]["testCombinedRmseMean"]),
    )[0]
    best_by_test_rmse_mean = min(summary_by_combo.items(), key=lambda kv: kv[1]["testCombinedRmseMean"])[0]

    payload = {
        "dataset": "peptimizer-fastflow",
        "baselineSource": str(baseline_path),
        "gruBaseline": {
            "testCombinedRmseMean": gru_mean,
            "testCombinedRmseStd": gru_std,
        },
        "seeds": seeds,
        "folds": int(args.folds),
        "valRatio": float(args.val_ratio),
        "epochScale": float(args.epoch_scale),
        "hiddenGrid": hidden_grid,
        "dropoutGrid": dropout_grid,
        "deltaGrid": delta_grid,
        "targetWinRate": float(args.target_win_rate),
        "runtime": asdict(runtime),
        "runs": run_rows,
        "summaryByCombo": summary_by_combo,
        "bestByWinRate": best_by_win_rate,
        "bestByTestRmseMean": best_by_test_rmse_mean,
    }

    root = project_root()
    json_path = root / "artifacts" / "rnn-attention-grid-benchmark.json"
    md_path = root / "docs" / "rnn-attention-grid-benchmark.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, md_path)
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
