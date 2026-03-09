from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from train_real_model import (
    ModelConfig,
    build_step_datasets_for_config,
    candidate_seed_offset,
    configure_runtime,
    load_real_dataset,
    state_dict_to_serializable,
    train_candidate,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_seed_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("seeds cannot be empty")
    return [int(value) for value in values]


def grouped_split_with_seed(df: pd.DataFrame, split_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(split_seed)
    serials = df["serial"].drop_duplicates().tolist()
    rng.shuffle(serials)

    total = len(serials)
    train_serials = set(serials[: int(total * 0.7)])
    val_serials = set(serials[int(total * 0.7): int(total * 0.85)])
    test_serials = set(serials[int(total * 0.85):])

    train_df = df[df["serial"].isin(train_serials)].copy()
    val_df = df[df["serial"].isin(val_serials)].copy()
    test_df = df[df["serial"].isin(test_serials)].copy()
    return train_df, val_df, test_df


def build_deploy_config() -> ModelConfig:
    return ModelConfig(
        name="conditional_gru_attention_deploy",
        architecture="conditional_gru_attention",
        max_length=40,
        embed_dim=24,
        sequence_hidden=40,
        numeric_hidden=48,
        trunk_hidden=128,
        dropout=0.18,
        learning_rate=8e-4,
        weight_decay=2e-4,
        batch_size=256,
        max_epochs=34,
        patience=6,
        huber_delta=0.75,
        include_prev_targets=True,
        predict_delta=True,
    )


def compute_stability_score(run: dict[str, Any], center_test_rmse: float) -> float:
    val_rmse = float(run["validation"]["combinedRmse"])
    test_rmse = float(run["test"]["combinedRmse"])
    gap = abs(val_rmse - test_rmse)
    center_distance = abs(test_rmse - center_test_rmse)
    return float(gap + (0.5 * center_distance))


def select_stable_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("runs cannot be empty")
    sorted_test = sorted(float(item["test"]["combinedRmse"]) for item in runs)
    center_test = sorted_test[len(sorted_test) // 2]
    scored = []
    for run in runs:
        score = compute_stability_score(run, center_test_rmse=center_test)
        scored.append((score, float(run["test"]["combinedRmse"]), run))
    scored.sort(key=lambda item: (item[0], item[1]))
    best = dict(scored[0][2])
    best["stabilityScore"] = float(scored[0][0])
    best["centerTestRmse"] = float(center_test)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed deployment training with automatic stable-model selection."
    )
    parser.add_argument("--seeds", default="20260305,20260315,20260325,20260335,20260345")
    parser.add_argument("--split-seed", type=int, default=20260302)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-dir", default=str(project_root() / "models" / "deploy"))
    parser.add_argument("--report-dir", default=str(project_root() / "artifacts" / "reports" / "deploy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    if args.output_dir:
        model_dir = Path(args.output_dir)
        report_dir = Path(args.output_dir)
    else:
        model_dir = Path(args.model_dir)
        report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    runtime = configure_runtime()
    raw_df = load_real_dataset(force_download=False)
    train_df, val_df, test_df = grouped_split_with_seed(raw_df, split_seed=args.split_seed)
    config = build_deploy_config()

    print(
        "Runtime:",
        json.dumps(
            {
                "device": runtime.device_name,
                "selectionReason": runtime.selection_reason,
                "cpuThreads": runtime.cpu_threads,
                "numWorkers": runtime.num_workers,
            },
            indent=2,
        ),
    )
    print(
        "Dataset:",
        json.dumps(
            {
                "rows": int(len(raw_df)),
                "uniqueSerials": int(raw_df["serial"].nunique()),
                "splitSeed": int(args.split_seed),
                "splitRows": {
                    "train": int(len(train_df)),
                    "validation": int(len(val_df)),
                    "test": int(len(test_df)),
                },
            },
            indent=2,
        ),
    )
    print("Seeds:", seeds)

    run_results: list[dict[str, Any]] = []
    for seed in seeds:
        train_dataset, val_dataset, test_dataset = build_step_datasets_for_config(train_df, val_df, test_df, config)

        result = train_candidate(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            runtime=runtime,
            seed_offset=candidate_seed_offset(config),
            seed_base=seed,
        )
        result_payload = {
            "seed": int(seed),
            "config": result["config"],
            "loss": result["loss"],
            "bestEpoch": int(result["bestEpoch"]),
            "validation": result["validation"],
            "test": result["test"],
            "stateDict": result["stateDict"],
            "normalization": {
                "numericMean": train_dataset.numeric_mean.tolist(),
                "numericStd": train_dataset.numeric_std.tolist(),
                "targetMean": train_dataset.target_mean.tolist(),
                "targetStd": train_dataset.target_std.tolist(),
            },
        }
        run_results.append(result_payload)
        print(
            f"[seed={seed}] val={float(result_payload['validation']['combinedRmse']):.6f} "
            f"test={float(result_payload['test']['combinedRmse']):.6f}"
        )

    selected = select_stable_run(run_results)
    selected_seed = int(selected["seed"])
    selected_state_dict = selected["stateDict"]
    selected_normalization = selected["normalization"]

    tests = [float(item["test"]["combinedRmse"]) for item in run_results]
    vals = [float(item["validation"]["combinedRmse"]) for item in run_results]
    summary = {
        "nRuns": int(len(run_results)),
        "seedList": seeds,
        "testCombinedRmseMean": float(sum(tests) / len(tests)),
        "testCombinedRmseStd": float(pd.Series(tests).std(ddof=1)) if len(tests) > 1 else 0.0,
        "validationCombinedRmseMean": float(sum(vals) / len(vals)),
        "validationCombinedRmseStd": float(pd.Series(vals).std(ddof=1)) if len(vals) > 1 else 0.0,
    }

    deploy_pt_path = model_dir / "final-deploy-model.pt"
    deploy_weights_json_path = model_dir / "final-deploy-weights.json"
    deploy_report_path = report_dir / "final-deploy-report.json"

    torch.save(
        {
            "config": selected["config"],
            "state_dict": selected_state_dict,
            "normalization": selected_normalization,
            "selection": {
                "selectedSeed": selected_seed,
                "stabilityScore": float(selected["stabilityScore"]),
                "centerTestRmse": float(selected["centerTestRmse"]),
            },
            "summary": summary,
        },
        deploy_pt_path,
    )

    deploy_weights_json_path.write_text(
        json.dumps(
            {
                "selectedSeed": selected_seed,
                "config": selected["config"],
                "normalization": selected_normalization,
                "stateDict": state_dict_to_serializable(selected_state_dict),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lightweight_runs = []
    for item in run_results:
        lightweight_runs.append(
            {
                "seed": int(item["seed"]),
                "bestEpoch": int(item["bestEpoch"]),
                "validation": item["validation"],
                "test": item["test"],
            }
        )
    deploy_report_path.write_text(
        json.dumps(
            {
                "mode": "multi_seed_data_first_conditional_gru_attention",
                "datasetRows": int(len(raw_df)),
                "datasetUniqueSerials": int(raw_df["serial"].nunique()),
                "splitSeed": int(args.split_seed),
                "summary": summary,
                "selection": {
                    "selectedSeed": selected_seed,
                    "stabilityScore": float(selected["stabilityScore"]),
                    "centerTestRmse": float(selected["centerTestRmse"]),
                    "selectedValidationCombinedRmse": float(selected["validation"]["combinedRmse"]),
                    "selectedTestCombinedRmse": float(selected["test"]["combinedRmse"]),
                },
                "runs": lightweight_runs,
                "paths": {
                    "deployModelPt": str(deploy_pt_path),
                    "deployWeightsJson": str(deploy_weights_json_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    compatibility_paths = [
        model_dir / "final-gru-deploy-model.pt",
        model_dir / "final-gru-deploy-weights.json",
        report_dir / "final-gru-deploy-report.json",
    ]
    for source_path, compatibility_path in zip(
        [deploy_pt_path, deploy_weights_json_path, deploy_report_path],
        compatibility_paths,
        strict=True,
    ):
        shutil.copy2(source_path, compatibility_path)

    print(f"Selected seed: {selected_seed}")
    print(f"Selected stability score: {float(selected['stabilityScore']):.6f}")
    print(f"Selected val/test RMSE: {float(selected['validation']['combinedRmse']):.6f} / {float(selected['test']['combinedRmse']):.6f}")
    print(f"Wrote deploy model: {deploy_pt_path}")
    print(f"Wrote deploy weights: {deploy_weights_json_path}")
    print(f"Wrote deploy report: {deploy_report_path}")


if __name__ == "__main__":
    main()
