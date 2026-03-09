from __future__ import annotations

import json
from pathlib import Path

import torch

from train_real_model import (
    ModelConfig,
    build_step_datasets_for_config,
    candidate_seed_offset,
    configure_runtime,
    grouped_split,
    load_real_dataset,
    train_candidate,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    runtime = configure_runtime()
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

    raw_df = load_real_dataset(force_download=False)
    train_df, val_df, test_df = grouped_split(raw_df)
    print(
        "Dataset:",
        json.dumps(
            {
                "rows": int(len(raw_df)),
                "uniqueSerials": int(raw_df["serial"].nunique()),
                "splitRows": {
                    "train": int(len(train_df)),
                    "validation": int(len(val_df)),
                    "test": int(len(test_df)),
                },
            },
            indent=2,
        ),
    )

    config = ModelConfig(
        name="gru_data_first_fixed",
        architecture="gru",
        max_length=40,
        embed_dim=16,
        sequence_hidden=32,
        numeric_hidden=32,
        trunk_hidden=96,
        dropout=0.12,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=36,
        patience=6,
        huber_delta=0.75,
    )

    train_dataset, val_dataset, test_dataset = build_step_datasets_for_config(train_df, val_df, test_df, config)

    result = train_candidate(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        runtime=runtime,
        seed_offset=candidate_seed_offset(config),
    )

    artifacts_dir = project_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "large-data-gru-report.json"
    model_path = artifacts_dir / "large-data-gru-model.pt"
    report = {
        "mode": "data_first_training",
        "config": result["config"],
        "loss": result["loss"],
        "bestEpoch": result["bestEpoch"],
        "validation": result["validation"],
        "test": result["test"],
        "datasetRows": int(len(raw_df)),
        "datasetUniqueSerials": int(raw_df["serial"].nunique()),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    torch.save(
        {
            "config": result["config"],
            "state_dict": result["stateDict"],
            "normalization": {
                "numericMean": train_dataset.numeric_mean.tolist(),
                "numericStd": train_dataset.numeric_std.tolist(),
                "targetMean": train_dataset.target_mean.tolist(),
                "targetStd": train_dataset.target_std.tolist(),
            },
        },
        model_path,
    )
    print(f"Wrote report: {report_path}")
    print(f"Wrote model: {model_path}")
    print("Validation combined RMSE:", f"{result['validation']['combinedRmse']:.6f}")
    print("Test combined RMSE:", f"{result['test']['combinedRmse']:.6f}")


if __name__ == "__main__":
    main()
