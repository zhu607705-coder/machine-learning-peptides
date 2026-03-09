from __future__ import annotations

import pandas as pd

from python.train_data_first_gru_multiseed import (
    build_deploy_config,
    compute_stability_score,
    grouped_split_with_seed,
    parse_seed_list,
    select_stable_run,
)


def test_parse_seed_list() -> None:
    assert parse_seed_list("1, 2,3") == [1, 2, 3]


def test_grouped_split_with_seed_preserves_group_disjointness() -> None:
    df = pd.DataFrame(
        {
            "serial": [f"s{idx//2}" for idx in range(20)],
            "v": list(range(20)),
        }
    )
    train_df, val_df, test_df = grouped_split_with_seed(df, split_seed=42)
    train_serials = set(train_df["serial"].astype(str))
    val_serials = set(val_df["serial"].astype(str))
    test_serials = set(test_df["serial"].astype(str))
    assert train_serials.isdisjoint(val_serials)
    assert train_serials.isdisjoint(test_serials)
    assert val_serials.isdisjoint(test_serials)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_compute_stability_score_prefers_small_gap_and_center_distance() -> None:
    run = {"validation": {"combinedRmse": 0.16}, "test": {"combinedRmse": 0.17}}
    score = compute_stability_score(run, center_test_rmse=0.171)
    assert 0.0 <= score < 0.02


def test_select_stable_run() -> None:
    runs = [
        {"seed": 1, "validation": {"combinedRmse": 0.160}, "test": {"combinedRmse": 0.180}},
        {"seed": 2, "validation": {"combinedRmse": 0.169}, "test": {"combinedRmse": 0.171}},
        {"seed": 3, "validation": {"combinedRmse": 0.170}, "test": {"combinedRmse": 0.170}},
    ]
    selected = select_stable_run(runs)
    assert selected["seed"] in {2, 3}
    assert "stabilityScore" in selected


def test_build_deploy_config_uses_conditional_attention_defaults() -> None:
    config = build_deploy_config()
    assert config.architecture == "conditional_gru_attention"
    assert config.include_prev_targets is True
    assert config.predict_delta is True
    assert config.sequence_hidden == 40
    assert abs(config.dropout - 0.18) < 1e-9
