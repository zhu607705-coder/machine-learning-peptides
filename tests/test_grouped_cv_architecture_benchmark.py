from __future__ import annotations

import pandas as pd

from python.architecture_grouped_cv_benchmark import (
    aggregate_metrics,
    parse_int_list,
    parse_str_list,
    split_train_val_by_group,
)


def test_parse_helpers() -> None:
    assert parse_int_list("1, 2,3") == [1, 2, 3]
    assert parse_str_list("gru, rnn_attention") == ["gru", "rnn_attention"]


def test_split_train_val_by_group_has_no_group_overlap() -> None:
    df = pd.DataFrame(
        {
            "serial": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "x": list(range(8)),
        }
    )
    train_df, val_df = split_train_val_by_group(df, seed=42, val_ratio=0.25)
    train_serials = set(train_df["serial"].astype(str).unique().tolist())
    val_serials = set(val_df["serial"].astype(str).unique().tolist())
    assert train_serials
    assert val_serials
    assert train_serials.isdisjoint(val_serials)
    assert len(train_df) + len(val_df) == len(df)


def test_aggregate_metrics_computes_vs_gru_summary() -> None:
    rows = [
        {"architecture": "gru", "seed": 1, "fold": 1, "validationCombinedRmse": 0.20, "testCombinedRmse": 0.22},
        {"architecture": "gru", "seed": 1, "fold": 2, "validationCombinedRmse": 0.21, "testCombinedRmse": 0.23},
        {"architecture": "rnn_attention", "seed": 1, "fold": 1, "validationCombinedRmse": 0.19, "testCombinedRmse": 0.21},
        {"architecture": "rnn_attention", "seed": 1, "fold": 2, "validationCombinedRmse": 0.20, "testCombinedRmse": 0.24},
    ]
    summary = aggregate_metrics(rows)
    assert "gru" in summary
    assert "rnn_attention" in summary
    assert summary["gru"]["nRuns"] == 2
    assert "vsGru" in summary["rnn_attention"]
    assert 0.0 <= summary["rnn_attention"]["vsGru"]["pairwiseWinRate"] <= 1.0
