from __future__ import annotations

from python.rnn_attention_grid_benchmark import (
    combo_name,
    parse_float_list,
    parse_int_list,
    stable_better_than_gru,
    summarize_grid,
)
from python.train_real_model import ModelConfig


def test_parse_lists() -> None:
    assert parse_int_list("1, 2,3") == [1, 2, 3]
    assert parse_float_list("0.1, 0.2") == [0.1, 0.2]


def test_combo_name_contains_hidden_dropout_delta() -> None:
    config = ModelConfig(
        name="x",
        architecture="rnn_attention",
        max_length=40,
        embed_dim=24,
        sequence_hidden=48,
        numeric_hidden=48,
        trunk_hidden=128,
        dropout=0.15,
        learning_rate=8e-4,
        weight_decay=2e-4,
        batch_size=256,
        max_epochs=30,
        patience=5,
        huber_delta=0.75,
    )
    assert combo_name(config) == "rnn_attn_h48_do0.15_delta0.75"


def test_summarize_grid_computes_win_rate_vs_baseline() -> None:
    runs = [
        {"combo": "a", "seed": 1, "fold": 1, "validationCombinedRmse": 0.2, "testCombinedRmse": 0.3},
        {"combo": "a", "seed": 1, "fold": 2, "validationCombinedRmse": 0.2, "testCombinedRmse": 0.2},
        {"combo": "b", "seed": 1, "fold": 1, "validationCombinedRmse": 0.3, "testCombinedRmse": 0.4},
        {"combo": "b", "seed": 1, "fold": 2, "validationCombinedRmse": 0.2, "testCombinedRmse": 0.1},
    ]
    baseline = {(1, 1): 0.35, (1, 2): 0.15}
    summary = summarize_grid(runs, baseline)
    assert abs(summary["a"]["pairwiseWinRateVsGru"] - 0.5) < 1e-9
    assert abs(summary["b"]["pairwiseWinRateVsGru"] - 0.5) < 1e-9


def test_stable_better_than_gru_rule() -> None:
    item = {
        "testCombinedRmseMean": 0.16,
        "testCombinedRmseStd": 0.003,
        "pairwiseWinRateVsGru": 0.67,
    }
    assert stable_better_than_gru(item, gru_mean=0.17, gru_std=0.004, target_win_rate=0.6)
    assert not stable_better_than_gru(item, gru_mean=0.15, gru_std=0.004, target_win_rate=0.6)
