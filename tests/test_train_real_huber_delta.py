from __future__ import annotations

import torch

import pandas as pd

from python.train_real_model import (
    ModelConfig,
    ResidualRegressor,
    StepDataset,
    TARGET_COLUMNS,
    build_candidate_configs,
    candidate_seed_offset,
    parse_huber_deltas,
)


def test_parse_huber_deltas_defaults_to_single_default() -> None:
    assert parse_huber_deltas("") == [1.0]


def test_parse_huber_deltas_accepts_comma_separated_values() -> None:
    assert parse_huber_deltas("0.5, 1.0, 1.5") == [0.5, 1.0, 1.5]


def test_build_candidate_configs_expands_each_architecture_by_delta() -> None:
    configs = build_candidate_configs([0.5, 1.0])

    names = [config.name for config in configs]
    deltas = [config.huber_delta for config in configs]

    assert len(configs) == 10
    assert "gru_residual_small_delta_0_5" in names
    assert "gru_residual_small_delta_1_0" in names
    assert "rnn_attention_residual_medium_delta_0_5" in names
    assert "rnn_attention_residual_medium_delta_1_0" in names
    assert "conditional_gru_attention_delta_heads_delta_0_5" in names
    assert "conditional_gru_attention_delta_heads_delta_1_0" in names
    assert deltas.count(0.5) == 5
    assert deltas.count(1.0) == 5


def test_candidate_seed_offset_is_stable_within_architecture() -> None:
    configs = build_candidate_configs([0.5, 1.0])
    gru_offsets = {candidate_seed_offset(config) for config in configs if config.architecture == "gru"}
    cnn_offsets = {candidate_seed_offset(config) for config in configs if config.architecture == "cnn"}
    rnn_attention_offsets = {
        candidate_seed_offset(config) for config in configs if config.architecture == "rnn_attention"
    }
    conditional_offsets = {
        candidate_seed_offset(config) for config in configs if config.architecture == "conditional_gru_attention"
    }

    assert gru_offsets == {0}
    assert cnn_offsets == {1}
    assert rnn_attention_offsets == {3}
    assert conditional_offsets == {4}


def test_rnn_attention_sequence_encoder_output_shape() -> None:
    from python.train_real_model import SequenceEncoder, TOKEN_ALPHABET

    encoder = SequenceEncoder(
        architecture="rnn_attention",
        vocab_size=len(TOKEN_ALPHABET),
        embed_dim=16,
        hidden_dim=12,
        dropout=0.1,
    )
    tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    output = encoder(tokens)
    assert output.shape == (2, encoder.output_dim)


def test_delta_dataset_exposes_prev_targets_and_absolute_targets() -> None:
    df = pd.DataFrame(
        {
            "pre-chain": ["AC", "DEF"],
            "amino_acid": ["G", "H"],
            "coupling_agent": ["HATU", "DIC"],
            "serial": ["s1", "s2"],
            "coupling_strokes": [1, 2],
            "deprotection_strokes": [2, 3],
            "flow_rate": [4.0, 5.0],
            "temp_coupling": [70.0, 75.0],
            "temp_reactor_1": [80.0, 85.0],
            "sequence_length": [2, 3],
            "prev_area": [10.0, 20.0],
            "prev_height": [11.0, 21.0],
            "prev_width": [12.0, 22.0],
            "prev_diff": [13.0, 23.0],
            "first_area": [15.0, 26.0],
            "first_height": [16.0, 27.0],
            "first_width": [17.0, 28.0],
            "first_diff": [18.0, 29.0],
        }
    )
    dataset = StepDataset(df, max_length=8, include_prev_targets=True, predict_delta=True, fit=True)
    sample = dataset[0]
    assert sample["prev_targets"].shape == (len(TARGET_COLUMNS),)
    assert sample["raw_targets"].tolist() == [15.0, 16.0, 17.0, 18.0]
    assert dataset.numeric.shape[1] == 10


def test_conditional_gru_attention_forward_shape() -> None:
    df = pd.DataFrame(
        {
            "pre-chain": ["AC", "DEF"],
            "amino_acid": ["G", "H"],
            "coupling_agent": ["HATU", "DIC"],
            "serial": ["s1", "s2"],
            "coupling_strokes": [1, 2],
            "deprotection_strokes": [2, 3],
            "flow_rate": [4.0, 5.0],
            "temp_coupling": [70.0, 75.0],
            "temp_reactor_1": [80.0, 85.0],
            "sequence_length": [2, 3],
            "prev_area": [10.0, 20.0],
            "prev_height": [11.0, 21.0],
            "prev_width": [12.0, 22.0],
            "prev_diff": [13.0, 23.0],
            "first_area": [15.0, 26.0],
            "first_height": [16.0, 27.0],
            "first_width": [17.0, 28.0],
            "first_diff": [18.0, 29.0],
        }
    )
    dataset = StepDataset(df, max_length=8, include_prev_targets=True, predict_delta=True, fit=True)
    batch = {
        "sequence_tokens": dataset.sequence_tokens,
        "next_tokens": dataset.next_tokens,
        "coupling_tokens": dataset.coupling_tokens,
        "numeric": dataset.numeric,
    }
    config = ModelConfig(
        name="conditional",
        architecture="conditional_gru_attention",
        max_length=8,
        embed_dim=16,
        sequence_hidden=12,
        numeric_hidden=16,
        trunk_hidden=24,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        max_epochs=5,
        patience=2,
        include_prev_targets=True,
        predict_delta=True,
    )
    model = ResidualRegressor(config, numeric_dim=dataset.numeric.shape[1], output_dim=len(TARGET_COLUMNS))
    output = model(batch)
    assert output.shape == (2, len(TARGET_COLUMNS))
