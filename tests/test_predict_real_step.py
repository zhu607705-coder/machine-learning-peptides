from __future__ import annotations

import numpy as np

from python.predict_real_step import build_numeric_values, restore_raw_prediction
from python.train_real_model import ModelConfig


def test_build_numeric_values_appends_prev_targets_for_delta_model() -> None:
    config = ModelConfig(
        name="conditional",
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
    values = build_numeric_values(
        config,
        pre_chain="ACD",
        coupling_strokes=7.0,
        deprotection_strokes=13.0,
        flow_rate=80000.0,
        temp_coupling=25.0,
        temp_reactor_1=90.0,
        prev_targets=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    assert values.tolist() == [7.0, 13.0, 80000.0, 25.0, 90.0, 3.0, 1.0, 2.0, 3.0, 4.0]


def test_restore_raw_prediction_adds_prev_targets_for_delta_model() -> None:
    config = ModelConfig(
        name="conditional",
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
    normalized = np.array([0.5, -0.5, 0.0, 1.0], dtype=np.float32)
    target_mean = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    target_std = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    prev_targets = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    restored = restore_raw_prediction(config, normalized, target_mean, target_std, prev_targets)
    assert restored.tolist() == [12.0, 20.0, 33.0, 52.0]
