from __future__ import annotations

import pandas as pd

from annotated_training_utils import (
    build_group_labels,
    build_regression_strata,
    drop_low_variance_sources,
    winsorize_target,
)


def test_drop_low_variance_sources_removes_constant_source() -> None:
    frame = pd.DataFrame(
        {
            "data_source": ["synthetic", "synthetic", "literature", "literature", "literature"],
            "yield_val": [50.5, 50.5, 10.0, 40.0, 80.0],
            "length": [10, 11, 12, 13, 14],
        }
    )

    filtered, stats = drop_low_variance_sources(frame, target_column="yield_val")

    assert set(filtered["data_source"]) == {"literature"}
    assert stats["dropped_sources"] == ["synthetic"]
    assert stats["kept_sources"] == ["literature"]


def test_build_group_labels_uses_source_and_length_bins() -> None:
    frame = pd.DataFrame(
        {
            "data_source": ["a", "a", "a", "b", "b", "b"],
            "length": [5, 6, 20, 7, 8, 30],
        }
    )

    labels = build_group_labels(frame, n_bins=3)

    assert len(labels) == len(frame)
    assert labels.iloc[0].startswith("a|")
    assert labels.iloc[-1].startswith("b|")
    assert labels.nunique() >= 3


def test_build_regression_strata_returns_quantile_bins() -> None:
    target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    strata = build_regression_strata(target, n_bins=3)

    assert len(strata) == len(target)
    assert set(strata.unique()) == {"q0", "q1", "q2"}


def test_build_regression_strata_falls_back_for_constant_values() -> None:
    target = pd.Series([50.5, 50.5, 50.5, 50.5])

    strata = build_regression_strata(target, n_bins=3)

    assert set(strata.unique()) == {"q0"}


def test_winsorize_target_clips_extremes() -> None:
    target = pd.Series([1.0, 2.0, 3.0, 100.0])

    clipped, stats = winsorize_target(target, lower_quantile=0.25, upper_quantile=0.75)

    assert float(clipped.min()) == 1.75
    assert float(clipped.max()) == 27.25
    assert stats["n_clipped"] == 2
