from __future__ import annotations

import pandas as pd

from annotated_source_router import SourceRoutedRegressor


def test_source_router_uses_constant_source_mean() -> None:
    frame = pd.DataFrame(
        {
            "data_source": ["synthetic", "synthetic", "literature", "literature"],
            "length": [10, 12, 9, 11],
            "avg_hydrophobicity": [1.0, 1.1, 0.2, 0.3],
            "total_charge": [0.0, 0.0, 1.0, 2.0],
        }
    )
    target = pd.Series([50.5, 50.5, 10.0, 20.0])

    model = SourceRoutedRegressor(feature_columns=["length", "avg_hydrophobicity", "total_charge"])
    model.fit(frame, target)
    predictions = model.predict(frame.iloc[[0, 1]])

    assert list(predictions) == [50.5, 50.5]


def test_source_router_falls_back_to_global_for_unseen_source() -> None:
    train = pd.DataFrame(
        {
            "data_source": ["literature", "literature", "literature"],
            "length": [8, 10, 12],
            "avg_hydrophobicity": [0.1, 0.2, 0.3],
            "total_charge": [1.0, 0.0, -1.0],
        }
    )
    target = pd.Series([30.0, 40.0, 50.0])

    model = SourceRoutedRegressor(feature_columns=["length", "avg_hydrophobicity", "total_charge"])
    model.fit(train, target)

    test = pd.DataFrame(
        {
            "data_source": ["experimental"],
            "length": [9],
            "avg_hydrophobicity": [0.15],
            "total_charge": [0.5],
        }
    )
    prediction = model.predict(test)

    assert len(prediction) == 1
    assert 20.0 <= float(prediction[0]) <= 60.0
