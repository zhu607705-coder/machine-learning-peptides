from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class SourceModelState:
    source: str
    mean_value: float
    is_constant: bool
    model: object | None


class SourceRoutedRegressor:
    def __init__(
        self,
        feature_columns: list[str],
        source_column: str = "data_source",
        min_unique_values: int = 2,
        min_std: float = 1e-8,
        model_factory: Callable[[], object] | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.source_column = source_column
        self.min_unique_values = min_unique_values
        self.min_std = min_std
        self.model_factory = model_factory or (
            lambda: GradientBoostingRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                random_state=20260307,
            )
        )
        self.global_model: object | None = None
        self.global_mean: float = 0.0
        self.source_states: dict[str, SourceModelState] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["model_factory"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.model_factory is None:
            self.model_factory = lambda: GradientBoostingRegressor(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                random_state=20260307,
            )

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "SourceRoutedRegressor":
        working = frame.copy()
        numeric_target = pd.to_numeric(target, errors="coerce")
        valid_mask = numeric_target.notna()
        working = working.loc[valid_mask].copy()
        working["_target"] = numeric_target.loc[valid_mask].astype(float)

        x_all = working[self.feature_columns].to_numpy(dtype=float)
        y_all = working["_target"].to_numpy(dtype=float)
        self.global_mean = float(y_all.mean())
        self.global_model = self.model_factory()
        self.global_model.fit(x_all, y_all)

        self.source_states = {}
        for source, subset in working.groupby(self.source_column):
            y_source = subset["_target"].to_numpy(dtype=float)
            unique_values = int(pd.Series(y_source).nunique())
            std_value = float(pd.Series(y_source).std(ddof=0) or 0.0)
            mean_value = float(y_source.mean())
            if unique_values < self.min_unique_values or std_value <= self.min_std:
                self.source_states[str(source)] = SourceModelState(
                    source=str(source),
                    mean_value=mean_value,
                    is_constant=True,
                    model=None,
                )
                continue

            model = self.model_factory()
            model.fit(subset[self.feature_columns].to_numpy(dtype=float), y_source)
            self.source_states[str(source)] = SourceModelState(
                source=str(source),
                mean_value=mean_value,
                is_constant=False,
                model=model,
            )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.global_model is None:
            raise RuntimeError("Model is not fitted.")

        predictions: list[float] = []
        for _, row in frame.iterrows():
            source = str(row.get(self.source_column, "unknown"))
            state = self.source_states.get(source)
            if state is None:
                pred = float(self.global_model.predict(np.asarray([row[self.feature_columns].to_numpy(dtype=float)]))[0])
            elif state.is_constant or state.model is None:
                pred = state.mean_value
            else:
                pred = float(state.model.predict(np.asarray([row[self.feature_columns].to_numpy(dtype=float)]))[0])
            predictions.append(pred)
        return np.asarray(predictions, dtype=float)
