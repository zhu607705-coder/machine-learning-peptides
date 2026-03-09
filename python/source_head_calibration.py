from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SourceAwareShrinkageSummary:
    prior_strength: float
    global_mean: float
    residual_std: float
    source_offsets: dict[str, float]
    source_counts: dict[str, int]


class SourceAwareShrinkageCalibrator:
    def __init__(self, prior_strength: float = 5.0) -> None:
        self.prior_strength = float(prior_strength)

    def fit(self, train_sources: pd.Series | list[str], residuals: np.ndarray | list[float]) -> "SourceAwareShrinkageCalibrator":
        source_series = pd.Series(train_sources).astype(str)
        residual_array = np.asarray(residuals, dtype=float)
        if len(source_series) != len(residual_array):
            raise ValueError("train_sources and residuals must have the same length")
        if len(source_series) == 0:
            raise ValueError("training data cannot be empty")

        self.global_mean_ = float(np.mean(residual_array))
        self.residual_std_ = float(np.std(residual_array, ddof=1)) if len(residual_array) > 1 else 0.0

        frame = pd.DataFrame({"source_id": source_series, "residual": residual_array})
        grouped = frame.groupby("source_id", as_index=False).agg(
            count=("residual", "size"),
            source_mean=("residual", "mean"),
        )
        self.source_offsets_ = {}
        self.source_counts_ = {}
        for _, row in grouped.iterrows():
            source_id = str(row["source_id"])
            count = int(row["count"])
            source_mean = float(row["source_mean"])
            shrink = count / (count + self.prior_strength)
            self.source_offsets_[source_id] = shrink * (source_mean - self.global_mean_)
            self.source_counts_[source_id] = count
        return self

    def predict_offsets(self, test_sources: pd.Series | list[str]) -> np.ndarray:
        source_series = pd.Series(test_sources).astype(str)
        return np.asarray([self.source_offsets_.get(source_id, 0.0) for source_id in source_series], dtype=float)

    def summary(self) -> dict[str, Any]:
        return {
            "priorStrength": float(self.prior_strength),
            "globalMean": float(getattr(self, "global_mean_", 0.0)),
            "residualStd": float(getattr(self, "residual_std_", 0.0)),
            "sourceOffsets": {str(key): float(value) for key, value in getattr(self, "source_offsets_", {}).items()},
            "sourceCounts": {str(key): int(value) for key, value in getattr(self, "source_counts_", {}).items()},
        }
