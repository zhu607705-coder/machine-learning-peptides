from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


FEATURE_COLUMNS = [
    "length",
    "avg_hydrophobicity",
    "total_charge",
    "molecular_weight",
    "avg_volume",
    "max_coupling_difficulty",
    "bulky_ratio",
    "polar_ratio",
    "charged_ratio",
    "aromatic_ratio",
    "sulfur_ratio",
    "reagent_score",
    "solvent_score",
    "temperature_score",
    "cleavage_score",
]


@dataclass(frozen=True)
class SourceVarianceStats:
    source: str
    count: int
    unique_values: int
    std: float


def build_group_labels(
    frame: pd.DataFrame,
    source_column: str = "data_source",
    length_column: str = "length",
    n_bins: int = 8,
) -> pd.Series:
    length_values = pd.to_numeric(frame[length_column], errors="coerce").fillna(0.0)
    effective_bins = max(1, min(int(n_bins), int(length_values.nunique())))
    if effective_bins == 1:
        bin_labels = pd.Series(["lenbin0"] * len(frame), index=frame.index)
    else:
        bin_ids = pd.qcut(length_values, q=effective_bins, labels=False, duplicates="drop")
        bin_labels = bin_ids.fillna(0).astype(int).map(lambda value: f"lenbin{value}")

    sources = frame[source_column].fillna("unknown").astype(str)
    return sources + "|" + bin_labels


def build_regression_strata(target: pd.Series, n_bins: int = 3) -> pd.Series:
    numeric = pd.to_numeric(target, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(["q0"] * len(target), index=target.index)

    effective_bins = max(1, min(int(n_bins), int(valid.nunique())))
    if effective_bins == 1:
        return pd.Series(["q0"] * len(target), index=target.index)

    raw_bins = pd.qcut(valid, q=effective_bins, labels=False, duplicates="drop")
    raw_bins = raw_bins.astype(int)
    strata = pd.Series(index=target.index, dtype=object)
    strata.loc[valid.index] = raw_bins.map(lambda value: f"q{value}")
    strata = strata.fillna("q0")
    return strata


def drop_low_variance_sources(
    frame: pd.DataFrame,
    target_column: str,
    source_column: str = "data_source",
    min_unique_values: int = 2,
    min_std: float = 1e-8,
) -> tuple[pd.DataFrame, dict[str, object]]:
    numeric_target = pd.to_numeric(frame[target_column], errors="coerce")
    working = frame.loc[numeric_target.notna()].copy()
    working[target_column] = numeric_target.loc[working.index]

    stats: list[SourceVarianceStats] = []
    kept_sources: list[str] = []
    dropped_sources: list[str] = []

    for source, subset in working.groupby(source_column):
        unique_values = int(subset[target_column].nunique())
        std = float(subset[target_column].std(ddof=0) or 0.0)
        stats.append(
            SourceVarianceStats(
                source=str(source),
                count=int(len(subset)),
                unique_values=unique_values,
                std=std,
            )
        )
        if unique_values >= min_unique_values and std > min_std:
            kept_sources.append(str(source))
        else:
            dropped_sources.append(str(source))

    filtered = working[working[source_column].astype(str).isin(kept_sources)].copy()
    return filtered, {
        "kept_sources": sorted(kept_sources),
        "dropped_sources": sorted(dropped_sources),
        "source_stats": [
            {
                "source": item.source,
                "count": item.count,
                "unique_values": item.unique_values,
                "std": item.std,
            }
            for item in stats
        ],
    }


def winsorize_target(
    target: pd.Series,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> tuple[pd.Series, dict[str, float]]:
    numeric = pd.to_numeric(target, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return numeric, {"lower_bound": 0.0, "upper_bound": 0.0, "n_clipped": 0}

    lower_bound = float(valid.quantile(lower_quantile))
    upper_bound = float(valid.quantile(upper_quantile))
    clipped = numeric.clip(lower=lower_bound, upper=upper_bound)
    n_clipped = int(((numeric < lower_bound) | (numeric > upper_bound)).fillna(False).sum())
    return clipped, {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_clipped": n_clipped,
    }
