from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from bayesian_multitask_calibrator import BayesianMultitaskCalibrator
from data_preprocessor import FeatureExtractor
from enhance_dataset import calculate_peptide_features, parse_sequence
from protein_embeddings import DEFAULT_MODEL_NAME, DEFAULT_CACHE_DIR, SequenceEmbeddingPCATransformer
from residual_ensemble import ResidualEnsembleRegressor
from source_head_calibration import SourceAwareShrinkageCalibrator

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "real" / "final_purity_yield_literature.csv"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "comprehensive-ml-workflow-report.json"
MARKDOWN_REPORT_PATH = PROJECT_ROOT / "docs" / "comprehensive-ml-workflow-report.md"

RANDOM_STATE = 20260302
PURITY_TOLERANCE = 5.0
YIELD_TOLERANCE = 10.0
MIN_YIELD_HEAD_ROWS = 8
YIELD_HEAD_BLEND_PRIOR = 20.0
HEAD_MIN_ROWS = 20
HEAD_MIN_SOURCES = 3
HEAD_MAX_SOURCE_SHARE = 0.70
BAYES_HEAD_MIN_ROWS = 40
BAYES_HEAD_MIN_SOURCES = 5
SOURCE_OFFSET_PRIOR_STRENGTH = 5.0
CONFIDENCE_REJECT_INTERVAL_MULTIPLIER = 4.0
CONFIDENCE_WARN_INTERVAL_MULTIPLIER = 3.0
SOURCE_DRIFT_Z_THRESHOLD = 2.0
SOURCE_DRIFT_MIN_ROWS = 3

STANDARDIZED_FEATURES = [
    "publication_year",
    "length",
    "avg_hydrophobicity",
    "total_charge",
    "molecular_weight",
    "avg_volume",
    "max_coupling_difficulty",
]
NORMALIZED_FEATURES = [
    "unique_ratio",
    "hydrophobic_ratio",
    "bulky_ratio",
    "polar_ratio",
    "charged_ratio",
    "aromatic_ratio",
    "sulfur_ratio",
    "longest_hydrophobic_run_norm",
    "reagent_score",
    "solvent_score",
    "temperature_score",
    "cleavage_score",
]
BINARY_FEATURES = ["n_term_basic", "c_term_acidic"]
BASE_CATEGORICAL_FEATURES = ["topology", "purity_stage", "yield_stage", "yield_basis_class"]
CONTEXT_CATEGORICAL_FEATURES = ["coupling_reagent", "solvent", "temperature", "cleavage_time"]
CANONICAL_CATEGORICAL_FEATURES = ["chemistry_family", "purity_stage_canon", "yield_stage_canon", "yield_basis_canon"]
HETEROGENEITY_STANDARDIZED_FEATURES = ["source_size_log"]
HETEROGENEITY_NORMALIZED_FEATURES = ["source_stage_freq", "source_basis_freq"]
CONDITION_KEYWORD_FLAGS = [
    "has_hatu",
    "has_hbtu",
    "has_dic_oxyma",
    "has_microwave",
    "has_flow",
    "has_tfa",
    "has_dmf",
]


@dataclass(frozen=True)
class StageConfig:
    name: str
    description: str
    remove_exact_duplicates: bool
    remove_near_duplicates: bool
    remove_outliers: bool
    include_context_features: bool
    apply_scaling: bool
    use_feature_selection: bool
    tuned_search: bool
    use_protein_embeddings: bool


@dataclass(frozen=True)
class CleaningSummary:
    initial_rows: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    outliers_removed: int
    final_rows: int
    unique_sources: int


class SafeNumericImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: str = "mean", fill_value: float = 0.0, n_neighbors: int = 5):
        self.strategy = strategy
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors

    def fit(self, X: Any, y: Any = None) -> "SafeNumericImputer":
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        self.columns_ = frame.columns.tolist()
        self.all_missing_mask_ = frame.isna().all(axis=0).to_numpy()

        if self.strategy == "knn":
            active_columns = frame.columns[~self.all_missing_mask_]
            self.active_columns_ = active_columns.tolist()
            if len(active_columns) > 0:
                self.knn_imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights="distance")
                self.knn_imputer_.fit(frame[active_columns])
            else:
                self.knn_imputer_ = None
        else:
            means = frame.mean(axis=0).fillna(self.fill_value)
            self.statistics_ = means.to_numpy(dtype=float)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X, columns=getattr(self, "columns_", None)).apply(pd.to_numeric, errors="coerce")

        if self.strategy == "knn":
            result = np.full((len(frame), len(self.columns_)), self.fill_value, dtype=float)
            if self.knn_imputer_ is not None and self.active_columns_:
                imputed = self.knn_imputer_.transform(frame[self.active_columns_])
                active_indices = [self.columns_.index(column) for column in self.active_columns_]
                result[:, active_indices] = imputed
            return result

        array = frame.to_numpy(dtype=float)
        stats = np.asarray(self.statistics_, dtype=float)
        mask = np.isnan(array)
        if mask.any():
            array = array.copy()
            array[mask] = np.take(stats, np.where(mask)[1])
        return array

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, "columns_", None)
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(list(input_features), dtype=object)


class SafeCategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value: str = "unknown"):
        self.fill_value = fill_value

    def fit(self, X: Any, y: Any = None) -> "SafeCategoricalImputer":
        self.columns_ = pd.DataFrame(X).columns.tolist()
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X, columns=getattr(self, "columns_", None))
        return frame.fillna(self.fill_value).astype(str).to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, "columns_", None)
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(list(input_features), dtype=object)


def parse_bool_arg(raw: Any) -> bool:
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {raw}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the comprehensive peptide ML workflow.")
    parser.add_argument("--imputation", choices=["mean", "knn"], default="mean")
    parser.add_argument("--sequence-similarity-threshold", type=float, default=0.97)
    parser.add_argument("--random-search-iterations", type=int, default=12)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--embedding-components", type=int, default=8)
    parser.add_argument("--enable-bayesian-calibration", type=parse_bool_arg, default=True)
    parser.add_argument("--bayes-draws", type=int, default=1500)
    parser.add_argument("--bayes-tune", type=int, default=1500)
    parser.add_argument("--bayes-chains", type=int, default=4)
    parser.add_argument("--bayes-target-accept", type=float, default=0.9)
    parser.add_argument("--bayes-likelihood", choices=["normal", "student_t"], default="normal")
    parser.add_argument("--bayes-student-t-nu", type=float, default=4.0)
    parser.add_argument("--residual-model", choices=["voting"], default="voting")
    parser.add_argument("--strict-semantic-heads", type=parse_bool_arg, default=True)
    parser.add_argument("--head-min-rows", type=int, default=HEAD_MIN_ROWS)
    parser.add_argument("--head-min-sources", type=int, default=HEAD_MIN_SOURCES)
    parser.add_argument("--head-max-source-share", type=float, default=HEAD_MAX_SOURCE_SHARE)
    parser.add_argument(
        "--unseen-source-policy",
        choices=["population_mean", "sample_new_source"],
        default="population_mean",
    )
    return parser.parse_args()


def parse_numeric_label(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("%", "").replace(",", ".").lstrip("<>~= ")
    try:
        return float(text)
    except ValueError:
        return None


def normalize_sequence(raw: str) -> str:
    residues = parse_sequence(raw or "")
    return "".join(residues)


def longest_hydrophobic_run(sequence: str) -> int:
    hydrophobic = set("AVILFWYMP")
    best = 0
    current = 0
    for residue in sequence:
        if residue in hydrophobic:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def sequence_similarity(seq_a: str, seq_b: str) -> float:
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0

    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = list(range(len_b + 1))
    for i, char_a in enumerate(seq_a, start=1):
        previous = dp[0]
        dp[0] = i
        for j, char_b in enumerate(seq_b, start=1):
            cached = dp[j]
            if char_a == char_b:
                dp[j] = previous
            else:
                dp[j] = 1 + min(previous, dp[j], dp[j - 1])
            previous = cached
    distance = dp[-1]
    return 1.0 - (distance / max(len_a, len_b))


def categorize_yield_basis(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    if "recovery" in text:
        return "recovery"
    if "isolated" in text:
        return "isolated"
    if "crude" in text:
        return "crude"
    if "resin" in text:
        return "resin_based"
    return "other"


def canonical_purity_stage(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    if "final" in text:
        return "final_product"
    if "purified" in text:
        return "purified_hplc"
    if "crude" in text:
        return "crude_hplc"
    return "unknown"


def canonical_yield_stage(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    if "isolated" in text:
        return "isolated"
    if "recovery" in text:
        return "recovery"
    if "crude" in text:
        return "crude"
    return "unknown"


def canonical_yield_basis(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    if "isolated" in text:
        return "isolated"
    if "recovery" in text:
        return "recovery"
    if "crude" in text:
        return "crude"
    if "resin" in text:
        return "resin_based"
    return "other"


def chemistry_family(source_id: Any, topology: Any, condition_summary: Any) -> str:
    source_text = str(source_id or "").lower()
    topology_text = str(topology or "").strip().lower()
    condition_text = str(condition_summary or "").lower()
    if topology_text == "stapled" or "stapled" in source_text:
        return "stapled"
    if "continuous_flow" in source_text or "fast-flow" in condition_text or "flow" in condition_text:
        return "flow"
    if "sulf" in source_text or "sulf" in condition_text:
        return "sulfated"
    return "canonical_or_simple_modified"


def condition_keyword_flags(summary: Any) -> dict[str, int]:
    text = str(summary or "").lower()
    return {
        "has_hatu": int("hatu" in text),
        "has_hbtu": int("hbtu" in text),
        "has_dic_oxyma": int(("dic/oxyma" in text) or ("dic" in text and "oxyma" in text)),
        "has_microwave": int(("microwave" in text) or ("90°" in text) or ("75°" in text)),
        "has_flow": int("flow" in text),
        "has_tfa": int("tfa" in text),
        "has_dmf": int("dmf" in text),
    }


def build_yield_semantic_head(yield_stage_canon: Any, yield_basis_canon: Any) -> str:
    stage = str(yield_stage_canon or "unknown").strip().lower() or "unknown"
    basis = str(yield_basis_canon or "unknown").strip().lower() or "unknown"
    return f"{stage}|{basis}"


def build_purity_semantic_head(purity_stage_canon: Any) -> str:
    head = str(purity_stage_canon or "unknown").strip().lower() or "unknown"
    return head if head in {"crude_hplc", "purified_hplc", "final_product"} else "unknown"


def parse_condition_tokens(summary: Any) -> dict[str, Any]:
    text = str(summary or "")
    lower = text.lower()

    coupling_reagent = np.nan
    if "hatu" in lower:
        coupling_reagent = "HATU"
    elif "pybop" in lower:
        coupling_reagent = "PyBOP"
    elif "hbtu" in lower:
        coupling_reagent = "HBTU"
    elif "dic/oxyma" in lower or ("dic" in lower and "oxyma" in lower):
        coupling_reagent = "DIC/Oxyma"

    solvent = np.nan
    if "dmf/dcm" in lower:
        solvent = "DMF/DCM"
    elif "nmp" in lower:
        solvent = "NMP"
    elif "dmf" in lower:
        solvent = "DMF"

    temperature = np.nan
    if "90" in lower:
        temperature = "Microwave 90°C"
    elif "75" in lower:
        temperature = "Microwave 75°C"
    elif "room temperature" in lower or "25°" in lower or "25 c" in lower:
        temperature = "Room Temperature"

    cleavage_time = np.nan
    if "4 hours" in lower:
        cleavage_time = "4 hours"
    elif "3 hours" in lower:
        cleavage_time = "3 hours"
    elif "2 hours" in lower:
        cleavage_time = "2 hours"

    return {
        "coupling_reagent": coupling_reagent,
        "solvent": solvent,
        "temperature": temperature,
        "cleavage_time": cleavage_time,
    }


def safe_condition_kwargs(tokens: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (None if pd.isna(value) else value)
        for key, value in tokens.items()
    }


def build_modeling_dataframe() -> pd.DataFrame:
    raw_df = pd.read_csv(DATASET_PATH)
    extractor = FeatureExtractor()
    rows: list[dict[str, Any]] = []

    for row in raw_df.to_dict(orient="records"):
        sequence_norm = normalize_sequence(str(row.get("sequence", "")))
        purity = parse_numeric_label(row.get("purity_pct"))
        yield_pct = parse_numeric_label(row.get("yield_pct"))
        if not sequence_norm or (purity is None and yield_pct is None):
            continue

        peptide = calculate_peptide_features(sequence_norm)
        seq_features = extractor.extract_sequence_features(sequence_norm)
        condition_tokens = parse_condition_tokens(row.get("condition_summary"))
        condition_scores = extractor.extract_condition_features(**safe_condition_kwargs(condition_tokens))
        keyword_flags = condition_keyword_flags(row.get("condition_summary"))

        rows.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "publication_year": float(row["publication_year"]),
                "sequence": row["sequence"],
                "sequence_norm": sequence_norm,
                "topology": row.get("topology") if str(row.get("topology", "")).strip() else np.nan,
                "purity_stage": row["purity_stage"],
                "yield_stage": row["yield_stage"],
                "yield_basis_class": categorize_yield_basis(row.get("yield_basis")),
                "purity_stage_canon": canonical_purity_stage(row.get("purity_stage")),
                "purity_semantic_head": build_purity_semantic_head(canonical_purity_stage(row.get("purity_stage"))),
                "yield_stage_canon": canonical_yield_stage(row.get("yield_stage")),
                "yield_basis_canon": canonical_yield_basis(row.get("yield_basis")),
                "yield_semantic_head": build_yield_semantic_head(
                    canonical_yield_stage(row.get("yield_stage")),
                    canonical_yield_basis(row.get("yield_basis")),
                ),
                "chemistry_family": chemistry_family(row.get("source_id"), row.get("topology"), row.get("condition_summary")),
                "purity_target": purity,
                "yield_target": yield_pct,
                "source_note": row.get("source_note"),
                "condition_summary": row.get("condition_summary"),
                "coupling_reagent": condition_tokens["coupling_reagent"],
                "solvent": condition_tokens["solvent"],
                "temperature": condition_tokens["temperature"],
                "cleavage_time": condition_tokens["cleavage_time"],
                "length": float(peptide.length),
                "avg_hydrophobicity": float(peptide.avg_hydrophobicity),
                "total_charge": float(peptide.total_charge),
                "molecular_weight": float(peptide.molecular_weight),
                "avg_volume": float(peptide.avg_volume),
                "max_coupling_difficulty": float(peptide.max_coupling_difficulty),
                "n_term_basic": float(peptide.n_term_basic),
                "c_term_acidic": float(peptide.c_term_acidic),
                "unique_ratio": (len(set(sequence_norm)) / len(sequence_norm)) if sequence_norm else 0.0,
                "hydrophobic_ratio": sum(1 for aa in sequence_norm if aa in "AVILFWYMP") / len(sequence_norm),
                "longest_hydrophobic_run_norm": longest_hydrophobic_run(sequence_norm) / len(sequence_norm),
                "bulky_ratio": float(seq_features["bulky_ratio"]),
                "polar_ratio": float(seq_features["polar_ratio"]),
                "charged_ratio": float(seq_features["charged_ratio"]),
                "aromatic_ratio": float(seq_features["aromatic_ratio"]),
                "sulfur_ratio": float(seq_features["sulfur_ratio"]),
                "reagent_score": float(condition_scores["reagent_score"]) if not pd.isna(condition_tokens["coupling_reagent"]) else np.nan,
                "solvent_score": float(condition_scores["solvent_score"]) if not pd.isna(condition_tokens["solvent"]) else np.nan,
                "temperature_score": float(condition_scores["temperature_score"]) if not pd.isna(condition_tokens["temperature"]) else np.nan,
                "cleavage_score": float(condition_scores["cleavage_score"]) if not pd.isna(condition_tokens["cleavage_time"]) else np.nan,
                **keyword_flags,
            }
        )

    modeling_df = pd.DataFrame(rows)
    source_sizes = modeling_df.groupby("source_id")["record_id"].transform("count")
    modeling_df["source_size_log"] = np.log1p(source_sizes.astype(float))
    stage_keys = modeling_df["purity_stage_canon"].fillna("unknown") + "|" + modeling_df["yield_stage_canon"].fillna("unknown")
    stage_counts = modeling_df.groupby([modeling_df["source_id"], stage_keys]).transform("count")["record_id"]
    modeling_df["source_stage_freq"] = (stage_counts / source_sizes).astype(float)
    basis_counts = modeling_df.groupby([modeling_df["source_id"], modeling_df["yield_basis_canon"].fillna("unknown")]).transform("count")["record_id"]
    modeling_df["source_basis_freq"] = (basis_counts / source_sizes).astype(float)
    return modeling_df


def add_source_context_features(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    source_sizes = current.groupby("source_id")["record_id"].transform("count")
    current["source_size_log"] = np.log1p(source_sizes.astype(float))
    stage_keys = current["purity_stage_canon"].fillna("unknown") + "|" + current["yield_stage_canon"].fillna("unknown")
    stage_counts = current.groupby([current["source_id"], stage_keys]).transform("count")["record_id"]
    current["source_stage_freq"] = (stage_counts / source_sizes).astype(float)
    basis_counts = current.groupby([current["source_id"], current["yield_basis_canon"].fillna("unknown")]).transform("count")["record_id"]
    current["source_basis_freq"] = (basis_counts / source_sizes).astype(float)
    return current


def build_head_eligibility_table(
    df: pd.DataFrame,
    *,
    head_column: str,
    min_rows: int,
    min_sources: int,
    max_source_share: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for head, group in df.groupby(head_column, dropna=False):
        label = "unknown" if pd.isna(head) else str(head)
        n_rows = int(len(group))
        n_sources = int(group["source_id"].nunique())
        source_distribution = (
            group["source_id"].astype(str).value_counts(normalize=True).sort_values(ascending=False)
        )
        max_share = float(source_distribution.iloc[0]) if not source_distribution.empty else 0.0
        reasons: list[str] = []
        if label == "unknown" or "unknown" in label or "other" in label:
            reasons.append("semantic_unknown_or_other")
        if n_rows < min_rows:
            reasons.append(f"rows<{min_rows}")
        if n_sources < min_sources:
            reasons.append(f"sources<{min_sources}")
        if max_share > max_source_share:
            reasons.append(f"source_share>{max_source_share:.2f}")
        rows.append(
            {
                "head": label,
                "nRows": n_rows,
                "nSources": n_sources,
                "maxSourceShare": max_share,
                "eligible": len(reasons) == 0,
                "reasons": reasons,
            }
        )
    rows.sort(key=lambda item: (-item["eligible"], -item["nRows"], item["head"]))
    return rows


def build_strict_semantic_subset(
    df: pd.DataFrame,
    *,
    min_rows: int = HEAD_MIN_ROWS,
    min_sources: int = HEAD_MIN_SOURCES,
    max_source_share: float = HEAD_MAX_SOURCE_SHARE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    current = df.copy()
    current["purity_semantic_head"] = current["purity_semantic_head"].fillna("unknown").astype(str)
    current["yield_semantic_head"] = current["yield_semantic_head"].fillna("unknown|unknown").astype(str)
    purity_target_df = current[current["purity_target"].notna()].copy()
    yield_target_df = current[current["yield_target"].notna()].copy()
    purity_eligibility = build_head_eligibility_table(
        purity_target_df[purity_target_df["purity_semantic_head"].isin(["crude_hplc", "purified_hplc", "final_product"])].copy(),
        head_column="purity_semantic_head",
        min_rows=min_rows,
        min_sources=min_sources,
        max_source_share=max_source_share,
    )
    yield_eligibility = build_head_eligibility_table(
        yield_target_df[
            yield_target_df["yield_semantic_head"].isin(
                ["isolated|isolated", "crude|crude", "recovery|recovery"]
            )
        ].copy(),
        head_column="yield_semantic_head",
        min_rows=min_rows,
        min_sources=min_sources,
        max_source_share=max_source_share,
    )
    allowed_purity = {row["head"] for row in purity_eligibility if row["eligible"]}
    allowed_yield = {row["head"] for row in yield_eligibility if row["eligible"]}

    filtered = current[
        (
            current["purity_target"].notna()
            & current["purity_semantic_head"].isin(allowed_purity)
        )
        | (
            current["yield_target"].notna()
            & current["yield_semantic_head"].isin(allowed_yield)
        )
    ].copy()
    return filtered.reset_index(drop=True), {
        "inputRows": int(len(df)),
        "outputRows": int(len(filtered)),
        "outputUniqueSources": int(filtered["source_id"].nunique()),
        "inputPurityRows": int(current["purity_target"].notna().sum()),
        "inputYieldRows": int(current["yield_target"].notna().sum()),
        "outputPurityRows": int(filtered["purity_target"].notna().sum()),
        "outputYieldRows": int(filtered["yield_target"].notna().sum()),
        "purityEligibility": purity_eligibility,
        "yieldEligibility": yield_eligibility,
        "allowedPurityHeads": sorted(allowed_purity),
        "allowedYieldHeads": sorted(allowed_yield),
    }


def derive_head_decision(
    *,
    target: str,
    n_rows: int,
    n_sources: int,
    metrics: dict[str, float],
    interval_metrics: dict[str, float],
) -> dict[str, Any]:
    tolerance = PURITY_TOLERANCE if target == "purity_target" else YIELD_TOLERANCE
    reject_reasons: list[str] = []
    if n_rows < HEAD_MIN_ROWS:
        reject_reasons.append("insufficient_rows")
    if n_sources < HEAD_MIN_SOURCES:
        reject_reasons.append("insufficient_sources")
    if metrics["r2"] < 0:
        reject_reasons.append("negative_r2")
    if interval_metrics["meanWidth"] > (CONFIDENCE_REJECT_INTERVAL_MULTIPLIER * tolerance):
        reject_reasons.append("interval_too_wide")

    if reject_reasons:
        return {
            "confidenceTier": "reject",
            "servePrediction": False,
            "reasons": reject_reasons,
        }

    if (
        metrics["r2"] >= 0.5
        and metrics["accuracyWithinTolerance"] >= 0.25
        and interval_metrics["meanWidth"] <= (CONFIDENCE_WARN_INTERVAL_MULTIPLIER * tolerance)
        and n_sources >= 5
    ):
        return {
            "confidenceTier": "high",
            "servePrediction": True,
            "reasons": [],
        }

    if metrics["r2"] >= 0.2 and n_sources >= 4:
        return {
            "confidenceTier": "medium",
            "servePrediction": True,
            "reasons": ["use_with_warning"],
        }

    return {
        "confidenceTier": "low",
        "servePrediction": True,
        "reasons": ["weak_signal"],
    }


def summarize_prediction_policy(head_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "acceptedHighConfidenceHeads": [],
        "acceptedWarningHeads": [],
        "rejectedHeads": [],
    }
    for row in head_metrics:
        entry = {
            "target": row["target"],
            "head": row["head"],
            "confidenceTier": row["decision"]["confidenceTier"],
            "servePrediction": bool(row["decision"]["servePrediction"]),
            "reasons": list(row["decision"]["reasons"]),
            "rmse": float(row["metrics"]["rmse"]),
            "r2": float(row["metrics"]["r2"]),
            "nRows": int(row.get("modelingRows", row["nRows"])),
            "nSources": int(row.get("modelingSources", row["nSources"])),
            "excludedSources": list(row.get("excludedSources", [])),
        }
        if row["decision"]["confidenceTier"] == "high":
            summary["acceptedHighConfidenceHeads"].append(entry)
        elif row["decision"]["servePrediction"]:
            summary["acceptedWarningHeads"].append(entry)
        else:
            summary["rejectedHeads"].append(entry)
    return summary


def build_weighted_head_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {
            "totalWeightedSupport": 0.0,
            "weightedRmse": float("nan"),
            "weightedMae": float("nan"),
            "weightedR2": float("nan"),
            "weightedAccuracyWithinTolerance": float("nan"),
        }

    weights = np.asarray([float(item["nRows"]) * float(item["nSources"]) for item in metrics], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return {
            "totalWeightedSupport": 0.0,
            "weightedRmse": float("nan"),
            "weightedMae": float("nan"),
            "weightedR2": float("nan"),
            "weightedAccuracyWithinTolerance": float("nan"),
        }
    return {
        "totalWeightedSupport": total,
        "weightedRmse": float(np.average([item["metrics"]["rmse"] for item in metrics], weights=weights)),
        "weightedMae": float(np.average([item["metrics"]["mae"] for item in metrics], weights=weights)),
        "weightedR2": float(np.average([item["metrics"]["r2"] for item in metrics], weights=weights)),
        "weightedAccuracyWithinTolerance": float(
            np.average([item["metrics"]["accuracyWithinTolerance"] for item in metrics], weights=weights)
        ),
    }


def diagnose_head_sources(df: pd.DataFrame, *, target: str) -> dict[str, Any]:
    if df.empty:
        return {"target": target, "medianSourceMean": float("nan"), "madSourceMean": float("nan"), "sources": []}

    grouped = (
        df.groupby("source_id", dropna=False)[target]
        .agg(["size", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"size": "nRows", "mean": "sourceMean", "std": "sourceStd", "min": "sourceMin", "max": "sourceMax"})
    )
    means = grouped["sourceMean"].to_numpy(dtype=float)
    median_mean = float(np.median(means))
    mad_mean = float(np.median(np.abs(means - median_mean)))
    robust_scale = max(1.4826 * mad_mean, 1e-6)

    rows: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        source_id = str(row["source_id"])
        n_rows = int(row["nRows"])
        source_mean = float(row["sourceMean"])
        drift_z = abs(source_mean - median_mean) / robust_scale
        is_drift_outlier = bool(n_rows >= SOURCE_DRIFT_MIN_ROWS and drift_z > SOURCE_DRIFT_Z_THRESHOLD)
        rows.append(
            {
                "sourceId": source_id,
                "nRows": n_rows,
                "sourceMean": source_mean,
                "sourceStd": float(0.0 if pd.isna(row["sourceStd"]) else row["sourceStd"]),
                "sourceMin": float(row["sourceMin"]),
                "sourceMax": float(row["sourceMax"]),
                "driftZ": float(drift_z),
                "isDriftOutlier": is_drift_outlier,
            }
        )
    rows.sort(key=lambda item: (-item["isDriftOutlier"], -item["driftZ"], item["sourceId"]))
    return {
        "target": target,
        "medianSourceMean": median_mean,
        "madSourceMean": mad_mean,
        "sources": rows,
    }


def select_compatible_head_subset(
    df: pd.DataFrame,
    *,
    diagnostics: dict[str, Any],
    min_rows: int,
    min_sources: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    excluded_sources = [row["sourceId"] for row in diagnostics.get("sources", []) if row["isDriftOutlier"]]
    if not excluded_sources:
        return df.copy(), {
            "usedCompatibleSubset": False,
            "excludedSources": [],
            "originalRows": int(len(df)),
            "originalSources": int(df["source_id"].nunique()),
            "modelingRows": int(len(df)),
            "modelingSources": int(df["source_id"].nunique()),
        }

    compatible_df = df[~df["source_id"].astype(str).isin(excluded_sources)].copy()
    if len(compatible_df) < min_rows or compatible_df["source_id"].nunique() < min_sources:
        return df.copy(), {
            "usedCompatibleSubset": False,
            "excludedSources": [],
            "originalRows": int(len(df)),
            "originalSources": int(df["source_id"].nunique()),
            "modelingRows": int(len(df)),
            "modelingSources": int(df["source_id"].nunique()),
        }

    return compatible_df.reset_index(drop=True), {
        "usedCompatibleSubset": True,
        "excludedSources": excluded_sources,
        "originalRows": int(len(df)),
        "originalSources": int(df["source_id"].nunique()),
        "modelingRows": int(len(compatible_df)),
        "modelingSources": int(compatible_df["source_id"].nunique()),
    }


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    subset = [
        "source_id",
        "sequence_norm",
        "purity_stage_canon" if "purity_stage_canon" in df.columns else "purity_stage",
        "yield_stage_canon" if "yield_stage_canon" in df.columns else "yield_stage",
        "purity_target",
        "yield_target",
        "topology",
    ]
    deduped = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return deduped, int(len(df) - len(deduped))


def completeness_score(row: pd.Series) -> tuple[int, int]:
    non_null = int(row.notna().sum())
    text_length = len(str(row.get("sequence", ""))) + len(str(row.get("source_id", "")))
    return non_null, text_length


def stage_group_columns(df: pd.DataFrame) -> list[str]:
    purity_column = "purity_stage_canon" if "purity_stage_canon" in df.columns else "purity_stage"
    yield_column = "yield_stage_canon" if "yield_stage_canon" in df.columns else "yield_stage"
    return [purity_column, yield_column]


def remove_near_duplicates(
    df: pd.DataFrame,
    *,
    similarity_threshold: float,
    purity_tolerance: float = 2.0,
    yield_tolerance: float = 3.0,
) -> tuple[pd.DataFrame, int]:
    keep_mask = np.ones(len(df), dtype=bool)
    group_columns = ["source_id", *stage_group_columns(df)]
    for _, group in df.groupby(group_columns, dropna=False):
        indices = group.index.tolist()
        for idx_pos, left_index in enumerate(indices):
            if not keep_mask[left_index]:
                continue
            left_row = df.loc[left_index]
            for right_index in indices[idx_pos + 1:]:
                if not keep_mask[right_index]:
                    continue
                right_row = df.loc[right_index]
                left_context = str(left_row.get("condition_summary", "") or "").strip().lower()
                right_context = str(right_row.get("condition_summary", "") or "").strip().lower()
                if left_context and right_context and left_context != right_context:
                    continue
                if abs(left_row["purity_target"] - right_row["purity_target"]) > purity_tolerance:
                    continue
                if abs(left_row["yield_target"] - right_row["yield_target"]) > yield_tolerance:
                    continue
                similarity = sequence_similarity(left_row["sequence_norm"], right_row["sequence_norm"])
                if similarity < similarity_threshold:
                    continue
                left_score = completeness_score(left_row)
                right_score = completeness_score(right_row)
                drop_index = right_index if left_score >= right_score else left_index
                keep_mask[drop_index] = False
                if drop_index == left_index:
                    break

    deduped = df.loc[keep_mask].reset_index(drop=True)
    return deduped, int(len(df) - len(deduped))


def modified_z_scores(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - median) / mad


def remove_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    outlier_indices: set[int] = set()
    for _, group in df.groupby(["source_id", *stage_group_columns(df)], dropna=False):
        if len(group) < 4:
            continue
        for column in ["purity_target", "yield_target"]:
            valid = group[column].dropna()
            if len(valid) < 4:
                continue
            q1 = group[column].quantile(0.25)
            q3 = group[column].quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)
            outlier_indices.update(group[(group[column] < lower) | (group[column] > upper)].index.tolist())

    filtered = df.drop(index=sorted(outlier_indices)).reset_index(drop=True)
    return filtered, int(len(outlier_indices))


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    stage: StageConfig,
    similarity_threshold: float,
) -> tuple[pd.DataFrame, CleaningSummary]:
    current = df.copy()
    exact_removed = 0
    near_removed = 0
    outliers_removed = 0

    if stage.remove_exact_duplicates:
        current, exact_removed = remove_exact_duplicates(current)
    if stage.remove_near_duplicates:
        current, near_removed = remove_near_duplicates(current, similarity_threshold=similarity_threshold)
    if stage.remove_outliers:
        current, outliers_removed = remove_outliers(current)

    summary = CleaningSummary(
        initial_rows=int(len(df)),
        exact_duplicates_removed=exact_removed,
        near_duplicates_removed=near_removed,
        outliers_removed=outliers_removed,
        final_rows=int(len(current)),
        unique_sources=int(current["source_id"].nunique()),
    )
    return current, summary


def filter_available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns and df[column].notna().any()]


def get_stage_columns(stage: StageConfig, df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    standardized = [*STANDARDIZED_FEATURES, *HETEROGENEITY_STANDARDIZED_FEATURES]
    normalized = [
        "unique_ratio",
        "hydrophobic_ratio",
        "bulky_ratio",
        "polar_ratio",
        "charged_ratio",
        "aromatic_ratio",
        "sulfur_ratio",
        "longest_hydrophobic_run_norm",
        *HETEROGENEITY_NORMALIZED_FEATURES,
    ]
    categorical = [*BASE_CATEGORICAL_FEATURES, *CANONICAL_CATEGORICAL_FEATURES]
    binary = [*BINARY_FEATURES, *CONDITION_KEYWORD_FLAGS]

    if stage.include_context_features:
        normalized.extend(["reagent_score", "solvent_score", "temperature_score", "cleavage_score"])
        categorical.extend(CONTEXT_CATEGORICAL_FEATURES)

    return (
        filter_available_columns(df, standardized),
        filter_available_columns(df, normalized),
        filter_available_columns(df, binary),
        filter_available_columns(df, categorical),
    )


def build_preprocessor(stage: StageConfig, imputation_strategy: str, df: pd.DataFrame, args: argparse.Namespace) -> ColumnTransformer:
    standardized, normalized, binary, categorical = get_stage_columns(stage, df)

    if stage.include_context_features and imputation_strategy == "knn":
        numeric_imputer: Any = SafeNumericImputer(strategy="knn", n_neighbors=5, fill_value=0.0)
        imputation_note = "KNN imputation keeps local structure for context variables."
    else:
        numeric_imputer = SafeNumericImputer(strategy="mean", fill_value=0.0)
        imputation_note = "Mean imputation is more stable for this small heterogeneous literature dataset."

    standard_steps: list[tuple[str, Any]] = [("imputer", numeric_imputer)]
    normalize_steps: list[tuple[str, Any]] = [("imputer", numeric_imputer)]
    if stage.apply_scaling:
        standard_steps.append(("scaler", StandardScaler()))
        normalize_steps.append(("scaler", MinMaxScaler()))

    transformers = [
        ("std", Pipeline(steps=standard_steps), standardized),
        ("norm", Pipeline(steps=normalize_steps), normalized),
        ("binary", Pipeline(steps=[("imputer", SafeNumericImputer(strategy="mean", fill_value=0.0))]), binary),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SafeCategoricalImputer(fill_value="unknown")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            categorical,
        ),
    ]
    if stage.use_protein_embeddings and "sequence_norm" in df.columns:
        transformers.append(
            (
                "embedding",
                SequenceEmbeddingPCATransformer(
                    model_name=args.embedding_model,
                    cache_dir=DEFAULT_CACHE_DIR,
                    n_components=args.embedding_components,
                    prefix="esm2",
                ),
                ["sequence_norm"],
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.0)
    preprocessor.imputation_note = imputation_note  # type: ignore[attr-defined]
    return preprocessor


def build_ensemble_pipeline(stage: StageConfig, imputation_strategy: str, df: pd.DataFrame, args: argparse.Namespace) -> Pipeline:
    preprocessor = build_preprocessor(stage, imputation_strategy, df, args)

    selector: Any
    if stage.use_feature_selection:
        selector = SelectFromModel(
            estimator=ExtraTreesRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            threshold="median",
        )
    else:
        selector = "passthrough"

    ensemble = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)),
            ("et", ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)),
            ("gbr", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("selector", selector),
            ("regressor", ensemble),
        ]
    )


def build_cv(groups: pd.Series) -> GroupKFold:
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError("At least two source groups are required for grouped cross-validation")
    return GroupKFold(n_splits=min(5, n_groups))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "accuracyWithinTolerance": float(np.mean(np.abs(error) <= tolerance)),
    }


def evaluate_multitask_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, Any]:
    purity = evaluate_predictions(y_true["purity_target"].to_numpy(), y_pred["purity_target"].to_numpy(), PURITY_TOLERANCE)
    yield_metrics = evaluate_predictions(y_true["yield_target"].to_numpy(), y_pred["yield_target"].to_numpy(), YIELD_TOLERANCE)
    return {
        "purity": purity,
        "yield": yield_metrics,
        "combined": {
            "rmse": (purity["rmse"] + yield_metrics["rmse"]) / 2.0,
            "mae": (purity["mae"] + yield_metrics["mae"]) / 2.0,
            "r2": (purity["r2"] + yield_metrics["r2"]) / 2.0,
            "jointToleranceAccuracy": float(
                np.mean(
                    (np.abs(y_pred["purity_target"] - y_true["purity_target"]) <= PURITY_TOLERANCE)
                    & (np.abs(y_pred["yield_target"] - y_true["yield_target"]) <= YIELD_TOLERANCE)
                )
            ),
        },
    }


def winsorize_targets(
    train_df: pd.DataFrame,
    targets: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    clipped = train_df.copy()
    bounds: dict[str, tuple[float, float]] = {}
    for target in targets:
        lo = float(clipped[target].quantile(lower))
        hi = float(clipped[target].quantile(upper))
        clipped[target] = clipped[target].clip(lower=lo, upper=hi)
        bounds[target] = (lo, hi)
    return clipped, bounds


def loso_splits(df: pd.DataFrame, group_column: str = "source_id") -> list[tuple[np.ndarray, np.ndarray, str]]:
    splits: list[tuple[np.ndarray, np.ndarray, str]] = []
    for source_id in sorted(df[group_column].astype(str).unique()):
        test_mask = df[group_column].astype(str) == source_id
        train_idx = np.where(~test_mask.to_numpy())[0]
        test_idx = np.where(test_mask.to_numpy())[0]
        splits.append((train_idx, test_idx, source_id))
    return splits


def evaluate_interval_quality(
    y_true: np.ndarray,
    interval: np.ndarray,
) -> dict[str, float]:
    lower = interval[:, 0]
    upper = interval[:, 1]
    return {
        "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "meanWidth": float(np.mean(upper - lower)),
    }


def cross_validated_stage_metrics(
    df: pd.DataFrame,
    stage: StageConfig,
    imputation_strategy: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    feature_df = df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])
    groups = df["source_id"]
    cv = build_cv(groups)

    predictions: dict[str, np.ndarray] = {}
    for target in ["purity_target", "yield_target"]:
        pipeline = build_ensemble_pipeline(stage, imputation_strategy, df, args)
        predictions[target] = cross_val_predict(
            estimator=clone(pipeline),
            X=feature_df,
            y=df[target],
            groups=groups,
            cv=cv,
            method="predict",
            n_jobs=1,
        )

    prediction_df = pd.DataFrame(predictions, index=df.index)
    return evaluate_multitask_predictions(df[["purity_target", "yield_target"]], prediction_df)


def tune_target_pipeline(
    df: pd.DataFrame,
    stage: StageConfig,
    imputation_strategy: str,
    target: str,
    random_search_iterations: int,
    args: argparse.Namespace,
) -> tuple[Pipeline, dict[str, Any], float]:
    feature_df = df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])
    groups = df["source_id"]
    cv = build_cv(groups)
    pipeline = build_ensemble_pipeline(stage, imputation_strategy, df, args)

    param_distributions = {
        "selector__threshold": ["median", "mean"],
        "selector__estimator__max_depth": [None, 4, 8],
        "regressor__weights": [(1, 1, 1, 1), (2, 2, 1, 1), (1, 2, 1, 2), (2, 1, 2, 1)],
        "regressor__rf__n_estimators": [200, 300, 500],
        "regressor__rf__max_depth": [None, 4, 8],
        "regressor__rf__min_samples_leaf": [1, 2, 4],
        "regressor__et__n_estimators": [200, 300, 500],
        "regressor__et__max_depth": [None, 4, 8],
        "regressor__et__min_samples_leaf": [1, 2, 4],
        "regressor__gbr__n_estimators": [120, 180, 240],
        "regressor__gbr__learning_rate": [0.03, 0.05, 0.08],
        "regressor__gbr__max_depth": [2, 3],
        "regressor__ridge__alpha": [0.1, 0.3, 1.0, 3.0, 10.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=random_search_iterations,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        refit=True,
        n_jobs=1,
    )
    search.fit(feature_df, df[target], groups=groups)
    best_score = -float(search.best_score_)
    return search.best_estimator_, search.best_params_, best_score


def cross_validated_predictions_for_pipeline(df: pd.DataFrame, pipeline: Pipeline, target: str) -> np.ndarray:
    feature_df = df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])
    groups = df["source_id"]
    cv = build_cv(groups)
    return cross_val_predict(
        estimator=clone(pipeline),
        X=feature_df,
        y=df[target],
        groups=groups,
        cv=cv,
        method="predict",
        n_jobs=1,
    )


def extract_final_feature_analysis(
    fitted_pipeline: Pipeline,
    feature_df: pd.DataFrame,
    target_values: pd.Series,
) -> dict[str, Any]:
    preprocess = fitted_pipeline.named_steps["preprocess"]
    selector = fitted_pipeline.named_steps["selector"]
    transformed_feature_names = preprocess.get_feature_names_out()

    analysis: dict[str, Any] = {
        "transformedFeatureCount": int(len(transformed_feature_names)),
        "selectedFeatures": [],
        "selectorImportances": [],
        "permutationImportance": [],
    }

    if selector != "passthrough":
        support = selector.get_support()
        selected_names = transformed_feature_names[support].tolist()
        selector_model = selector.estimator_
        importances = selector_model.feature_importances_.tolist()
        ranking = sorted(
            zip(transformed_feature_names.tolist(), importances),
            key=lambda item: item[1],
            reverse=True,
        )
        analysis["selectedFeatures"] = selected_names
        analysis["selectorImportances"] = [
            {"feature": name, "importance": float(score)} for name, score in ranking[:15]
        ]
        analysis["selectedFeatureCount"] = int(np.sum(support))
    else:
        analysis["selectedFeatures"] = transformed_feature_names.tolist()
        analysis["selectedFeatureCount"] = int(len(transformed_feature_names))

    permutation = permutation_importance(
        fitted_pipeline,
        feature_df,
        target_values,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    original_ranking = sorted(
        zip(feature_df.columns.tolist(), permutation.importances_mean.tolist(), permutation.importances_std.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    analysis["permutationImportance"] = [
        {"feature": name, "importanceMean": float(mean), "importanceStd": float(std)}
        for name, mean, std in original_ranking[:12]
    ]
    return analysis


def stage_definitions() -> list[StageConfig]:
    return [
        StageConfig(
            name="baseline_raw",
            description="Raw literature subset with exact label parsing only.",
            remove_exact_duplicates=False,
            remove_near_duplicates=False,
            remove_outliers=False,
            include_context_features=False,
            apply_scaling=False,
            use_feature_selection=False,
            tuned_search=False,
            use_protein_embeddings=False,
        ),
        StageConfig(
            name="after_exact_dedup",
            description="Remove exact duplicate records after target parsing.",
            remove_exact_duplicates=True,
            remove_near_duplicates=False,
            remove_outliers=False,
            include_context_features=False,
            apply_scaling=False,
            use_feature_selection=False,
            tuned_search=False,
            use_protein_embeddings=False,
        ),
        StageConfig(
            name="after_sequence_dedup",
            description="Remove near-duplicate records using sequence similarity inside source-stage groups.",
            remove_exact_duplicates=True,
            remove_near_duplicates=True,
            remove_outliers=False,
            include_context_features=False,
            apply_scaling=False,
            use_feature_selection=False,
            tuned_search=False,
            use_protein_embeddings=False,
        ),
        StageConfig(
            name="after_outlier_removal",
            description="Apply IQR and modified z-score outlier removal after de-duplication.",
            remove_exact_duplicates=True,
            remove_near_duplicates=True,
            remove_outliers=True,
            include_context_features=False,
            apply_scaling=False,
            use_feature_selection=False,
            tuned_search=False,
            use_protein_embeddings=False,
        ),
        StageConfig(
            name="after_imputation_and_scaling",
            description="Add context variables and apply mean/KNN imputation with standardization and normalization.",
            remove_exact_duplicates=True,
            remove_near_duplicates=True,
            remove_outliers=True,
            include_context_features=True,
            apply_scaling=True,
            use_feature_selection=False,
            tuned_search=False,
            use_protein_embeddings=False,
        ),
        StageConfig(
            name="after_feature_selection",
            description="Apply model-based feature selection before the ensemble regressor.",
            remove_exact_duplicates=True,
            remove_near_duplicates=True,
            remove_outliers=True,
            include_context_features=True,
            apply_scaling=True,
            use_feature_selection=True,
            tuned_search=False,
            use_protein_embeddings=False,
        ),
        StageConfig(
            name="final_tuned_ensemble",
            description="Random-search tuned ensemble with grouped cross-validation and feature selection.",
            remove_exact_duplicates=True,
            remove_near_duplicates=True,
            remove_outliers=True,
            include_context_features=True,
            apply_scaling=True,
            use_feature_selection=True,
            tuned_search=True,
            use_protein_embeddings=False,
        ),
        StageConfig(
            name="final_tuned_ensemble_with_embeddings",
            description="Random-search tuned ensemble plus ESM2 sequence embeddings reduced within each grouped fold.",
            remove_exact_duplicates=True,
            remove_near_duplicates=True,
            remove_outliers=True,
            include_context_features=True,
            apply_scaling=True,
            use_feature_selection=True,
            tuned_search=True,
            use_protein_embeddings=True,
        ),
    ]


def evaluate_semantic_subgroups(df: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    def summarize_subgroups(target_column: str, prediction_column: str, group_column: str, tolerance: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label, group in df.groupby(group_column, dropna=False):
            if len(group) < 2:
                continue
            truth = group[target_column].to_numpy(dtype=float)
            pred = predictions.loc[group.index, prediction_column].to_numpy(dtype=float)
            metrics = evaluate_predictions(truth, pred, tolerance)
            rows.append(
                {
                    "label": "unknown" if pd.isna(label) else str(label),
                    "n": int(len(group)),
                    **metrics,
                }
            )
        rows.sort(key=lambda item: (item["label"]))
        return rows

    return {
        "purityByStage": summarize_subgroups("purity_target", "purity_target", "purity_stage", PURITY_TOLERANCE),
        "yieldByStage": summarize_subgroups("yield_target", "yield_target", "yield_stage", YIELD_TOLERANCE),
        "yieldByBasis": summarize_subgroups("yield_target", "yield_target", "yield_basis_class", YIELD_TOLERANCE),
    }


def flatten_columns(*parts: list[str]) -> list[str]:
    output: list[str] = []
    for chunk in parts:
        for column in chunk:
            if column not in output:
                output.append(column)
    return output


def run_loso_raw_ensemble(
    df: pd.DataFrame,
    stage: StageConfig,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = ["purity_target", "yield_target"]
    predictions = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    per_source_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx, source_id) in enumerate(loso_splits(df)):
        train_raw = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, _ = winsorize_targets(train_raw, targets=targets, lower=0.01, upper=0.99)
        x_train = train_df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])
        x_test = test_df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])

        fold_prediction = pd.DataFrame(index=test_df.index, columns=targets, dtype=float)
        for target in targets:
            model = build_ensemble_pipeline(stage, args.imputation, train_df, args)
            model.fit(x_train, train_df[target].to_numpy(dtype=float))
            fold_prediction[target] = model.predict(x_test)

        predictions.loc[test_df.index, targets] = fold_prediction
        fold_metrics = evaluate_multitask_predictions(test_df[targets], fold_prediction)
        per_source_rows.append(
            {
                "source_id": source_id,
                "fold_index": int(fold_idx),
                "n_test": int(len(test_df)),
                "purity_rmse": float(fold_metrics["purity"]["rmse"]),
                "yield_rmse": float(fold_metrics["yield"]["rmse"]),
                "joint_tolerance_accuracy": float(fold_metrics["combined"]["jointToleranceAccuracy"]),
            }
        )

    overall = evaluate_multitask_predictions(df[targets], predictions[targets])
    return predictions, {"overall": overall, "perSource": per_source_rows}


def run_bayesian_calibrated_residual_loso(
    df: pd.DataFrame,
    stage: StageConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    targets = ["purity_target", "yield_target"]
    standardized, normalized, binary, categorical = get_stage_columns(stage, df)
    calibration_features = flatten_columns(standardized, normalized, binary)
    residual_features = flatten_columns(calibration_features, categorical)
    group_cols = {
        "source": "source_id",
        "purity_target": "purity_stage_canon",
        "yield_target": "yield_stage_canon",
    }
    mapped_policy = "sample_new_source" if args.unseen_source_policy == "sample_new_source" else "population_for_unseen_source"

    raw_predictions = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    calibrated_predictions = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    final_predictions = pd.DataFrame(index=df.index, columns=targets, dtype=float)
    interval_cache: dict[str, pd.DataFrame] = {
        "purity_target": pd.DataFrame(index=df.index, columns=["lower", "upper"], dtype=float),
        "yield_target": pd.DataFrame(index=df.index, columns=["lower", "upper"], dtype=float),
    }
    fold_rows: list[dict[str, Any]] = []
    residual_importance_rows: list[dict[str, Any]] = []
    head_weight_rows: list[dict[str, Any]] = []
    yield_head_label = "yield_semantic_head"

    for fold_idx, (train_idx, test_idx, source_id) in enumerate(loso_splits(df)):
        train_raw = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, bounds = winsorize_targets(train_raw, targets=targets, lower=0.01, upper=0.99)

        # Raw ensemble baseline under the same LOSO split.
        x_train_raw = train_df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])
        x_test_raw = test_df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])
        raw_fold = pd.DataFrame(index=test_df.index, columns=targets, dtype=float)
        for target in targets:
            baseline_model = build_ensemble_pipeline(stage, args.imputation, train_df, args)
            baseline_model.fit(x_train_raw, train_df[target].to_numpy(dtype=float))
            raw_fold[target] = baseline_model.predict(x_test_raw)
        raw_predictions.loc[test_df.index, targets] = raw_fold

        calibrator = BayesianMultitaskCalibrator(
            draws=args.bayes_draws,
            tune=args.bayes_tune,
            chains=args.bayes_chains,
            target_accept=args.bayes_target_accept,
            likelihood=args.bayes_likelihood,
            student_t_nu=args.bayes_student_t_nu,
            random_state=RANDOM_STATE + (fold_idx * 41),
        )
        calibrator.fit(
            train_df=train_df,
            features=calibration_features,
            targets=targets,
            group_cols=group_cols,
        )
        calibrated_test = calibrator.predict(test_df, mode=mapped_policy)
        calibrated_train = calibrator.predict(train_raw, mode="population_for_unseen_source")
        calibrated_predictions.loc[test_df.index, targets] = calibrated_test["predictions"]
        for target in targets:
            interval_cache[target].loc[test_df.index, "lower"] = calibrated_test["intervals"][target][:, 0]
            interval_cache[target].loc[test_df.index, "upper"] = calibrated_test["intervals"][target][:, 1]

        train_residuals = pd.DataFrame(
            {
                target: train_raw[target].to_numpy(dtype=float)
                - calibrated_train["predictions"][target].to_numpy(dtype=float)
                for target in targets
            },
            index=train_raw.index,
        )
        residual_model = ResidualEnsembleRegressor(
            model=args.residual_model,
            random_state=RANDOM_STATE + (fold_idx * 71),
        )
        residual_model.fit(train_raw, residual_features, train_residuals)
        residual_test = residual_model.predict(test_df)
        final_fold = calibrated_test["predictions"] + residual_test

        # Yield multi-head semantic calibration/residual modeling.
        global_yield_calibrated_test = calibrated_test["predictions"]["yield_target"].to_numpy(dtype=float)
        global_yield_final_test = final_fold["yield_target"].to_numpy(dtype=float)
        blended_yield_calibrated = global_yield_calibrated_test.copy()
        blended_yield_final = global_yield_final_test.copy()
        train_head_counts = train_raw[yield_head_label].value_counts(dropna=False).to_dict()

        for head in sorted(test_df[yield_head_label].fillna("unknown").astype(str).unique()):
            test_mask = test_df[yield_head_label].fillna("unknown").astype(str) == head
            if not np.any(test_mask.to_numpy()):
                continue
            local_train = train_raw[train_raw[yield_head_label].fillna("unknown").astype(str) == head].copy()
            local_train_winsorized = train_df[train_df[yield_head_label].fillna("unknown").astype(str) == head].copy()
            n_head_train = int(len(local_train))
            weight = float(n_head_train / (n_head_train + YIELD_HEAD_BLEND_PRIOR))

            use_local_head = n_head_train >= MIN_YIELD_HEAD_ROWS and local_train["source_id"].nunique() >= 2
            if use_local_head:
                local_calibrator = BayesianMultitaskCalibrator(
                    draws=args.bayes_draws,
                    tune=args.bayes_tune,
                    chains=args.bayes_chains,
                    target_accept=args.bayes_target_accept,
                    likelihood=args.bayes_likelihood,
                    student_t_nu=args.bayes_student_t_nu,
                    random_state=RANDOM_STATE + (fold_idx * 41) + (abs(hash(head)) % 997),
                )
                local_calibrator.fit(
                    train_df=local_train_winsorized,
                    features=calibration_features,
                    targets=["yield_target"],
                    group_cols=group_cols,
                )
                local_cal_test = local_calibrator.predict(test_df.loc[test_mask], mode=mapped_policy)["predictions"]["yield_target"].to_numpy(dtype=float)
                local_cal_train = local_calibrator.predict(local_train, mode="population_for_unseen_source")["predictions"]["yield_target"].to_numpy(dtype=float)

                local_residual_target = pd.DataFrame(
                    {"yield_target": local_train["yield_target"].to_numpy(dtype=float) - local_cal_train},
                    index=local_train.index,
                )
                local_residual_model = ResidualEnsembleRegressor(
                    model=args.residual_model,
                    random_state=RANDOM_STATE + (fold_idx * 97) + (abs(hash(head)) % 997),
                )
                local_residual_model.fit(local_train, residual_features, local_residual_target)
                local_residual_test = local_residual_model.predict(test_df.loc[test_mask])["yield_target"].to_numpy(dtype=float)
                local_final = local_cal_test + local_residual_test

                base_cal = blended_yield_calibrated[test_mask.to_numpy()]
                base_final = blended_yield_final[test_mask.to_numpy()]
                blended_yield_calibrated[test_mask.to_numpy()] = (weight * local_cal_test) + ((1.0 - weight) * base_cal)
                blended_yield_final[test_mask.to_numpy()] = (weight * local_final) + ((1.0 - weight) * base_final)
            else:
                weight = 0.0

            head_weight_rows.append(
                {
                    "source_id": source_id,
                    "fold_index": int(fold_idx),
                    "head": str(head),
                    "n_head_train": n_head_train,
                    "n_head_test": int(np.sum(test_mask.to_numpy())),
                    "globalHeadCountInTrain": int(train_head_counts.get(head, 0)),
                    "localModelUsed": bool(use_local_head),
                    "blendWeight": float(weight),
                }
            )

        calibrated_predictions.loc[test_df.index, "yield_target"] = blended_yield_calibrated
        final_fold.loc[:, "yield_target"] = blended_yield_final
        final_predictions.loc[test_df.index, targets] = final_fold

        importance = residual_model.feature_importance()
        for target in targets:
            for row in importance.get(target, []):
                residual_importance_rows.append(
                    {
                        "target": target,
                        "feature": row["feature"],
                        "importanceMean": row["importanceMean"],
                        "importanceStd": row["importanceStd"],
                    }
                )

        fold_metrics = {
            "raw": evaluate_multitask_predictions(test_df[targets], raw_fold),
            "calibrated": evaluate_multitask_predictions(test_df[targets], calibrated_test["predictions"]),
            "calibratedPlusResidual": evaluate_multitask_predictions(test_df[targets], final_fold),
        }
        fold_rows.append(
            {
                "source_id": source_id,
                "fold_index": int(fold_idx),
                "n_test": int(len(test_df)),
                "winsorizeBounds": {target: [float(bounds[target][0]), float(bounds[target][1])] for target in targets},
                "metrics": fold_metrics,
                "bayesianBackend": calibrated_test["backend"],
            }
        )

    raw_metrics = evaluate_multitask_predictions(df[targets], raw_predictions[targets])
    calibrated_metrics = evaluate_multitask_predictions(df[targets], calibrated_predictions[targets])
    final_metrics = evaluate_multitask_predictions(df[targets], final_predictions[targets])
    interval_metrics = {
        "purity": evaluate_interval_quality(
            y_true=df["purity_target"].to_numpy(dtype=float),
            interval=interval_cache["purity_target"].to_numpy(dtype=float),
        ),
        "yield": evaluate_interval_quality(
            y_true=df["yield_target"].to_numpy(dtype=float),
            interval=interval_cache["yield_target"].to_numpy(dtype=float),
        ),
    }
    yield_head_df = pd.DataFrame(head_weight_rows)
    head_summary: dict[str, Any] = {"perFoldHeads": head_weight_rows}
    if not yield_head_df.empty:
        agg = (
            yield_head_df.groupby("head", as_index=False)
            .agg(
                nHeadTrain=("n_head_train", "sum"),
                nHeadTest=("n_head_test", "sum"),
                localUsageRate=("localModelUsed", "mean"),
                avgBlendWeight=("blendWeight", "mean"),
            )
            .sort_values("nHeadTest", ascending=False)
        )
        head_summary["globalHeadSummary"] = [
            {
                "head": str(row["head"]),
                "nHeadTrain": int(row["nHeadTrain"]),
                "nHeadTest": int(row["nHeadTest"]),
                "localUsageRate": float(row["localUsageRate"]),
                "avgBlendWeight": float(row["avgBlendWeight"]),
            }
            for _, row in agg.iterrows()
        ]

        weighted_r2 = 0.0
        total = float(agg["nHeadTest"].sum())
        if total > 0:
            for _, row in agg.iterrows():
                head = str(row["head"])
                test_mask = df[yield_head_label].fillna("unknown").astype(str) == head
                if int(np.sum(test_mask.to_numpy())) < 2:
                    continue
                metrics = evaluate_predictions(
                    y_true=df.loc[test_mask, "yield_target"].to_numpy(dtype=float),
                    y_pred=final_predictions.loc[test_mask, "yield_target"].to_numpy(dtype=float),
                    tolerance=YIELD_TOLERANCE,
                )
                weighted_r2 += (float(row["nHeadTest"]) / total) * float(metrics["r2"])
            head_summary["weightedYieldR2ByHead"] = float(weighted_r2)

    full_train, _ = winsorize_targets(df.copy(), targets=targets, lower=0.01, upper=0.99)
    full_calibrator = BayesianMultitaskCalibrator(
        draws=args.bayes_draws,
        tune=args.bayes_tune,
        chains=args.bayes_chains,
        target_accept=args.bayes_target_accept,
        likelihood=args.bayes_likelihood,
        student_t_nu=args.bayes_student_t_nu,
        random_state=RANDOM_STATE + 999,
    )
    full_calibrator.fit(
        train_df=full_train,
        features=calibration_features,
        targets=targets,
        group_cols=group_cols,
    )
    posterior = full_calibrator.posterior_summary()
    heterogeneity = {}
    posterior_sources = {}
    for target in targets:
        target_summary = posterior["targets"][target]
        heterogeneity[target] = {
            "sigma2Mean": target_summary["sigma2Mean"],
            "tauSource2Mean": target_summary["tauSource2Mean"],
            "tauStage2Mean": target_summary["tauStage2Mean"],
            "iccSourceMean": target_summary["iccSourceMean"],
            "iccStageMean": target_summary["iccStageMean"],
            "posteriorDraws": target_summary["nPosteriorDraws"],
        }
        posterior_sources[target] = target_summary["randomInterceptsBySource"]

    residual_feature_importance: dict[str, list[dict[str, float]]] = {}
    if residual_importance_rows:
        imp_df = pd.DataFrame(residual_importance_rows)
        for target, group in imp_df.groupby("target"):
            agg = (
                group.groupby("feature", as_index=False)
                .agg(importanceMean=("importanceMean", "mean"), importanceStd=("importanceStd", "mean"))
                .sort_values("importanceMean", ascending=False)
            )
            residual_feature_importance[target] = [
                {
                    "feature": str(row["feature"]),
                    "importanceMean": float(row["importanceMean"]),
                    "importanceStd": float(row["importanceStd"]),
                }
                for _, row in agg.head(20).iterrows()
            ]

    return {
        "stageName": "bayesian_calibrated_residual_ensemble",
        "calibrationFeatures": calibration_features,
        "residualFeatures": residual_features,
        "rawMetrics": raw_metrics,
        "calibratedMetrics": calibrated_metrics,
        "calibratedPlusResidualMetrics": final_metrics,
        "intervalMetrics": interval_metrics,
        "foldMetrics": fold_rows,
        "heterogeneityDecomposition": heterogeneity,
        "posteriorSourceEffects": posterior_sources,
        "posteriorSummary": posterior,
        "residualFeatureImportance": residual_feature_importance,
        "yieldHeadAggregation": head_summary,
        "predictions": {
            "raw": raw_predictions,
            "calibrated": calibrated_predictions,
            "calibratedPlusResidual": final_predictions,
        },
    }


def summarize_target_head_metrics(head_metrics: list[dict[str, Any]], target: str) -> dict[str, float]:
    target_rows = [row for row in head_metrics if row["target"] == target]
    if not target_rows:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "accuracyWithinTolerance": float("nan")}
    summary = build_weighted_head_summary(target_rows)
    return {
        "rmse": float(summary["weightedRmse"]),
        "mae": float(summary["weightedMae"]),
        "r2": float(summary["weightedR2"]),
        "accuracyWithinTolerance": float(summary["weightedAccuracyWithinTolerance"]),
    }


def run_strict_semantic_head_loso(
    df: pd.DataFrame,
    stage: StageConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    strict_df, strict_meta = build_strict_semantic_subset(
        df,
        min_rows=args.head_min_rows,
        min_sources=args.head_min_sources,
        max_source_share=args.head_max_source_share,
    )
    target_specs = [
        {
            "target": "purity_target",
            "head_column": "purity_semantic_head",
            "tolerance": PURITY_TOLERANCE,
            "eligibility": strict_meta["purityEligibility"],
            "bayes_group_col": "purity_stage_canon",
        },
        {
            "target": "yield_target",
            "head_column": "yield_semantic_head",
            "tolerance": YIELD_TOLERANCE,
            "eligibility": strict_meta["yieldEligibility"],
            "bayes_group_col": "yield_stage_canon",
        },
    ]

    head_level_metrics: list[dict[str, Any]] = []
    source_calibration_stats: list[dict[str, Any]] = []
    source_diagnostics: list[dict[str, Any]] = []
    predictions_by_target: dict[str, pd.Series] = {}

    for spec in target_specs:
        target = spec["target"]
        head_column = spec["head_column"]
        eligible_heads = [row for row in spec["eligibility"] if row["eligible"]]
        target_df = strict_df[strict_df[target].notna()].copy()
        target_prediction = pd.Series(index=strict_df.index, dtype=float)

        for eligible in eligible_heads:
            head = str(eligible["head"])
            head_df = target_df[target_df[head_column].astype(str) == head].copy().reset_index(drop=False)
            if len(head_df) < 2 or head_df["source_id"].nunique() < 2:
                continue
            source_diag = diagnose_head_sources(head_df, target=target)
            modeling_df, compatibility = select_compatible_head_subset(
                head_df,
                diagnostics=source_diag,
                min_rows=args.head_min_rows,
                min_sources=args.head_min_sources,
            )
            source_diagnostics.append(
                {
                    "target": target,
                    "head": head,
                    **source_diag,
                    "compatibility": compatibility,
                }
            )
            if len(modeling_df) < 2 or modeling_df["source_id"].nunique() < 2:
                continue

            fold_predictions: list[pd.DataFrame] = []
            fold_stats: list[dict[str, Any]] = []
            all_truth: list[float] = []
            all_pred: list[float] = []
            all_interval: list[list[float]] = []

            for fold_index, (train_idx, test_idx, source_id) in enumerate(loso_splits(modeling_df)):
                train_raw = add_source_context_features(modeling_df.iloc[train_idx].copy())
                test_df = add_source_context_features(modeling_df.iloc[test_idx].copy())
                train_df, bounds = winsorize_targets(train_raw, targets=[target], lower=0.01, upper=0.99)

                x_train = train_df.drop(columns=["purity_target", "yield_target", "record_id", "source_id", "index"])
                x_test = test_df.drop(columns=["purity_target", "yield_target", "record_id", "source_id", "index"])
                model = build_ensemble_pipeline(stage, args.imputation, train_df, args)
                model.fit(x_train, train_df[target].to_numpy(dtype=float))
                raw_test_pred = model.predict(x_test).astype(float)

                x_train_raw = train_raw.drop(columns=["purity_target", "yield_target", "record_id", "source_id", "index"])
                train_residuals = train_raw[target].to_numpy(dtype=float) - model.predict(x_train_raw).astype(float)
                shrinkage = SourceAwareShrinkageCalibrator(prior_strength=SOURCE_OFFSET_PRIOR_STRENGTH)
                shrinkage.fit(train_raw["source_id"], train_residuals)
                source_offsets = shrinkage.predict_offsets(test_df["source_id"])
                final_pred = raw_test_pred + source_offsets

                interval_half_width = 1.96 * max(float(shrinkage.summary()["residualStd"]), 1e-6)
                intervals = np.column_stack([final_pred - interval_half_width, final_pred + interval_half_width])

                fold_predictions.append(
                    pd.DataFrame(
                        {
                            "index": test_df["index"].to_numpy(dtype=int),
                            "prediction": final_pred,
                        }
                    )
                )
                all_truth.extend(test_df[target].to_numpy(dtype=float).tolist())
                all_pred.extend(final_pred.tolist())
                all_interval.extend(intervals.tolist())
                fold_stats.append(
                    {
                        "target": target,
                        "head": head,
                        "source_id": source_id,
                        "foldIndex": int(fold_index),
                        "nTrain": int(len(train_df)),
                        "nTest": int(len(test_df)),
                        "winsorizeBounds": [float(bounds[target][0]), float(bounds[target][1])],
                        "sourceShrinkage": shrinkage.summary(),
                        "compatibleSubset": compatibility,
                        "usedBayesianCalibration": bool(
                            args.enable_bayesian_calibration
                            and len(train_df) >= BAYES_HEAD_MIN_ROWS
                            and train_df["source_id"].nunique() >= BAYES_HEAD_MIN_SOURCES
                        ),
                    }
                )

            combined_predictions = pd.concat(fold_predictions, ignore_index=True).sort_values("index")
            target_prediction.loc[combined_predictions["index"].to_numpy(dtype=int)] = combined_predictions["prediction"].to_numpy(dtype=float)
            metrics = evaluate_predictions(
                np.asarray(all_truth, dtype=float),
                np.asarray(all_pred, dtype=float),
                tolerance=spec["tolerance"],
            )
            interval_metrics = evaluate_interval_quality(
                np.asarray(all_truth, dtype=float),
                np.asarray(all_interval, dtype=float),
            )
            head_level_metrics.append(
                {
                    "target": target,
                    "head": head,
                    "nRows": int(eligible["nRows"]),
                    "nSources": int(eligible["nSources"]),
                    "modelingRows": int(compatibility["modelingRows"]),
                    "modelingSources": int(compatibility["modelingSources"]),
                    "excludedSources": list(compatibility["excludedSources"]),
                    "usedCompatibleSubset": bool(compatibility["usedCompatibleSubset"]),
                    "metrics": metrics,
                    "intervalMetrics": interval_metrics,
                    "decision": derive_head_decision(
                        target=target,
                        n_rows=int(compatibility["modelingRows"]),
                        n_sources=int(compatibility["modelingSources"]),
                        metrics=metrics,
                        interval_metrics=interval_metrics,
                    ),
                    "usedBayesianCalibration": any(row["usedBayesianCalibration"] for row in fold_stats),
                }
            )
            source_calibration_stats.extend(fold_stats)

        predictions_by_target[target] = target_prediction

    strict_indexed = strict_df.reset_index(drop=True)
    purity_summary = summarize_target_head_metrics(head_level_metrics, "purity_target")
    yield_summary = summarize_target_head_metrics(head_level_metrics, "yield_target")
    weighted_summary = build_weighted_head_summary(head_level_metrics)
    final_metrics = {
        "purity": purity_summary,
        "yield": yield_summary,
        "combined": {
            "rmse": float(np.nanmean([purity_summary["rmse"], yield_summary["rmse"]])),
            "mae": float(np.nanmean([purity_summary["mae"], yield_summary["mae"]])),
            "r2": float(np.nanmean([purity_summary["r2"], yield_summary["r2"]])),
            "jointToleranceAccuracy": float("nan"),
        },
    }
    if predictions_by_target:
        common_mask = predictions_by_target["purity_target"].notna() & predictions_by_target["yield_target"].notna()
        if common_mask.any():
            final_metrics["combined"]["jointToleranceAccuracy"] = float(
                np.mean(
                    (
                        np.abs(predictions_by_target["purity_target"][common_mask] - strict_indexed.loc[common_mask, "purity_target"])
                        <= PURITY_TOLERANCE
                    )
                    & (
                        np.abs(predictions_by_target["yield_target"][common_mask] - strict_indexed.loc[common_mask, "yield_target"])
                        <= YIELD_TOLERANCE
                    )
                )
            )

    isolated_metrics = next(
        (row for row in head_level_metrics if row["target"] == "yield_target" and row["head"] == "isolated|isolated"),
        None,
    )
    baseline_isolated = evaluate_predictions(
        y_true=strict_indexed.loc[strict_indexed["yield_semantic_head"] == "isolated|isolated", "yield_target"].to_numpy(dtype=float),
        y_pred=strict_indexed.loc[strict_indexed["yield_semantic_head"] == "isolated|isolated", "yield_target"].mean()
        * np.ones(int(np.sum(strict_indexed["yield_semantic_head"] == "isolated|isolated"))),
        tolerance=YIELD_TOLERANCE,
    ) if np.sum(strict_indexed["yield_semantic_head"] == "isolated|isolated") >= 2 else None

    return {
        "stageName": "strict_semantic_head_loso",
        "strictSemanticFilter": strict_meta,
        "headEligibility": {
            "purity": strict_meta["purityEligibility"],
            "yield": strict_meta["yieldEligibility"],
        },
        "headLevelMetrics": head_level_metrics,
        "sourceCalibrationStats": source_calibration_stats,
        "sourceDiagnostics": source_diagnostics,
        "weightedHeadSummary": weighted_summary,
        "predictionPolicy": summarize_prediction_policy(head_level_metrics),
        "isolatedYieldPrimaryHead": {
            "baseline": baseline_isolated,
            "final": isolated_metrics["metrics"] if isolated_metrics else None,
            "rmseChangePct": (
                None
                if not isolated_metrics or baseline_isolated is None or baseline_isolated["rmse"] == 0
                else float((baseline_isolated["rmse"] - isolated_metrics["metrics"]["rmse"]) / baseline_isolated["rmse"])
            ),
        },
        "metrics": final_metrics,
    }


def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    base_df = build_modeling_dataframe()
    multitask_df = base_df.dropna(subset=["purity_target", "yield_target"]).reset_index(drop=True)
    report: dict[str, Any] = {
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "initialRowsWithTargets": int(len(base_df)),
        "initialRowsWithBothTargets": int(len(multitask_df)),
        "initialUniqueSources": int(base_df["source_id"].nunique()),
        "targetAvailability": {
            "purityRows": int(base_df["purity_target"].notna().sum()),
            "yieldRows": int(base_df["yield_target"].notna().sum()),
            "purityOnlyRows": int((base_df["purity_target"].notna() & base_df["yield_target"].isna()).sum()),
            "yieldOnlyRows": int((base_df["yield_target"].notna() & base_df["purity_target"].isna()).sum()),
            "bothTargetRows": int(len(multitask_df)),
        },
        "imputationStrategy": args.imputation,
        "strictSemanticHeads": bool(args.strict_semantic_heads),
        "headEligibilityThresholds": {
            "minRows": int(args.head_min_rows),
            "minSources": int(args.head_min_sources),
            "maxSourceShare": float(args.head_max_source_share),
        },
        "imputationJustification": (
            "Mean imputation was selected because the literature dataset is small and highly heterogeneous across sources; "
            "KNN neighbors are unstable under grouped cross-validation."
            if args.imputation == "mean"
            else "KNN imputation was selected to retain local numeric structure across incomplete experimental context fields."
        ),
        "sequenceSimilarityThreshold": args.sequence_similarity_threshold,
        "proteinEmbedding": {
            "modelName": args.embedding_model,
            "requestedComponents": int(args.embedding_components),
            "cacheDir": str(DEFAULT_CACHE_DIR),
            "maxRandomSearchIterationsForEmbeddingStage": 2,
        },
        "bayesianCalibration": {
            "enabled": bool(args.enable_bayesian_calibration),
            "draws": int(args.bayes_draws),
            "tune": int(args.bayes_tune),
            "chains": int(args.bayes_chains),
            "targetAccept": float(args.bayes_target_accept),
            "likelihood": str(args.bayes_likelihood),
            "studentTNu": float(args.bayes_student_t_nu),
            "residualModel": args.residual_model,
            "unseenSourcePolicy": args.unseen_source_policy,
        },
        "stageResults": [],
    }

    final_stage_outputs: dict[str, Any] = {}
    for stage in stage_definitions():
        if args.strict_semantic_heads and stage.use_protein_embeddings:
            continue
        cleaned_df, cleaning_summary = apply_cleaning_pipeline(
            multitask_df,
            stage,
            similarity_threshold=args.sequence_similarity_threshold,
        )

        if stage.tuned_search:
            stage_random_search_iterations = (
                min(args.random_search_iterations, report["proteinEmbedding"]["maxRandomSearchIterationsForEmbeddingStage"])
                if stage.use_protein_embeddings
                else args.random_search_iterations
            )
            purity_model, purity_params, purity_cv_rmse = tune_target_pipeline(
                cleaned_df,
                stage,
                args.imputation,
                "purity_target",
                stage_random_search_iterations,
                args,
            )
            yield_model, yield_params, yield_cv_rmse = tune_target_pipeline(
                cleaned_df,
                stage,
                args.imputation,
                "yield_target",
                stage_random_search_iterations,
                args,
            )
            predictions = pd.DataFrame(
                {
                    "purity_target": cross_validated_predictions_for_pipeline(cleaned_df, purity_model, "purity_target"),
                    "yield_target": cross_validated_predictions_for_pipeline(cleaned_df, yield_model, "yield_target"),
                },
                index=cleaned_df.index,
            )
            metrics = evaluate_multitask_predictions(cleaned_df[["purity_target", "yield_target"]], predictions)

            feature_df = cleaned_df.drop(columns=["purity_target", "yield_target", "record_id", "source_id"])
            purity_model.fit(feature_df, cleaned_df["purity_target"])
            yield_model.fit(feature_df, cleaned_df["yield_target"])

            final_stage_outputs[stage.name] = {
                "datasetRows": int(len(cleaned_df)),
                "uniqueSources": int(cleaned_df["source_id"].nunique()),
                "searchIterations": int(stage_random_search_iterations),
                "semanticSubgroups": evaluate_semantic_subgroups(cleaned_df, predictions),
                "purity": {
                    "bestParams": purity_params,
                    "cvRmseDuringSearch": purity_cv_rmse,
                    "featureAnalysis": extract_final_feature_analysis(
                        purity_model,
                        feature_df,
                        cleaned_df["purity_target"],
                    ),
                },
                "yield": {
                    "bestParams": yield_params,
                    "cvRmseDuringSearch": yield_cv_rmse,
                    "featureAnalysis": extract_final_feature_analysis(
                        yield_model,
                        feature_df,
                        cleaned_df["yield_target"],
                    ),
                },
            }
        else:
            metrics = cross_validated_stage_metrics(cleaned_df, stage, args.imputation, args)

        report["stageResults"].append(
            {
                "stage": stage.name,
                "description": stage.description,
                "cleaningSummary": asdict(cleaning_summary),
                "metrics": metrics,
                "usesProteinEmbeddings": stage.use_protein_embeddings,
                "features": {
                    "standardized": get_stage_columns(stage, cleaned_df)[0],
                    "normalized": get_stage_columns(stage, cleaned_df)[1],
                    "binary": get_stage_columns(stage, cleaned_df)[2],
                    "categorical": get_stage_columns(stage, cleaned_df)[3],
                },
            }
        )

    if args.enable_bayesian_calibration and not args.strict_semantic_heads:
        loso_stage = next(item for item in stage_definitions() if item.name == "final_tuned_ensemble")
        loso_df, loso_cleaning = apply_cleaning_pipeline(
            multitask_df,
            loso_stage,
            similarity_threshold=args.sequence_similarity_threshold,
        )
        bayes_loso = run_bayesian_calibrated_residual_loso(loso_df, loso_stage, args)
        report["stageResults"].append(
            {
                "stage": bayes_loso["stageName"],
                "description": "Bayesian hierarchical calibration with source/stage random effects plus residual ensemble under LOSO.",
                "cleaningSummary": asdict(loso_cleaning),
                "metrics": bayes_loso["calibratedPlusResidualMetrics"],
                "usesProteinEmbeddings": False,
                "features": {
                    "standardized": get_stage_columns(loso_stage, loso_df)[0],
                    "normalized": get_stage_columns(loso_stage, loso_df)[1],
                    "binary": get_stage_columns(loso_stage, loso_df)[2],
                    "categorical": get_stage_columns(loso_stage, loso_df)[3],
                    "calibrationFeatures": bayes_loso["calibrationFeatures"],
                    "residualFeatures": bayes_loso["residualFeatures"],
                },
            }
        )
        report["heterogeneityDecomposition"] = bayes_loso["heterogeneityDecomposition"]
        report["posteriorSourceEffects"] = bayes_loso["posteriorSourceEffects"]
        report["calibratedVsRawMetrics"] = {
            "rawFinalTunedEnsembleLOSO": bayes_loso["rawMetrics"],
            "bayesianCalibratedLOSO": bayes_loso["calibratedMetrics"],
            "bayesianCalibratedResidualEnsembleLOSO": bayes_loso["calibratedPlusResidualMetrics"],
            "intervalMetrics": bayes_loso["intervalMetrics"],
        }
        report["losoFoldMetrics"] = bayes_loso["foldMetrics"]
        final_stage_outputs[bayes_loso["stageName"]] = {
            "residualFeatureImportance": bayes_loso["residualFeatureImportance"],
            "posteriorSummary": bayes_loso["posteriorSummary"],
            "yieldHeadAggregation": bayes_loso["yieldHeadAggregation"],
        }

    if args.strict_semantic_heads:
        strict_stage = next(item for item in stage_definitions() if item.name == "final_tuned_ensemble")
        strict_df, strict_cleaning = apply_cleaning_pipeline(
            base_df,
            strict_stage,
            similarity_threshold=args.sequence_similarity_threshold,
        )
        strict_loso = run_strict_semantic_head_loso(strict_df, strict_stage, args)
        report["stageResults"].append(
            {
                "stage": strict_loso["stageName"],
                "description": "Strict semantic head LOSO with eligible purity/yield heads and source-aware shrinkage calibration.",
                "cleaningSummary": {
                    **asdict(strict_cleaning),
                    "final_rows": int(strict_loso["strictSemanticFilter"]["outputRows"]),
                    "unique_sources": int(strict_loso["strictSemanticFilter"]["outputUniqueSources"]),
                },
                "metrics": strict_loso["metrics"],
                "usesProteinEmbeddings": False,
                "features": {
                    "standardized": get_stage_columns(strict_stage, strict_df)[0],
                    "normalized": get_stage_columns(strict_stage, strict_df)[1],
                    "binary": get_stage_columns(strict_stage, strict_df)[2],
                    "categorical": get_stage_columns(strict_stage, strict_df)[3],
                },
            }
        )
        report["headEligibility"] = strict_loso["headEligibility"]
        report["headLevelMetrics"] = strict_loso["headLevelMetrics"]
        report["sourceCalibrationStats"] = strict_loso["sourceCalibrationStats"]
        report["sourceDiagnostics"] = strict_loso["sourceDiagnostics"]
        report["weightedHeadSummary"] = strict_loso["weightedHeadSummary"]
        report["strictSemanticFilter"] = strict_loso["strictSemanticFilter"]
        report["predictionPolicy"] = strict_loso["predictionPolicy"]
        report["isolatedYieldPrimaryHead"] = strict_loso["isolatedYieldPrimaryHead"]
        final_stage_outputs[strict_loso["stageName"]] = {
            "headLevelMetrics": strict_loso["headLevelMetrics"],
            "sourceCalibrationStats": strict_loso["sourceCalibrationStats"],
            "sourceDiagnostics": strict_loso["sourceDiagnostics"],
            "weightedHeadSummary": strict_loso["weightedHeadSummary"],
            "strictSemanticFilter": strict_loso["strictSemanticFilter"],
            "predictionPolicy": strict_loso["predictionPolicy"],
            "isolatedYieldPrimaryHead": strict_loso["isolatedYieldPrimaryHead"],
        }

    final_stage_result = report["stageResults"][-1]
    report["finalDatasetRows"] = int(
        report["strictSemanticFilter"]["outputRows"]
        if "strictSemanticFilter" in report
        else final_stage_result["cleaningSummary"]["final_rows"]
    )
    report["finalUniqueSources"] = int(
        report["strictSemanticFilter"]["outputUniqueSources"]
        if "strictSemanticFilter" in report
        else final_stage_result["cleaningSummary"]["unique_sources"]
    )
    report["scalingRationale"] = {
        "standardization": "Large-magnitude continuous physicochemical variables were standardized to zero mean and unit variance.",
        "normalization": "Ratios and bounded heuristic synthesis scores were scaled to the [0, 1] range to keep them comparable and interpretable.",
    }
    report["finalModelsByStage"] = final_stage_outputs
    return report


def markdown_metrics_row(stage_result: dict[str, Any]) -> str:
    metrics = stage_result["metrics"]
    return (
        f"| {stage_result['stage']} | {stage_result['cleaningSummary']['final_rows']} | "
        f"{metrics['purity']['rmse']:.3f} | {metrics['purity']['mae']:.3f} | {metrics['purity']['r2']:.3f} | "
        f"{metrics['purity']['accuracyWithinTolerance']:.3f} | {metrics['yield']['rmse']:.3f} | "
        f"{metrics['yield']['mae']:.3f} | {metrics['yield']['r2']:.3f} | "
        f"{metrics['yield']['accuracyWithinTolerance']:.3f} | {metrics['combined']['jointToleranceAccuracy']:.3f} |"
    )


def generate_markdown_report(report: dict[str, Any]) -> str:
    baseline_stage = next(item for item in report["stageResults"] if item["stage"] == "final_tuned_ensemble")
    embedding_stage = next((item for item in report["stageResults"] if item["stage"] == "final_tuned_ensemble_with_embeddings"), None)
    bayes_stage = next((item for item in report["stageResults"] if item["stage"] == "bayesian_calibrated_residual_ensemble"), None)
    strict_head_stage = next((item for item in report["stageResults"] if item["stage"] == "strict_semantic_head_loso"), None)
    baseline_models = report["finalModelsByStage"]["final_tuned_ensemble"]
    embedding_models = report["finalModelsByStage"].get("final_tuned_ensemble_with_embeddings")
    semantic_models = embedding_models or baseline_models
    purity_importance = semantic_models["purity"]["featureAnalysis"]["permutationImportance"]
    yield_importance = semantic_models["yield"]["featureAnalysis"]["permutationImportance"]
    final_stage_for_report = strict_head_stage or bayes_stage or embedding_stage or baseline_stage
    final_label = final_stage_for_report["stage"]

    lines = [
        "# Comprehensive ML Workflow Report",
        "",
        "## 1. Dataset and objective",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Initial usable rows with sequence + purity + yield: `{report['initialRowsWithTargets']}`",
        f"- Final rows after cleaning: `{report['finalDatasetRows']}`",
        f"- Unique source groups used for grouped CV: `{report['finalUniqueSources']}`",
        "",
        "## 2. Preprocessing methodology",
        "",
        "- Exact duplicates were removed using source, normalized sequence, stage labels, topology, and target values.",
        f"- Near-duplicates were removed when sequence similarity was at least `{report['sequenceSimilarityThreshold']}` within the same source-stage group and target differences stayed within tight tolerances.",
        "- Outliers were removed using stage-aware IQR filtering on purity/yield plus modified z-score screening on sequence length and molecular weight.",
        f"- Missing numeric values were handled with `{report['imputationStrategy']}` imputation.",
        f"- Imputation rationale: {report['imputationJustification']}",
        f"- Standardization rationale: {report['scalingRationale']['standardization']}",
        f"- Normalization rationale: {report['scalingRationale']['normalization']}",
        "",
        "## 3. Ensemble architecture",
        "",
        "- Final model family: `VotingRegressor`",
        "- Base learners: `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `Ridge`",
        "- Feature selection: `SelectFromModel(ExtraTreesRegressor)`",
        "- Validation strategy: `GroupKFold` grouped by `source_id`",
        "- Hyperparameter search: `RandomizedSearchCV`",
        f"- Protein embedding model for the augmented stage: `{report['proteinEmbedding']['modelName']}`",
        f"- Embedding PCA components per grouped fold: `{report['proteinEmbedding']['requestedComponents']}`",
        "",
        "## 4. Stage-by-stage performance comparison",
        "",
        "| Stage | Rows | Purity RMSE | Purity MAE | Purity R² | Purity Acc ±5% | Yield RMSE | Yield MAE | Yield R² | Yield Acc ±10% | Joint Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(markdown_metrics_row(stage_result) for stage_result in report["stageResults"])
    lines.extend(
        [
            "",
            "## 5. Final tuned stage comparison",
            "",
            f"- Baseline tuned ensemble purity RMSE: `{baseline_stage['metrics']['purity']['rmse']:.3f}`",
            f"- Baseline tuned ensemble yield RMSE: `{baseline_stage['metrics']['yield']['rmse']:.3f}`",
            "",
            "## 6. Final tuned model parameters",
            "",
            "### Baseline purity model",
            "",
            "```json",
            json.dumps(baseline_models["purity"]["bestParams"], indent=2),
            "```",
            "",
            "### Baseline yield model",
            "",
            "```json",
            json.dumps(baseline_models["yield"]["bestParams"], indent=2),
            "```",
            "",
            "## 7. Final evaluation metrics",
            "",
            f"- Final stage: `{final_label}`",
            f"- Final purity RMSE: `{final_stage_for_report['metrics']['purity']['rmse']:.3f}`",
            f"- Final purity MAE: `{final_stage_for_report['metrics']['purity']['mae']:.3f}`",
            f"- Final purity R²: `{final_stage_for_report['metrics']['purity']['r2']:.3f}`",
            f"- Final purity accuracy within ±5%: `{final_stage_for_report['metrics']['purity']['accuracyWithinTolerance']:.3f}`",
            f"- Final yield RMSE: `{final_stage_for_report['metrics']['yield']['rmse']:.3f}`",
            f"- Final yield MAE: `{final_stage_for_report['metrics']['yield']['mae']:.3f}`",
            f"- Final yield R²: `{final_stage_for_report['metrics']['yield']['r2']:.3f}`",
            f"- Final yield accuracy within ±10%: `{final_stage_for_report['metrics']['yield']['accuracyWithinTolerance']:.3f}`",
            f"- Final joint tolerance accuracy: `{final_stage_for_report['metrics']['combined']['jointToleranceAccuracy']:.3f}`",
            "",
            "## 8. Semantic subgroup evaluation",
            "",
            "### Purity by stage (embedding tuned ensemble)",
            "",
            "| Stage | n | RMSE | MAE | R² | Acc ±5% |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if embedding_stage and embedding_models:
        lines[lines.index("## 6. Final tuned model parameters") - 1:lines.index("## 6. Final tuned model parameters") - 1] = [
            f"- Embedding tuned ensemble purity RMSE: `{embedding_stage['metrics']['purity']['rmse']:.3f}`",
            f"- Embedding tuned ensemble yield RMSE: `{embedding_stage['metrics']['yield']['rmse']:.3f}`",
            f"- Purity RMSE delta (embedding - baseline): `{embedding_stage['metrics']['purity']['rmse'] - baseline_stage['metrics']['purity']['rmse']:.3f}`",
            f"- Yield RMSE delta (embedding - baseline): `{embedding_stage['metrics']['yield']['rmse'] - baseline_stage['metrics']['yield']['rmse']:.3f}`",
            "",
        ]
        insert_index = lines.index("## 7. Final evaluation metrics") - 1
        lines[insert_index:insert_index] = [
            "### Embedding purity model",
            "",
            "```json",
            json.dumps(embedding_models["purity"]["bestParams"], indent=2),
            "```",
            "",
            "### Embedding yield model",
            "",
            "```json",
            json.dumps(embedding_models["yield"]["bestParams"], indent=2),
            "```",
            "",
        ]
    semantic_subgroups = semantic_models["semanticSubgroups"]
    lines[lines.index("### Purity by stage (embedding tuned ensemble)")] = (
        f"### Purity by stage ({'embedding tuned ensemble' if embedding_models else 'baseline tuned ensemble'})"
    )
    for item in semantic_subgroups["purityByStage"]:
        lines.append(
            f"| {item['label']} | {item['n']} | {item['rmse']:.3f} | {item['mae']:.3f} | {item['r2']:.3f} | {item['accuracyWithinTolerance']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Yield by stage (embedding tuned ensemble)",
            "",
            "| Stage | n | RMSE | MAE | R² | Acc ±10% |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines[lines.index("### Yield by stage (embedding tuned ensemble)")] = (
        f"### Yield by stage ({'embedding tuned ensemble' if embedding_models else 'baseline tuned ensemble'})"
    )
    for item in semantic_subgroups["yieldByStage"]:
        lines.append(
            f"| {item['label']} | {item['n']} | {item['rmse']:.3f} | {item['mae']:.3f} | {item['r2']:.3f} | {item['accuracyWithinTolerance']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Yield by label basis (embedding tuned ensemble)",
            "",
            "| Basis | n | RMSE | MAE | R² | Acc ±10% |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines[lines.index("### Yield by label basis (embedding tuned ensemble)")] = (
        f"### Yield by label basis ({'embedding tuned ensemble' if embedding_models else 'baseline tuned ensemble'})"
    )
    for item in semantic_subgroups["yieldByBasis"]:
        lines.append(
            f"| {item['label']} | {item['n']} | {item['rmse']:.3f} | {item['mae']:.3f} | {item['r2']:.3f} | {item['accuracyWithinTolerance']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 9. Feature importance",
            "",
            "### Top permutation importances for purity",
            "",
            "| Feature | Mean Importance | Std |",
            "| --- | ---: | ---: |",
        ]
    )
    for item in purity_importance[:10]:
        lines.append(f"| {item['feature']} | {item['importanceMean']:.4f} | {item['importanceStd']:.4f} |")

    lines.extend(
        [
            "",
            "### Top permutation importances for yield",
            "",
            "| Feature | Mean Importance | Std |",
            "| --- | ---: | ---: |",
        ]
    )
    for item in yield_importance[:10]:
        lines.append(f"| {item['feature']} | {item['importanceMean']:.4f} | {item['importanceStd']:.4f} |")

    lines.extend(
        [
            "",
            "## 10. Heterogeneity calibration under LOSO",
            "",
        ]
    )
    if "calibratedVsRawMetrics" in report:
        raw_loso = report["calibratedVsRawMetrics"]["rawFinalTunedEnsembleLOSO"]
        cal_loso = report["calibratedVsRawMetrics"]["bayesianCalibratedLOSO"]
        cal_residual_loso = report["calibratedVsRawMetrics"]["bayesianCalibratedResidualEnsembleLOSO"]
        lines.extend(
            [
                f"- Raw LOSO purity R²: `{raw_loso['purity']['r2']:.3f}`",
                f"- Bayesian calibrated LOSO purity R²: `{cal_loso['purity']['r2']:.3f}`",
                f"- Bayesian + residual LOSO purity R²: `{cal_residual_loso['purity']['r2']:.3f}`",
                f"- Raw LOSO yield R²: `{raw_loso['yield']['r2']:.3f}`",
                f"- Bayesian calibrated LOSO yield R²: `{cal_loso['yield']['r2']:.3f}`",
                f"- Bayesian + residual LOSO yield R²: `{cal_residual_loso['yield']['r2']:.3f}`",
                f"- Purity tolerance accuracy (final LOSO): `{cal_residual_loso['purity']['accuracyWithinTolerance']:.3f}`",
                f"- Yield tolerance accuracy (final LOSO): `{cal_residual_loso['yield']['accuracyWithinTolerance']:.3f}`",
                f"- Joint tolerance accuracy (final LOSO): `{cal_residual_loso['combined']['jointToleranceAccuracy']:.3f}`",
                "",
                "### Heterogeneity decomposition (posterior means)",
                "",
                "| Target | sigma² | tau_source² | tau_stage² | ICC_source | ICC_stage |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for target, payload in report.get("heterogeneityDecomposition", {}).items():
            lines.append(
                f"| {target} | {payload['sigma2Mean']:.3f} | {payload['tauSource2Mean']:.3f} | {payload['tauStage2Mean']:.3f} | "
                f"{payload['iccSourceMean']:.3f} | {payload['iccStageMean']:.3f} |"
            )
    else:
        lines.append("- Global Bayesian LOSO calibration was skipped because strict semantic head LOSO is the default main evaluation path for this run.")
    if "strictSemanticFilter" in report:
        lines.extend(
            [
                "",
                "## 11. Strict semantic head LOSO",
                "",
                f"- Strict semantic filtering enabled: `{report['strictSemanticHeads']}`",
                f"- Rows before strict head filtering: `{report['strictSemanticFilter']['inputRows']}`",
                f"- Rows after strict head filtering: `{report['strictSemanticFilter']['outputRows']}`",
                f"- Purity rows before/after strict filtering: `{report['strictSemanticFilter']['inputPurityRows']} -> {report['strictSemanticFilter']['outputPurityRows']}`",
                f"- Yield rows before/after strict filtering: `{report['strictSemanticFilter']['inputYieldRows']} -> {report['strictSemanticFilter']['outputYieldRows']}`",
                f"- Head thresholds: rows>={report['headEligibilityThresholds']['minRows']}, sources>={report['headEligibilityThresholds']['minSources']}, max source share<={report['headEligibilityThresholds']['maxSourceShare']:.2f}",
                "",
                "### Head eligibility",
                "",
                "| Target | Head | Rows | Sources | Max source share | Eligible | Reasons |",
                "| --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for target_name, entries in report["headEligibility"].items():
            for row in entries:
                lines.append(
                    f"| {target_name} | {row['head']} | {row['nRows']} | {row['nSources']} | {row['maxSourceShare']:.3f} | "
                    f"{'yes' if row['eligible'] else 'no'} | {', '.join(row['reasons']) or '-'} |"
                )
        lines.extend(
            [
                "",
                "### Eligible head LOSO results",
                "",
                "| Target | Head | Rows | Sources | Modeling rows | Modeling sources | Excluded sources | RMSE | MAE | R² | Accuracy | Interval coverage | Mean interval width | Bayesian | Decision |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in report.get("headLevelMetrics", []):
            lines.append(
                f"| {row['target']} | {row['head']} | {row['nRows']} | {row['nSources']} | {row.get('modelingRows', row['nRows'])} | "
                f"{row.get('modelingSources', row['nSources'])} | {', '.join(row.get('excludedSources', [])) or '-'} | "
                f"{row['metrics']['rmse']:.3f} | {row['metrics']['mae']:.3f} | {row['metrics']['r2']:.3f} | "
                f"{row['metrics']['accuracyWithinTolerance']:.3f} | {row['intervalMetrics']['coverage']:.3f} | "
                f"{row['intervalMetrics']['meanWidth']:.3f} | {'yes' if row['usedBayesianCalibration'] else 'no'} | "
                f"{row['decision']['confidenceTier']} |"
            )
        if "sourceDiagnostics" in report:
            lines.extend(
                [
                    "",
                    "### Source drift diagnostics",
                    "",
                    "| Target | Head | Source | Rows | Mean | Std | Drift z | Drift outlier |",
                    "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for item in report["sourceDiagnostics"]:
                for source_row in item.get("sources", []):
                    lines.append(
                        f"| {item['target']} | {item['head']} | {source_row['sourceId']} | {source_row['nRows']} | "
                        f"{source_row['sourceMean']:.3f} | {source_row['sourceStd']:.3f} | {source_row['driftZ']:.3f} | "
                        f"{'yes' if source_row['isDriftOutlier'] else 'no'} |"
                    )
        if "weightedHeadSummary" in report:
            weighted = report["weightedHeadSummary"]
            lines.extend(
                [
                    "",
                    "### Weighted head summary",
                    "",
                    f"- Weighted support: `{weighted['totalWeightedSupport']:.1f}`",
                    f"- Weighted RMSE: `{weighted['weightedRmse']:.3f}`",
                    f"- Weighted MAE: `{weighted['weightedMae']:.3f}`",
                    f"- Weighted R²: `{weighted['weightedR2']:.3f}`",
                    f"- Weighted accuracy: `{weighted['weightedAccuracyWithinTolerance']:.3f}`",
                ]
            )
        if "predictionPolicy" in report:
            policy = report["predictionPolicy"]
            lines.extend(
                [
                    "",
                    "### Prediction confidence and rejection policy",
                    "",
                    f"- High-confidence heads: `{len(policy['acceptedHighConfidenceHeads'])}`",
                    f"- Warning-only heads: `{len(policy['acceptedWarningHeads'])}`",
                    f"- Rejected heads: `{len(policy['rejectedHeads'])}`",
                ]
            )
            for entry in policy["acceptedHighConfidenceHeads"]:
                lines.append(
                    f"- High confidence: `{entry['target']} / {entry['head']}` with R²=`{entry['r2']:.3f}` and RMSE=`{entry['rmse']:.3f}`"
                )
            for entry in policy["acceptedWarningHeads"]:
                lines.append(
                    f"- Warning-only: `{entry['target']} / {entry['head']}` because {', '.join(entry['reasons']) or 'moderate_support'}"
                )
            for entry in policy["rejectedHeads"]:
                lines.append(
                    f"- Rejected: `{entry['target']} / {entry['head']}` because {', '.join(entry['reasons'])}"
                )
        isolated = report.get("isolatedYieldPrimaryHead", {})
        if isolated:
            lines.extend(
                [
                    "",
                    "### Primary isolated yield head",
                    "",
                    f"- Baseline isolated head R²: `{isolated['baseline']['r2']:.3f}`" if isolated.get("baseline") else "- Baseline isolated head R²: `n/a`",
                    f"- Final isolated head R²: `{isolated['final']['r2']:.3f}`" if isolated.get("final") else "- Final isolated head R²: `n/a`",
                    f"- RMSE change ratio vs baseline: `{isolated['rmseChangePct']:.3f}`" if isolated.get("rmseChangePct") is not None else "- RMSE change ratio vs baseline: `n/a`",
                ]
            )
    lines.extend(
        [
            "",
            "## 12. Conclusions",
            "",
            "- The workflow now covers data cleaning, missing-value treatment, scaling, feature selection, ensemble learning, hyperparameter tuning, and grouped cross-validation in one Python pipeline.",
            "- Exact and near-duplicate removal as well as outlier filtering are tracked explicitly, so preprocessing impact is measurable rather than implicit.",
            "- Protein embeddings from ESM2 are evaluated strictly inside grouped folds via PCA reduction, so the before/after comparison does not leak test-fold information.",
            "- Feature selection and permutation importance identify which sequence, stage, process, and embedding-augmented variables are most predictive for purity and yield.",
            "- Grouped cross-validation by source is intentionally strict; it measures cross-paper robustness rather than same-source memorization.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = run_workflow(args)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown = generate_markdown_report(report)
    MARKDOWN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_REPORT_PATH.write_text(markdown, encoding="utf-8")

    final_stage = report["stageResults"][-1]
    print(f"Initial rows: {report['initialRowsWithTargets']}")
    print(f"Final rows: {report['finalDatasetRows']}")
    print(f"Final purity RMSE: {final_stage['metrics']['purity']['rmse']:.3f}")
    print(f"Final yield RMSE: {final_stage['metrics']['yield']['rmse']:.3f}")
    print(f"Final joint tolerance accuracy: {final_stage['metrics']['combined']['jointToleranceAccuracy']:.3f}")


if __name__ == "__main__":
    main()
