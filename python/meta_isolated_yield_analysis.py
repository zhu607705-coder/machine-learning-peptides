from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from enhance_dataset import AMINO_ACID_PROPERTIES, calculate_peptide_features, parse_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "real" / "final_purity_yield_literature.csv"
JSON_REPORT_PATH = PROJECT_ROOT / "artifacts" / "isolated-yield-meta-analysis.json"
MARKDOWN_REPORT_PATH = PROJECT_ROOT / "docs" / "isolated-yield-meta-analysis.md"

RANDOM_STATE = 20260302
FEATURE_COLUMNS = [
    "length",
    "avg_hydrophobicity",
    "total_charge",
    "max_coupling_difficulty",
    "hydrophobic_ratio",
    "aromatic_ratio",
]


@dataclass(frozen=True)
class DatasetSummary:
    total_rows: int
    sequence_isolated_rows: int
    unique_sources: int
    canonical_linear_rows: int
    modified_rows: int
    stapled_rows: int


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


def normalize_sequence(raw: Any) -> str:
    return "".join(parse_sequence(str(raw or "")))


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


def chemistry_family(row: pd.Series) -> str:
    source_id = str(row["source_id"])
    topology = str(row.get("topology", "") or "").strip().lower()
    if topology == "stapled" or "stapled" in source_id:
        return "stapled"
    if "sulfotyrosine" in source_id or "sulf" in str(row.get("condition_summary", "")).lower():
        return "sulfated"
    return "canonical_or_simple_modified"


def build_features(sequence: str) -> dict[str, float]:
    peptide = calculate_peptide_features(sequence)
    length = max(peptide.length, 1)
    residues = list(sequence)
    hydrophobic = set("AVILFWYMP")
    aromatic = set("FWY")
    return {
        "length": float(peptide.length),
        "avg_hydrophobicity": float(peptide.avg_hydrophobicity),
        "total_charge": float(peptide.total_charge),
        "max_coupling_difficulty": float(peptide.max_coupling_difficulty),
        "hydrophobic_ratio": float(sum(residue in hydrophobic for residue in residues) / length),
        "aromatic_ratio": float(sum(residue in aromatic for residue in residues) / length),
        "molecular_weight": float(peptide.molecular_weight),
        "longest_hydrophobic_run_norm": float(longest_hydrophobic_run(sequence) / length),
    }


def load_dataset() -> tuple[pd.DataFrame, DatasetSummary]:
    frame = pd.read_csv(DATASET_PATH)
    frame["yield_value"] = frame["yield_pct"].apply(parse_numeric_label)
    frame["purity_value"] = frame["purity_pct"].apply(parse_numeric_label)
    frame["sequence_norm"] = frame["sequence"].apply(normalize_sequence)

    isolated = frame[
        frame["sequence_norm"].astype(str).str.len().gt(0)
        & frame["yield_stage"].fillna("").eq("isolated")
        & frame["yield_value"].notna()
    ].copy()

    feature_rows: list[dict[str, Any]] = []
    for row in isolated.to_dict(orient="records"):
        features = build_features(row["sequence_norm"])
        row.update(features)
        row["publication_year"] = int(row["publication_year"])
        row["has_purity"] = bool(pd.notna(row["purity_value"]))
        row["topology"] = str(row.get("topology", "") or "linear")
        row["chemistry_family"] = chemistry_family(pd.Series(row))
        row["is_stapled"] = int(row["chemistry_family"] == "stapled")
        row["is_modified"] = int(row["chemistry_family"] != "canonical_or_simple_modified")
        row["is_canonical_linear"] = int(row["chemistry_family"] == "canonical_or_simple_modified" and row["topology"] == "linear")
        feature_rows.append(row)

    enriched = pd.DataFrame(feature_rows)
    summary = DatasetSummary(
        total_rows=len(frame),
        sequence_isolated_rows=len(enriched),
        unique_sources=enriched["source_id"].nunique(),
        canonical_linear_rows=int(enriched["is_canonical_linear"].sum()),
        modified_rows=int(enriched["is_modified"].sum()),
        stapled_rows=int(enriched["is_stapled"].sum()),
    )
    return enriched, summary


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def study_level_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    for source_id, group in df.groupby("source_id", sort=True):
        summaries.append(
            {
                "source_id": source_id,
                "publication_year": int(group["publication_year"].iloc[0]),
                "chemistry_family": str(group["chemistry_family"].iloc[0]),
                "n": int(len(group)),
                "mean_yield": float(group["yield_value"].mean()),
                "std_yield": float(group["yield_value"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_purity": float(group["purity_value"].mean()) if group["purity_value"].notna().any() else None,
                "mean_length": float(group["length"].mean()),
            }
        )
    return summaries


def dersimonian_laird_meta(studies: pd.DataFrame) -> dict[str, Any]:
    effects = studies["mean_yield"].to_numpy(dtype=float)
    variances = []
    for _, row in studies.iterrows():
        n = int(row["n"])
        if n > 1 and row["std_yield"] > 0:
            variances.append((float(row["std_yield"]) ** 2) / n)
        else:
            variances.append(25.0)
    variances = np.asarray(variances, dtype=float)
    weights = 1.0 / variances
    fixed_mean = float(np.sum(weights * effects) / np.sum(weights))
    q = float(np.sum(weights * (effects - fixed_mean) ** 2))
    df_q = max(len(effects) - 1, 1)
    c = float(np.sum(weights) - (np.sum(weights ** 2) / np.sum(weights)))
    tau2 = max((q - df_q) / c, 0.0) if c > 0 else 0.0
    random_weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(random_weights * effects) / np.sum(random_weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(random_weights)))
    i2 = max((q - df_q) / q, 0.0) if q > 0 else 0.0
    return {
        "study_count": int(len(effects)),
        "fixed_effect_mean": fixed_mean,
        "random_effect_mean": pooled,
        "random_effect_se": pooled_se,
        "random_effect_ci95": [pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se],
        "tau2": tau2,
        "q": q,
        "i2": i2,
    }


def prepare_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["yield_value"].to_numpy(dtype=float)
    meta = df[["record_id", "source_id", "sequence_norm", "chemistry_family"]].copy()
    return x, y, meta


def fit_mixedlm(train_df: pd.DataFrame) -> tuple[sm.regression.mixed_linear_model.MixedLMResults, StandardScaler]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    exog = sm.add_constant(x_train, has_constant="add")
    endog = train_df["yield_value"].to_numpy(dtype=float)
    groups = train_df["source_id"].to_numpy()
    model = sm.MixedLM(endog=endog, exog=exog, groups=groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
    return result, scaler


def fixed_effect_coefficients(result: sm.regression.mixed_linear_model.MixedLMResults) -> dict[str, float]:
    names = ["intercept", *FEATURE_COLUMNS]
    params = result.fe_params
    return {name: float(params[idx]) for idx, name in enumerate(names)}


def mixedlm_random_effects(result: sm.regression.mixed_linear_model.MixedLMResults) -> dict[str, float]:
    random_effects = {}
    for source_id, values in result.random_effects.items():
        if isinstance(values, (pd.Series, np.ndarray, list, tuple)):
            random_effects[source_id] = float(np.asarray(values)[0])
        else:
            random_effects[source_id] = float(values)
    return random_effects


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) < 2:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
    }


def run_grouped_ridge(df: pd.DataFrame) -> dict[str, Any]:
    predictions: list[float] = []
    truths: list[float] = []
    per_source = []
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    for source_id in sorted(df["source_id"].unique()):
        train_df = df[df["source_id"] != source_id]
        test_df = df[df["source_id"] == source_id]
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].to_numpy(dtype=float))
        x_test = scaler.transform(test_df[FEATURE_COLUMNS].to_numpy(dtype=float))
        y_train = train_df["yield_value"].to_numpy(dtype=float)
        y_test = test_df["yield_value"].to_numpy(dtype=float)

        best_alpha = None
        best_score = float("inf")
        for alpha in alphas:
            model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            score = rmse(y_train, train_pred)
            if score < best_score:
                best_score = score
                best_alpha = alpha

        model = Ridge(alpha=float(best_alpha), random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        predictions.extend(y_pred.tolist())
        truths.extend(y_test.tolist())
        per_source.append(
            {
                "source_id": source_id,
                "n_test": int(len(test_df)),
                "alpha": float(best_alpha),
                **evaluate_predictions(y_test, y_pred),
            }
        )

    return {
        "overall": evaluate_predictions(np.asarray(truths), np.asarray(predictions)),
        "per_source": per_source,
    }


def run_grouped_mixedlm(df: pd.DataFrame) -> dict[str, Any]:
    predictions: list[float] = []
    truths: list[float] = []
    per_source = []

    for source_id in sorted(df["source_id"].unique()):
        train_df = df[df["source_id"] != source_id]
        test_df = df[df["source_id"] == source_id]
        result, scaler = fit_mixedlm(train_df)

        x_test = scaler.transform(test_df[FEATURE_COLUMNS].to_numpy(dtype=float))
        exog_test = sm.add_constant(x_test, has_constant="add")
        y_test = test_df["yield_value"].to_numpy(dtype=float)
        y_pred = np.asarray(result.predict(exog=exog_test), dtype=float)

        predictions.extend(y_pred.tolist())
        truths.extend(y_test.tolist())
        per_source.append(
            {
                "source_id": source_id,
                "n_test": int(len(test_df)),
                **evaluate_predictions(y_test, y_pred),
            }
        )

    return {
        "overall": evaluate_predictions(np.asarray(truths), np.asarray(predictions)),
        "per_source": per_source,
    }


def fit_full_mixedlm(df: pd.DataFrame) -> dict[str, Any]:
    result, scaler = fit_mixedlm(df)
    x_full = scaler.transform(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    exog_full = sm.add_constant(x_full, has_constant="add")
    fitted = np.asarray(result.predict(exog=exog_full), dtype=float)

    covariance_re = np.asarray(result.cov_re)
    random_var = float(covariance_re[0, 0]) if covariance_re.size else 0.0
    resid_var = float(result.scale)
    icc = random_var / (random_var + resid_var) if (random_var + resid_var) > 0 else 0.0

    coeffs = fixed_effect_coefficients(result)
    ranking = [
        {"feature": key, "coefficient": value}
        for key, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True)
        if key != "intercept"
    ]

    return {
        "full_fit_metrics": evaluate_predictions(df["yield_value"].to_numpy(dtype=float), fitted),
        "fixed_effect_coefficients": coeffs,
        "fixed_effect_ranking": ranking,
        "random_intercepts": mixedlm_random_effects(result),
        "random_effect_variance": random_var,
        "residual_variance": resid_var,
        "intraclass_correlation": icc,
    }


def subset_definitions(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    purity_observed = df[df["purity_value"].notna()].copy()
    canonical_only = df[df["is_canonical_linear"] == 1].copy()
    no_stapled = df[df["is_stapled"] == 0].copy()
    return {
        "all_sequence_isolated": df.copy(),
        "purity_observed_only": purity_observed,
        "canonical_linear_only": canonical_only,
        "exclude_stapled": no_stapled,
    }


def run_sensitivity(df: pd.DataFrame) -> dict[str, Any]:
    results = {}
    for name, subset in subset_definitions(df).items():
        if subset["source_id"].nunique() < 2 or len(subset) < 6:
            results[name] = {"skipped": True, "reason": "not enough rows or studies"}
            continue
        results[name] = {
            "rows": int(len(subset)),
            "sources": int(subset["source_id"].nunique()),
            "grouped_ridge": run_grouped_ridge(subset)["overall"],
            "grouped_mixedlm": run_grouped_mixedlm(subset)["overall"],
            "meta_summary": dersimonian_laird_meta(pd.DataFrame(study_level_summary(subset))),
        }
    return results


def build_markdown(
    dataset_summary: DatasetSummary,
    study_summary: list[dict[str, Any]],
    meta_summary: dict[str, Any],
    grouped_ridge: dict[str, Any],
    grouped_mixedlm: dict[str, Any],
    full_mixedlm: dict[str, Any],
    sensitivity: dict[str, Any],
) -> str:
    lines = [
        "# Isolated Yield Meta-Style Analysis",
        "",
        "## Dataset scope",
        f"- Source-tracked rows in full literature table: {dataset_summary.total_rows}",
        f"- Sequence-resolved isolated-yield rows used here: {dataset_summary.sequence_isolated_rows}",
        f"- Unique studies: {dataset_summary.unique_sources}",
        f"- Canonical/simple linear rows: {dataset_summary.canonical_linear_rows}",
        f"- Modified-chemistry rows: {dataset_summary.modified_rows}",
        f"- Stapled rows: {dataset_summary.stapled_rows}",
        "",
        "## Study-level summary",
        "| Study | Year | Family | n | Mean isolated yield | Mean purity | Mean length |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in study_summary:
        purity = "NA" if row["mean_purity"] is None else f'{row["mean_purity"]:.2f}'
        lines.append(
            f'| {row["source_id"]} | {row["publication_year"]} | {row["chemistry_family"]} | {row["n"]} | '
            f'{row["mean_yield"]:.2f} | {purity} | {row["mean_length"]:.2f} |'
        )

    lines.extend(
        [
            "",
            "## Random-effects meta summary",
            f'- Study count: {meta_summary["study_count"]}',
            f'- Fixed-effect pooled mean isolated yield: {meta_summary["fixed_effect_mean"]:.2f}',
            f'- Random-effect pooled mean isolated yield: {meta_summary["random_effect_mean"]:.2f}',
            f'- Random-effect 95% CI: {meta_summary["random_effect_ci95"][0]:.2f} to {meta_summary["random_effect_ci95"][1]:.2f}',
            f'- Tau^2: {meta_summary["tau2"]:.3f}',
            f'- I^2: {meta_summary["i2"]:.3f}',
            "",
            "## Leave-one-study-out predictive comparison",
            f'- Grouped Ridge RMSE: {grouped_ridge["overall"]["rmse"]:.3f}',
            f'- Grouped Ridge MAE: {grouped_ridge["overall"]["mae"]:.3f}',
            f'- Grouped Ridge R^2: {grouped_ridge["overall"]["r2"]:.3f}',
            f'- MixedLM RMSE: {grouped_mixedlm["overall"]["rmse"]:.3f}',
            f'- MixedLM MAE: {grouped_mixedlm["overall"]["mae"]:.3f}',
            f'- MixedLM R^2: {grouped_mixedlm["overall"]["r2"]:.3f}',
            "",
            "## Full mixed-effects fit",
            f'- In-sample RMSE: {full_mixedlm["full_fit_metrics"]["rmse"]:.3f}',
            f'- In-sample MAE: {full_mixedlm["full_fit_metrics"]["mae"]:.3f}',
            f'- In-sample R^2: {full_mixedlm["full_fit_metrics"]["r2"]:.3f}',
            f'- Random intercept variance: {full_mixedlm["random_effect_variance"]:.3f}',
            f'- Residual variance: {full_mixedlm["residual_variance"]:.3f}',
            f'- Intraclass correlation (study-level ICC): {full_mixedlm["intraclass_correlation"]:.3f}',
            "",
            "Top fixed effects by absolute standardized coefficient:",
        ]
    )
    for row in full_mixedlm["fixed_effect_ranking"][:6]:
        lines.append(f'- {row["feature"]}: {row["coefficient"]:.3f}')

    lines.extend(
        [
            "",
            "Study random intercepts:",
        ]
    )
    for source_id, value in sorted(full_mixedlm["random_intercepts"].items()):
        lines.append(f"- {source_id}: {value:.3f}")

    lines.extend(["", "## Sensitivity analysis"])
    for name, payload in sensitivity.items():
        if payload.get("skipped"):
            lines.append(f"- {name}: skipped ({payload['reason']})")
            continue
        lines.append(
            f"- {name}: rows={payload['rows']}, sources={payload['sources']}, "
            f"ridge_rmse={payload['grouped_ridge']['rmse']:.3f}, "
            f"mixedlm_rmse={payload['grouped_mixedlm']['rmse']:.3f}, "
            f"meta_mean={payload['meta_summary']['random_effect_mean']:.2f}, "
            f"I^2={payload['meta_summary']['i2']:.3f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- The random-effects meta summary quantifies between-study heterogeneity in mean isolated yield.",
            "- The mixed-effects model treats `source_id` as a random intercept, so chemistry and reporting conventions that cluster within studies are partially separated from sequence-level signals.",
            "- Sensitivity subsets show whether the sequence-yield relationship survives after excluding specialized chemistries such as stapled or sulfated peptides.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    np.random.seed(RANDOM_STATE)
    df, dataset_summary = load_dataset()
    study_summary = study_level_summary(df)
    studies_df = pd.DataFrame(study_summary)
    meta_summary = dersimonian_laird_meta(studies_df)
    grouped_ridge = run_grouped_ridge(df)
    grouped_mixedlm = run_grouped_mixedlm(df)
    full_mixedlm = fit_full_mixedlm(df)
    sensitivity = run_sensitivity(df)

    report = {
        "dataset_summary": asdict(dataset_summary),
        "study_summary": study_summary,
        "meta_summary": meta_summary,
        "grouped_ridge": grouped_ridge,
        "grouped_mixedlm": grouped_mixedlm,
        "full_mixedlm": full_mixedlm,
        "sensitivity": sensitivity,
    }

    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MARKDOWN_REPORT_PATH.write_text(
        build_markdown(
            dataset_summary=dataset_summary,
            study_summary=study_summary,
            meta_summary=meta_summary,
            grouped_ridge=grouped_ridge,
            grouped_mixedlm=grouped_mixedlm,
            full_mixedlm=full_mixedlm,
            sensitivity=sensitivity,
        ),
        encoding="utf-8",
    )

    print(f"Wrote JSON report to {JSON_REPORT_PATH}")
    print(f"Wrote Markdown report to {MARKDOWN_REPORT_PATH}")


if __name__ == "__main__":
    main()
