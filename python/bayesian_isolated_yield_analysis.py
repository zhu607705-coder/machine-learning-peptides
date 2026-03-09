from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from meta_isolated_yield_analysis import (
    FEATURE_COLUMNS,
    DATASET_PATH,
    DatasetSummary,
    dersimonian_laird_meta,
    evaluate_predictions,
    load_dataset,
    run_grouped_mixedlm,
    run_grouped_ridge,
    study_level_summary,
    subset_definitions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT_PATH = PROJECT_ROOT / "artifacts" / "bayesian-isolated-yield-analysis.json"
MARKDOWN_REPORT_PATH = PROJECT_ROOT / "docs" / "bayesian-isolated-yield-analysis.md"
RANDOM_STATE = 20260302


class BayesianHierarchicalRegressor:
    def __init__(
        self,
        iterations: int = 6000,
        burn_in: int = 2000,
        thin: int = 4,
        beta_prior_var: float = 25.0,
        intercept_prior_var: float = 2500.0,
        sigma_prior_a: float = 2.0,
        sigma_prior_b: float = 50.0,
        tau_prior_a: float = 2.0,
        tau_prior_b: float = 50.0,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.iterations = iterations
        self.burn_in = burn_in
        self.thin = thin
        self.beta_prior_var = beta_prior_var
        self.intercept_prior_var = intercept_prior_var
        self.sigma_prior_a = sigma_prior_a
        self.sigma_prior_b = sigma_prior_b
        self.tau_prior_a = tau_prior_a
        self.tau_prior_b = tau_prior_b
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "BayesianHierarchicalRegressor":
        rng = np.random.default_rng(self.random_state)
        group_labels, group_index = np.unique(groups, return_inverse=True)
        n_obs, n_features = x.shape

        z = np.column_stack([np.ones(n_obs), x])
        prior_var = np.array([self.intercept_prior_var, *([self.beta_prior_var] * n_features)], dtype=float)
        prior_prec = np.diag(1.0 / prior_var)
        prior_mean = np.zeros(n_features + 1, dtype=float)

        beta = np.zeros(n_features + 1, dtype=float)
        u = np.zeros(len(group_labels), dtype=float)
        sigma2 = float(np.var(y, ddof=1)) if n_obs > 1 else 100.0
        sigma2 = max(sigma2, 1.0)
        tau2 = sigma2

        beta_samples = []
        u_samples = []
        sigma2_samples = []
        tau2_samples = []

        group_positions = [np.where(group_index == idx)[0] for idx in range(len(group_labels))]

        for step in range(self.iterations):
            residual_without_beta = y - u[group_index]
            precision = (z.T @ z) / sigma2 + prior_prec
            covariance = np.linalg.inv(precision)
            mean = covariance @ ((z.T @ residual_without_beta) / sigma2 + prior_prec @ prior_mean)
            beta = rng.multivariate_normal(mean=mean, cov=covariance)

            fixed_part = z @ beta
            for idx, positions in enumerate(group_positions):
                n_group = len(positions)
                var_j = 1.0 / (n_group / sigma2 + 1.0 / tau2)
                mean_j = var_j * np.sum(y[positions] - fixed_part[positions]) / sigma2
                u[idx] = rng.normal(loc=mean_j, scale=np.sqrt(var_j))

            residual = y - fixed_part - u[group_index]
            shape_sigma = self.sigma_prior_a + n_obs / 2.0
            scale_sigma = self.sigma_prior_b + 0.5 * float(residual @ residual)
            sigma2 = 1.0 / rng.gamma(shape_sigma, 1.0 / scale_sigma)

            shape_tau = self.tau_prior_a + len(group_labels) / 2.0
            scale_tau = self.tau_prior_b + 0.5 * float(u @ u)
            tau2 = 1.0 / rng.gamma(shape_tau, 1.0 / scale_tau)

            if step >= self.burn_in and ((step - self.burn_in) % self.thin == 0):
                beta_samples.append(beta.copy())
                u_samples.append(u.copy())
                sigma2_samples.append(float(sigma2))
                tau2_samples.append(float(tau2))

        self.group_labels_ = group_labels
        self.beta_samples_ = np.asarray(beta_samples, dtype=float)
        self.u_samples_ = np.asarray(u_samples, dtype=float)
        self.sigma2_samples_ = np.asarray(sigma2_samples, dtype=float)
        self.tau2_samples_ = np.asarray(tau2_samples, dtype=float)
        self.posterior_beta_mean_ = self.beta_samples_.mean(axis=0)
        self.posterior_u_mean_ = self.u_samples_.mean(axis=0)
        self.posterior_sigma2_mean_ = float(self.sigma2_samples_.mean())
        self.posterior_tau2_mean_ = float(self.tau2_samples_.mean())
        return self

    def predict_new_study(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = np.column_stack([np.ones(len(x)), x])
        sample_means = self.beta_samples_ @ design.T
        point_mean = sample_means.mean(axis=0)

        predictive_std = np.sqrt(self.sigma2_samples_[:, None] + self.tau2_samples_[:, None])
        predictive_samples = sample_means + np.random.default_rng(self.random_state).normal(
            loc=0.0,
            scale=predictive_std,
            size=sample_means.shape,
        )
        credible = np.percentile(predictive_samples, [2.5, 97.5], axis=0).T
        return point_mean, credible

    def predict_known_groups(self, x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        design = np.column_stack([np.ones(len(x)), x])
        group_map = {label: idx for idx, label in enumerate(self.group_labels_)}
        offsets = np.array([self.posterior_u_mean_[group_map[group]] if group in group_map else 0.0 for group in groups], dtype=float)
        sample_means = self.beta_samples_ @ design.T
        sample_offsets = np.array(
            [self.u_samples_[:, group_map[group]] if group in group_map else np.zeros(len(self.u_samples_)) for group in groups],
            dtype=float,
        ).T
        predictive_samples = sample_means + sample_offsets
        point_mean = predictive_samples.mean(axis=0)
        credible = np.percentile(predictive_samples, [2.5, 97.5], axis=0).T
        return point_mean, credible

    def posterior_summary(self, feature_names: list[str]) -> dict[str, Any]:
        names = ["intercept", *feature_names]
        coefficient_summary = {}
        for idx, name in enumerate(names):
            samples = self.beta_samples_[:, idx]
            coefficient_summary[name] = {
                "mean": float(np.mean(samples)),
                "sd": float(np.std(samples, ddof=1)),
                "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                "prob_positive": float(np.mean(samples > 0.0)),
            }

        random_intercepts = {}
        for idx, source_id in enumerate(self.group_labels_):
            samples = self.u_samples_[:, idx]
            random_intercepts[str(source_id)] = {
                "mean": float(np.mean(samples)),
                "sd": float(np.std(samples, ddof=1)),
                "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            }

        icc_samples = self.tau2_samples_ / (self.tau2_samples_ + self.sigma2_samples_)
        return {
            "coefficients": coefficient_summary,
            "random_intercepts": random_intercepts,
            "sigma2": {
                "mean": float(np.mean(self.sigma2_samples_)),
                "ci95": [float(np.percentile(self.sigma2_samples_, 2.5)), float(np.percentile(self.sigma2_samples_, 97.5))],
            },
            "tau2": {
                "mean": float(np.mean(self.tau2_samples_)),
                "ci95": [float(np.percentile(self.tau2_samples_, 2.5)), float(np.percentile(self.tau2_samples_, 97.5))],
            },
            "icc": {
                "mean": float(np.mean(icc_samples)),
                "ci95": [float(np.percentile(icc_samples, 2.5)), float(np.percentile(icc_samples, 97.5))],
            },
            "n_posterior_draws": int(len(self.beta_samples_)),
        }


def fit_bayesian_model(
    train_df: pd.DataFrame,
    random_state: int,
) -> tuple[BayesianHierarchicalRegressor, StandardScaler]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    y_train = train_df["yield_value"].to_numpy(dtype=float)
    groups = train_df["source_id"].to_numpy()
    model = BayesianHierarchicalRegressor(random_state=random_state)
    model.fit(x=x_train, y=y_train, groups=groups)
    return model, scaler


def run_grouped_bayesian(df: pd.DataFrame) -> dict[str, Any]:
    predictions: list[float] = []
    truths: list[float] = []
    intervals = []
    per_source = []

    for fold_idx, source_id in enumerate(sorted(df["source_id"].unique())):
        train_df = df[df["source_id"] != source_id]
        test_df = df[df["source_id"] == source_id]
        model, scaler = fit_bayesian_model(train_df, random_state=RANDOM_STATE + fold_idx)
        x_test = scaler.transform(test_df[FEATURE_COLUMNS].to_numpy(dtype=float))
        y_test = test_df["yield_value"].to_numpy(dtype=float)
        y_pred, credible = model.predict_new_study(x_test)

        predictions.extend(y_pred.tolist())
        truths.extend(y_test.tolist())
        intervals.extend(credible.tolist())
        per_source.append(
            {
                "source_id": source_id,
                "n_test": int(len(test_df)),
                **evaluate_predictions(y_test, y_pred),
                "mean_interval_width": float(np.mean(credible[:, 1] - credible[:, 0])),
                "coverage": float(np.mean((y_test >= credible[:, 0]) & (y_test <= credible[:, 1]))),
            }
        )

    y_true = np.asarray(truths, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    credible_array = np.asarray(intervals, dtype=float)
    overall = evaluate_predictions(y_true, y_pred)
    overall["mean_interval_width"] = float(np.mean(credible_array[:, 1] - credible_array[:, 0]))
    overall["coverage"] = float(np.mean((y_true >= credible_array[:, 0]) & (y_true <= credible_array[:, 1])))
    return {"overall": overall, "per_source": per_source}


def fit_full_bayesian(df: pd.DataFrame) -> dict[str, Any]:
    model, scaler = fit_bayesian_model(df, random_state=RANDOM_STATE)
    x_full = scaler.transform(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    groups = df["source_id"].to_numpy()
    y_true = df["yield_value"].to_numpy(dtype=float)
    y_pred, credible = model.predict_known_groups(x_full, groups)

    posterior = model.posterior_summary(FEATURE_COLUMNS)
    ranking = [
        {
            "feature": feature,
            "mean": payload["mean"],
            "prob_positive": payload["prob_positive"],
            "ci95": payload["ci95"],
        }
        for feature, payload in posterior["coefficients"].items()
        if feature != "intercept"
    ]
    ranking.sort(key=lambda row: abs(row["mean"]), reverse=True)

    return {
        "full_fit_metrics": {
            **evaluate_predictions(y_true, y_pred),
            "mean_interval_width": float(np.mean(credible[:, 1] - credible[:, 0])),
            "coverage": float(np.mean((y_true >= credible[:, 0]) & (y_true <= credible[:, 1]))),
        },
        "posterior_summary": posterior,
        "feature_ranking": ranking,
    }


def run_sensitivity_bayesian(df: pd.DataFrame) -> dict[str, Any]:
    results = {}
    for name, subset in subset_definitions(df).items():
        if subset["source_id"].nunique() < 2 or len(subset) < 6:
            results[name] = {"skipped": True, "reason": "not enough rows or studies"}
            continue
        bayesian = run_grouped_bayesian(subset)["overall"]
        results[name] = {
            "rows": int(len(subset)),
            "sources": int(subset["source_id"].nunique()),
            "grouped_ridge": run_grouped_ridge(subset)["overall"],
            "grouped_bayesian": bayesian,
            "meta_summary": dersimonian_laird_meta(pd.DataFrame(study_level_summary(subset))),
        }
    return results


def build_markdown(
    dataset_summary: DatasetSummary,
    study_summary: list[dict[str, Any]],
    meta_summary: dict[str, Any],
    grouped_ridge: dict[str, Any],
    grouped_mixedlm: dict[str, Any],
    grouped_bayesian: dict[str, Any],
    full_bayesian: dict[str, Any],
    sensitivity: dict[str, Any],
) -> str:
    lines = [
        "# Bayesian Hierarchical Isolated-Yield Analysis",
        "",
        "## Dataset scope",
        f"- Full literature rows: {dataset_summary.total_rows}",
        f"- Sequence-resolved isolated-yield rows: {dataset_summary.sequence_isolated_rows}",
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
            "## Meta heterogeneity",
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
            f'- Bayesian RMSE: {grouped_bayesian["overall"]["rmse"]:.3f}',
            f'- Bayesian MAE: {grouped_bayesian["overall"]["mae"]:.3f}',
            f'- Bayesian R^2: {grouped_bayesian["overall"]["r2"]:.3f}',
            f'- Bayesian interval coverage: {grouped_bayesian["overall"]["coverage"]:.3f}',
            f'- Bayesian mean 95% interval width: {grouped_bayesian["overall"]["mean_interval_width"]:.3f}',
            "",
            "## Full Bayesian hierarchical fit",
            f'- In-sample RMSE: {full_bayesian["full_fit_metrics"]["rmse"]:.3f}',
            f'- In-sample MAE: {full_bayesian["full_fit_metrics"]["mae"]:.3f}',
            f'- In-sample R^2: {full_bayesian["full_fit_metrics"]["r2"]:.3f}',
            f'- In-sample coverage: {full_bayesian["full_fit_metrics"]["coverage"]:.3f}',
            f'- Mean interval width: {full_bayesian["full_fit_metrics"]["mean_interval_width"]:.3f}',
            f'- Posterior ICC mean: {full_bayesian["posterior_summary"]["icc"]["mean"]:.3f}',
            f'- Posterior ICC 95% CI: {full_bayesian["posterior_summary"]["icc"]["ci95"][0]:.3f} to {full_bayesian["posterior_summary"]["icc"]["ci95"][1]:.3f}',
            "",
            "Top posterior fixed effects by absolute posterior mean:",
        ]
    )
    for row in full_bayesian["feature_ranking"][:6]:
        lines.append(
            f'- {row["feature"]}: mean={row["mean"]:.3f}, '
            f'P(>0)={row["prob_positive"]:.3f}, '
            f'95% CI={row["ci95"][0]:.3f} to {row["ci95"][1]:.3f}'
        )

    lines.extend(["", "Posterior random intercepts:"])
    for source_id, payload in sorted(full_bayesian["posterior_summary"]["random_intercepts"].items()):
        lines.append(
            f'- {source_id}: mean={payload["mean"]:.3f}, '
            f'95% CI={payload["ci95"][0]:.3f} to {payload["ci95"][1]:.3f}'
        )

    lines.extend(["", "## Sensitivity analysis"])
    for name, payload in sensitivity.items():
        if payload.get("skipped"):
            lines.append(f"- {name}: skipped ({payload['reason']})")
            continue
        lines.append(
            f"- {name}: rows={payload['rows']}, sources={payload['sources']}, "
            f"ridge_rmse={payload['grouped_ridge']['rmse']:.3f}, "
            f"bayes_rmse={payload['grouped_bayesian']['rmse']:.3f}, "
            f"bayes_coverage={payload['grouped_bayesian']['coverage']:.3f}, "
            f"meta_mean={payload['meta_summary']['random_effect_mean']:.2f}, "
            f"I^2={payload['meta_summary']['i2']:.3f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- This model keeps study-specific random intercepts but estimates them in a Bayesian way, which is more stable on tiny multi-study datasets than relying only on asymptotic MixedLM summaries.",
            "- Leave-one-study-out prediction for a new study integrates over an unseen-study random intercept with mean zero, so point predictions stay conservative.",
            "- Posterior ICC close to 1 means study/source effects dominate the shared sequence-level signal.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    np.random.seed(RANDOM_STATE)
    df, dataset_summary = load_dataset()
    study_summary = study_level_summary(df)
    meta_summary = dersimonian_laird_meta(pd.DataFrame(study_summary))
    grouped_ridge = run_grouped_ridge(df)
    grouped_mixedlm = run_grouped_mixedlm(df)
    grouped_bayesian = run_grouped_bayesian(df)
    full_bayesian = fit_full_bayesian(df)
    sensitivity = run_sensitivity_bayesian(df)

    report = {
        "dataset_path": str(DATASET_PATH),
        "dataset_summary": asdict(dataset_summary),
        "study_summary": study_summary,
        "meta_summary": meta_summary,
        "grouped_ridge": grouped_ridge,
        "grouped_mixedlm": grouped_mixedlm,
        "grouped_bayesian": grouped_bayesian,
        "full_bayesian": full_bayesian,
        "sensitivity": sensitivity,
        "sampling_config": {
            "iterations": 6000,
            "burn_in": 2000,
            "thin": 4,
        },
    }

    JSON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MARKDOWN_REPORT_PATH.write_text(
        build_markdown(
            dataset_summary=dataset_summary,
            study_summary=study_summary,
            meta_summary=meta_summary,
            grouped_ridge=grouped_ridge,
            grouped_mixedlm=grouped_mixedlm,
            grouped_bayesian=grouped_bayesian,
            full_bayesian=full_bayesian,
            sensitivity=sensitivity,
        ),
        encoding="utf-8",
    )

    print(f"Wrote JSON report to {JSON_REPORT_PATH}")
    print(f"Wrote Markdown report to {MARKDOWN_REPORT_PATH}")


if __name__ == "__main__":
    main()
