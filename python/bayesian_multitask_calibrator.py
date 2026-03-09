from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import pymc as pm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pm = None  # type: ignore


RANDOM_STATE = 20260305


@dataclass
class _TargetPosterior:
    beta_samples: np.ndarray
    source_samples: np.ndarray
    stage_samples: np.ndarray
    sigma2_samples: np.ndarray
    tau_source2_samples: np.ndarray
    tau_stage2_samples: np.ndarray
    source_labels: np.ndarray
    stage_labels: np.ndarray
    feature_names: list[str]


class _GibbsTargetModel:
    def __init__(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        random_state: int,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source: np.ndarray,
        stage: np.ndarray,
        feature_names: list[str],
    ) -> _TargetPosterior:
        source_labels, source_index = np.unique(source, return_inverse=True)
        stage_labels, stage_index = np.unique(stage, return_inverse=True)
        n_obs, n_features = x.shape
        z = np.column_stack([np.ones(n_obs), x])
        prior_var = np.array([2500.0] + [25.0] * n_features, dtype=float)
        prior_prec = np.diag(1.0 / prior_var)

        total_samples = self.draws * max(self.chains, 1)
        beta_samples: list[np.ndarray] = []
        source_samples: list[np.ndarray] = []
        stage_samples: list[np.ndarray] = []
        sigma2_samples: list[float] = []
        tau_source2_samples: list[float] = []
        tau_stage2_samples: list[float] = []
        group_positions_source = [np.where(source_index == idx)[0] for idx in range(len(source_labels))]
        group_positions_stage = [np.where(stage_index == idx)[0] for idx in range(len(stage_labels))]

        for chain in range(max(self.chains, 1)):
            rng = np.random.default_rng(self.random_state + (chain * 997))
            beta = np.zeros(n_features + 1, dtype=float)
            u_source = np.zeros(len(source_labels), dtype=float)
            u_stage = np.zeros(len(stage_labels), dtype=float)
            sigma2 = max(float(np.var(y, ddof=1)) if len(y) > 1 else 100.0, 1.0)
            tau_source2 = sigma2
            tau_stage2 = sigma2

            chain_samples = 0
            max_iterations = self.tune + self.draws
            thin = 1

            for step in range(max_iterations):
                residual_wo_beta = y - u_source[source_index] - u_stage[stage_index]
                precision = (z.T @ z) / sigma2 + prior_prec + (1e-8 * np.eye(z.shape[1]))
                covariance = np.linalg.inv(precision)
                mean = covariance @ ((z.T @ residual_wo_beta) / sigma2)
                beta = rng.multivariate_normal(mean=mean, cov=covariance)

                fixed = z @ beta
                for idx, positions in enumerate(group_positions_source):
                    n_group = len(positions)
                    var_j = 1.0 / (n_group / sigma2 + 1.0 / tau_source2)
                    mean_j = var_j * np.sum(y[positions] - fixed[positions] - u_stage[stage_index[positions]]) / sigma2
                    u_source[idx] = rng.normal(loc=mean_j, scale=np.sqrt(var_j))

                for idx, positions in enumerate(group_positions_stage):
                    n_group = len(positions)
                    var_j = 1.0 / (n_group / sigma2 + 1.0 / tau_stage2)
                    mean_j = var_j * np.sum(y[positions] - fixed[positions] - u_source[source_index[positions]]) / sigma2
                    u_stage[idx] = rng.normal(loc=mean_j, scale=np.sqrt(var_j))

                residual = y - fixed - u_source[source_index] - u_stage[stage_index]
                sigma2 = 1.0 / rng.gamma(2.0 + (n_obs / 2.0), 1.0 / (50.0 + 0.5 * float(residual @ residual)))
                tau_source2 = 1.0 / rng.gamma(
                    2.0 + (len(source_labels) / 2.0),
                    1.0 / (50.0 + 0.5 * float(u_source @ u_source)),
                )
                tau_stage2 = 1.0 / rng.gamma(
                    2.0 + (len(stage_labels) / 2.0),
                    1.0 / (50.0 + 0.5 * float(u_stage @ u_stage)),
                )

                should_collect = step >= self.tune and ((step - self.tune) % thin == 0)
                if should_collect:
                    beta_samples.append(beta.copy())
                    source_samples.append(u_source.copy())
                    stage_samples.append(u_stage.copy())
                    sigma2_samples.append(float(sigma2))
                    tau_source2_samples.append(float(tau_source2))
                    tau_stage2_samples.append(float(tau_stage2))
                    chain_samples += 1
                    if chain_samples >= self.draws:
                        break

        return _TargetPosterior(
            beta_samples=np.asarray(beta_samples, dtype=float),
            source_samples=np.asarray(source_samples, dtype=float),
            stage_samples=np.asarray(stage_samples, dtype=float),
            sigma2_samples=np.asarray(sigma2_samples, dtype=float),
            tau_source2_samples=np.asarray(tau_source2_samples, dtype=float),
            tau_stage2_samples=np.asarray(tau_stage2_samples, dtype=float),
            source_labels=source_labels,
            stage_labels=stage_labels,
            feature_names=feature_names,
        )


class _PyMCTargetModel:
    def __init__(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        target_accept: float,
        likelihood: str,
        student_t_nu: float,
        random_state: int,
    ) -> None:
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.likelihood = likelihood
        self.student_t_nu = student_t_nu
        self.random_state = random_state

    @staticmethod
    def _flatten_samples(array: np.ndarray) -> np.ndarray:
        # Input shape from arviz/xarray is (chains, draws, ...).
        return array.reshape((-1, *array.shape[2:]))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source: np.ndarray,
        stage: np.ndarray,
        feature_names: list[str],
    ) -> _TargetPosterior:
        if pm is None:
            raise RuntimeError("pymc is unavailable in this environment")

        source_labels, source_index = np.unique(source, return_inverse=True)
        stage_labels, stage_index = np.unique(stage, return_inverse=True)
        n_features = x.shape[1]
        coords = {
            "feature": np.asarray(feature_names, dtype=object),
            "source": np.asarray(source_labels, dtype=object),
            "stage": np.asarray(stage_labels, dtype=object),
        }

        with pm.Model(coords=coords):
            x_data = pm.Data("x_data", x)
            source_idx = pm.Data("source_idx", source_index.astype(np.int32))
            stage_idx = pm.Data("stage_idx", stage_index.astype(np.int32))

            beta0 = pm.Normal("beta0", mu=0.0, sigma=10.0)
            beta = pm.Normal("beta", mu=0.0, sigma=5.0, dims="feature")

            tau_source = pm.HalfNormal("tau_source", sigma=10.0)
            tau_stage = pm.HalfNormal("tau_stage", sigma=10.0)
            source_raw = pm.Normal("source_raw", mu=0.0, sigma=1.0, dims="source")
            stage_raw = pm.Normal("stage_raw", mu=0.0, sigma=1.0, dims="stage")
            source_effect = pm.Deterministic("source_effect", source_raw * tau_source, dims="source")
            stage_effect = pm.Deterministic("stage_effect", stage_raw * tau_stage, dims="stage")

            sigma = pm.HalfNormal("sigma", sigma=20.0)
            mu = beta0 + pm.math.dot(x_data, beta) + source_effect[source_idx] + stage_effect[stage_idx]
            if self.likelihood == "student_t":
                pm.StudentT("obs", nu=float(self.student_t_nu), mu=mu, sigma=sigma, observed=y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=1,
                random_seed=self.random_state,
                target_accept=self.target_accept,
                progressbar=False,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )

        posterior = idata.posterior
        beta0_samples = self._flatten_samples(np.asarray(posterior["beta0"], dtype=float)).reshape(-1)
        beta_samples_only = self._flatten_samples(np.asarray(posterior["beta"], dtype=float))
        source_samples = self._flatten_samples(np.asarray(posterior["source_effect"], dtype=float))
        stage_samples = self._flatten_samples(np.asarray(posterior["stage_effect"], dtype=float))
        sigma_samples = self._flatten_samples(np.asarray(posterior["sigma"], dtype=float)).reshape(-1)
        tau_source_samples = self._flatten_samples(np.asarray(posterior["tau_source"], dtype=float)).reshape(-1)
        tau_stage_samples = self._flatten_samples(np.asarray(posterior["tau_stage"], dtype=float)).reshape(-1)

        beta_samples = np.column_stack([beta0_samples, beta_samples_only[:, :n_features]])
        return _TargetPosterior(
            beta_samples=beta_samples.astype(float),
            source_samples=source_samples.astype(float),
            stage_samples=stage_samples.astype(float),
            sigma2_samples=(sigma_samples ** 2).astype(float),
            tau_source2_samples=(tau_source_samples ** 2).astype(float),
            tau_stage2_samples=(tau_stage_samples ** 2).astype(float),
            source_labels=np.asarray(source_labels, dtype=object),
            stage_labels=np.asarray(stage_labels, dtype=object),
            feature_names=feature_names,
        )


class BayesianMultitaskCalibrator:
    def __init__(
        self,
        *,
        draws: int = 1500,
        tune: int = 1500,
        chains: int = 4,
        target_accept: float = 0.9,
        likelihood: str = "normal",
        student_t_nu: float = 4.0,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.draws = int(draws)
        self.tune = int(tune)
        self.chains = int(chains)
        self.target_accept = float(target_accept)
        self.likelihood = str(likelihood).strip().lower()
        if self.likelihood not in {"normal", "student_t"}:
            raise ValueError("likelihood must be one of {'normal', 'student_t'}")
        self.student_t_nu = float(student_t_nu)
        self.random_state = int(random_state)

    def _resolve_stage_column(self, target: str, group_cols: dict[str, str]) -> str:
        candidates = [
            target,
            f"{target}_stage",
            target.replace("_target", "_stage"),
            target.replace("_target", "_stage_canon"),
            "stage",
        ]
        for candidate in candidates:
            if candidate in group_cols:
                return group_cols[candidate]
        return group_cols.get("stage", "yield_stage_canon")

    def fit(
        self,
        train_df: pd.DataFrame,
        features: list[str],
        targets: list[str],
        group_cols: dict[str, str],
    ) -> "BayesianMultitaskCalibrator":
        self.features_ = list(features)
        self.targets_ = list(targets)
        self.group_cols_ = dict(group_cols)
        self.scaler_ = StandardScaler()
        x_frame = train_df[self.features_].apply(pd.to_numeric, errors="coerce")
        self.feature_fill_values_ = x_frame.mean(axis=0).fillna(0.0)
        x_train = self.scaler_.fit_transform(x_frame.fillna(self.feature_fill_values_).to_numpy(dtype=float))
        source_col = self.group_cols_.get("source", "source_id")
        source_values = train_df[source_col].fillna("unknown").astype(str).to_numpy()
        self.posteriors_: dict[str, _TargetPosterior] = {}
        self.target_backends_: dict[str, str] = {}

        for offset, target in enumerate(self.targets_):
            stage_col = self._resolve_stage_column(target, self.group_cols_)
            stage_values = train_df[stage_col].fillna("unknown").astype(str).to_numpy()
            y_train = train_df[target].to_numpy(dtype=float)

            fitted = False
            if pm is not None:
                try:
                    pymc_model = _PyMCTargetModel(
                        draws=self.draws,
                        tune=self.tune,
                        chains=self.chains,
                        target_accept=self.target_accept,
                        likelihood=self.likelihood,
                        student_t_nu=self.student_t_nu,
                        random_state=self.random_state + (offset * 131),
                    )
                    self.posteriors_[target] = pymc_model.fit(
                        x=x_train,
                        y=y_train,
                        source=source_values,
                        stage=stage_values,
                        feature_names=self.features_,
                    )
                    self.target_backends_[target] = "pymc_nuts"
                    fitted = True
                except Exception:
                    fitted = False

            if not fitted:
                gibbs = _GibbsTargetModel(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    random_state=self.random_state + (offset * 131),
                )
                self.posteriors_[target] = gibbs.fit(
                    x=x_train,
                    y=y_train,
                    source=source_values,
                    stage=stage_values,
                    feature_names=self.features_,
                )
                self.target_backends_[target] = "gibbs_fallback"

        unique_backends = sorted(set(self.target_backends_.values()))
        self.backend_ = unique_backends[0] if len(unique_backends) == 1 else "hybrid_pymc_gibbs"
        self.pymc_available_ = bool(pm is not None)
        return self

    def _predict_target_samples(
        self,
        posterior: _TargetPosterior,
        x_test: np.ndarray,
        source_values: np.ndarray,
        stage_values: np.ndarray,
        mode: str,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        design = np.column_stack([np.ones(len(x_test)), x_test])
        fixed_samples = posterior.beta_samples @ design.T
        draws = fixed_samples.shape[0]

        source_map = {label: idx for idx, label in enumerate(posterior.source_labels)}
        stage_map = {label: idx for idx, label in enumerate(posterior.stage_labels)}
        source_offsets = np.zeros((draws, len(x_test)), dtype=float)
        stage_offsets = np.zeros((draws, len(x_test)), dtype=float)

        for idx, source_label in enumerate(source_values):
            mapped = source_map.get(source_label)
            if mapped is not None:
                source_offsets[:, idx] = posterior.source_samples[:, mapped]
            elif mode == "sample_new_source":
                source_offsets[:, idx] = rng.normal(
                    loc=0.0,
                    scale=np.sqrt(np.maximum(posterior.tau_source2_samples, 1e-9)),
                )

        for idx, stage_label in enumerate(stage_values):
            mapped = stage_map.get(stage_label)
            if mapped is not None:
                stage_offsets[:, idx] = posterior.stage_samples[:, mapped]
            elif mode == "sample_new_source":
                stage_offsets[:, idx] = rng.normal(
                    loc=0.0,
                    scale=np.sqrt(np.maximum(posterior.tau_stage2_samples, 1e-9)),
                )

        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(np.maximum(posterior.sigma2_samples[:, None], 1e-9)),
            size=fixed_samples.shape,
        )
        if self.likelihood == "student_t":
            nu = max(float(self.student_t_nu), 2.1)
            noise = rng.standard_t(df=nu, size=fixed_samples.shape) * np.sqrt(
                np.maximum(posterior.sigma2_samples[:, None], 1e-9)
            )
        return fixed_samples + source_offsets + stage_offsets + noise

    def predict(
        self,
        test_df: pd.DataFrame,
        mode: str = "population_for_unseen_source",
    ) -> dict[str, Any]:
        x_frame = test_df[self.features_].apply(pd.to_numeric, errors="coerce")
        x_test = self.scaler_.transform(x_frame.fillna(self.feature_fill_values_).to_numpy(dtype=float))
        source_col = self.group_cols_.get("source", "source_id")
        source_values = test_df[source_col].fillna("unknown").astype(str).to_numpy()
        mapped_mode = "sample_new_source" if mode == "sample_new_source" else "population_for_unseen_source"

        predictions: dict[str, np.ndarray] = {}
        intervals: dict[str, np.ndarray] = {}
        widths: dict[str, float] = {}

        for idx, target in enumerate(self.targets_):
            posterior = self.posteriors_[target]
            stage_col = self._resolve_stage_column(target, self.group_cols_)
            stage_values = test_df[stage_col].fillna("unknown").astype(str).to_numpy()
            sample_matrix = self._predict_target_samples(
                posterior=posterior,
                x_test=x_test,
                source_values=source_values,
                stage_values=stage_values,
                mode=mapped_mode,
                seed=self.random_state + (idx * 211),
            )
            point = sample_matrix.mean(axis=0)
            ci = np.percentile(sample_matrix, [2.5, 97.5], axis=0).T
            predictions[target] = point
            intervals[target] = ci
            widths[target] = float(np.mean(ci[:, 1] - ci[:, 0]))

        return {
            "predictions": pd.DataFrame(predictions, index=test_df.index),
            "intervals": intervals,
            "meanIntervalWidth": widths,
            "backend": self.backend_,
        }

    def posterior_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "backend": getattr(self, "backend_", "gibbs_fallback"),
            "pymcAvailable": bool(getattr(self, "pymc_available_", False)),
            "targetBackends": dict(getattr(self, "target_backends_", {})),
            "likelihood": self.likelihood,
            "studentTNu": float(self.student_t_nu),
            "targets": {},
        }
        for target in self.targets_:
            posterior = self.posteriors_[target]
            names = ["intercept", *posterior.feature_names]
            coef_summary = {}
            for idx, name in enumerate(names):
                samples = posterior.beta_samples[:, idx]
                coef_summary[name] = {
                    "mean": float(np.mean(samples)),
                    "sd": float(np.std(samples, ddof=1)),
                    "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                }

            source_summary = {}
            for idx, label in enumerate(posterior.source_labels):
                samples = posterior.source_samples[:, idx]
                source_summary[str(label)] = {
                    "mean": float(np.mean(samples)),
                    "sd": float(np.std(samples, ddof=1)),
                    "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                }

            icc_source = posterior.tau_source2_samples / (
                posterior.tau_source2_samples + posterior.tau_stage2_samples + posterior.sigma2_samples
            )
            icc_stage = posterior.tau_stage2_samples / (
                posterior.tau_source2_samples + posterior.tau_stage2_samples + posterior.sigma2_samples
            )
            summary["targets"][target] = {
                "coefficients": coef_summary,
                "randomInterceptsBySource": source_summary,
                "sigma2Mean": float(np.mean(posterior.sigma2_samples)),
                "tauSource2Mean": float(np.mean(posterior.tau_source2_samples)),
                "tauStage2Mean": float(np.mean(posterior.tau_stage2_samples)),
                "iccSourceMean": float(np.mean(icc_source)),
                "iccStageMean": float(np.mean(icc_stage)),
                "nPosteriorDraws": int(posterior.beta_samples.shape[0]),
            }
        return summary
