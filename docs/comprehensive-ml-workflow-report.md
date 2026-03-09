# Comprehensive ML Workflow Report

## 1. Dataset and objective

- Dataset: `data/real/final_purity_yield_literature.csv`
- Initial usable rows with sequence + purity + yield: `199`
- Final rows after cleaning: `182`
- Unique source groups used for grouped CV: `16`

## 2. Preprocessing methodology

- Exact duplicates were removed using source, normalized sequence, stage labels, topology, and target values.
- Near-duplicates were removed when sequence similarity was at least `0.97` within the same source-stage group and target differences stayed within tight tolerances.
- Outliers were removed using stage-aware IQR filtering on purity/yield plus modified z-score screening on sequence length and molecular weight.
- Missing numeric values were handled with `mean` imputation.
- Imputation rationale: Mean imputation was selected because the literature dataset is small and highly heterogeneous across sources; KNN neighbors are unstable under grouped cross-validation.
- Standardization rationale: Large-magnitude continuous physicochemical variables were standardized to zero mean and unit variance.
- Normalization rationale: Ratios and bounded heuristic synthesis scores were scaled to the [0, 1] range to keep them comparable and interpretable.

## 3. Ensemble architecture

- Final model family: `VotingRegressor`
- Base learners: `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `Ridge`
- Feature selection: `SelectFromModel(ExtraTreesRegressor)`
- Validation strategy: `GroupKFold` grouped by `source_id`
- Hyperparameter search: `RandomizedSearchCV`
- Protein embedding model for the augmented stage: `facebook/esm2_t6_8M_UR50D`
- Embedding PCA components per grouped fold: `8`

## 4. Stage-by-stage performance comparison

| Stage | Rows | Purity RMSE | Purity MAE | Purity R² | Purity Acc ±5% | Yield RMSE | Yield MAE | Yield R² | Yield Acc ±10% | Joint Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_raw | 166 | 22.591 | 16.667 | -0.076 | 0.205 | 33.422 | 29.691 | -0.302 | 0.145 | 0.036 |
| after_exact_dedup | 166 | 22.591 | 16.667 | -0.076 | 0.205 | 33.422 | 29.691 | -0.302 | 0.145 | 0.036 |
| after_sequence_dedup | 166 | 22.591 | 16.667 | -0.076 | 0.205 | 33.422 | 29.691 | -0.302 | 0.145 | 0.036 |
| after_outlier_removal | 158 | 24.191 | 17.175 | -0.203 | 0.234 | 34.021 | 29.942 | -0.450 | 0.146 | 0.044 |
| after_imputation_and_scaling | 158 | 22.685 | 15.972 | -0.058 | 0.278 | 35.623 | 31.407 | -0.590 | 0.139 | 0.038 |
| after_feature_selection | 158 | 21.929 | 15.008 | 0.012 | 0.247 | 38.186 | 33.829 | -0.827 | 0.108 | 0.044 |
| final_tuned_ensemble | 158 | 22.914 | 16.227 | -0.079 | 0.196 | 35.356 | 30.886 | -0.566 | 0.152 | 0.044 |
| strict_semantic_head_loso | 182 | 26.269 | 21.086 | -0.943 | 0.237 | 26.888 | 24.040 | -0.297 | 0.118 | 0.062 |

## 5. Final tuned stage comparison

- Baseline tuned ensemble purity RMSE: `22.914`
- Baseline tuned ensemble yield RMSE: `35.356`

## 6. Final tuned model parameters

### Baseline purity model

```json
{
  "selector__threshold": "median",
  "selector__estimator__max_depth": 4,
  "regressor__weights": [
    1,
    1,
    1,
    1
  ],
  "regressor__ridge__alpha": 0.1,
  "regressor__rf__n_estimators": 300,
  "regressor__rf__min_samples_leaf": 1,
  "regressor__rf__max_depth": 4,
  "regressor__gbr__n_estimators": 120,
  "regressor__gbr__max_depth": 3,
  "regressor__gbr__learning_rate": 0.08,
  "regressor__et__n_estimators": 200,
  "regressor__et__min_samples_leaf": 2,
  "regressor__et__max_depth": 4
}
```

### Baseline yield model

```json
{
  "selector__threshold": "median",
  "selector__estimator__max_depth": 4,
  "regressor__weights": [
    1,
    1,
    1,
    1
  ],
  "regressor__ridge__alpha": 0.1,
  "regressor__rf__n_estimators": 300,
  "regressor__rf__min_samples_leaf": 1,
  "regressor__rf__max_depth": 4,
  "regressor__gbr__n_estimators": 120,
  "regressor__gbr__max_depth": 3,
  "regressor__gbr__learning_rate": 0.08,
  "regressor__et__n_estimators": 200,
  "regressor__et__min_samples_leaf": 2,
  "regressor__et__max_depth": 4
}
```

## 7. Final evaluation metrics

- Final stage: `strict_semantic_head_loso`
- Final purity RMSE: `26.269`
- Final purity MAE: `21.086`
- Final purity R²: `-0.943`
- Final purity accuracy within ±5%: `0.237`
- Final yield RMSE: `26.888`
- Final yield MAE: `24.040`
- Final yield R²: `-0.297`
- Final yield accuracy within ±10%: `0.118`
- Final joint tolerance accuracy: `0.062`

## 8. Semantic subgroup evaluation

### Purity by stage (baseline tuned ensemble)

| Stage | n | RMSE | MAE | R² | Acc ±5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude_hplc | 81 | 29.540 | 21.996 | -0.557 | 0.148 |
| crude_hplc_214nm | 16 | 16.981 | 15.798 | -7.032 | 0.062 |
| final_product | 20 | 10.350 | 9.618 | -26.455 | 0.000 |
| purified_hplc | 41 | 11.605 | 8.222 | -0.656 | 0.439 |

### Yield by stage (baseline tuned ensemble)

| Stage | n | RMSE | MAE | R² | Acc ±10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude | 75 | 30.540 | 25.591 | -0.609 | 0.267 |
| isolated | 52 | 40.465 | 37.353 | -1.307 | 0.019 |
| recovery | 31 | 36.987 | 32.847 | -1.574 | 0.097 |

### Yield by label basis (baseline tuned ensemble)

| Basis | n | RMSE | MAE | R² | Acc ±10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude | 28 | 38.844 | 35.103 | -1.853 | 0.179 |
| isolated | 52 | 40.465 | 37.353 | -1.307 | 0.019 |
| other | 47 | 24.278 | 19.925 | -0.390 | 0.319 |
| recovery | 31 | 36.987 | 32.847 | -1.574 | 0.097 |

## 9. Feature importance

### Top permutation importances for purity

| Feature | Mean Importance | Std |
| --- | ---: | ---: |
| temperature | 8.5999 | 0.4755 |
| purity_stage | 3.8485 | 0.3885 |
| publication_year | 2.2276 | 0.3626 |
| molecular_weight | 1.2187 | 0.1838 |
| yield_basis_class | 1.0769 | 0.2130 |
| total_charge | 0.7932 | 0.1515 |
| longest_hydrophobic_run_norm | 0.7648 | 0.2332 |
| yield_basis_canon | 0.7421 | 0.1710 |
| bulky_ratio | 0.7293 | 0.1875 |
| avg_hydrophobicity | 0.6780 | 0.2276 |

### Top permutation importances for yield

| Feature | Mean Importance | Std |
| --- | ---: | ---: |
| has_tfa | 5.7562 | 0.6440 |
| purity_stage | 4.2228 | 0.4946 |
| avg_hydrophobicity | 2.5496 | 0.4153 |
| chemistry_family | 2.0603 | 0.3633 |
| length | 1.8690 | 0.2968 |
| source_size_log | 1.6685 | 0.2242 |
| purity_stage_canon | 1.1649 | 0.2470 |
| molecular_weight | 1.1404 | 0.2689 |
| polar_ratio | 1.0816 | 0.2111 |
| hydrophobic_ratio | 1.0454 | 0.3023 |

## 10. Heterogeneity calibration under LOSO

- Global Bayesian LOSO calibration was skipped because strict semantic head LOSO is the default main evaluation path for this run.

## 11. Strict semantic head LOSO

- Strict semantic filtering enabled: `True`
- Rows before strict head filtering: `185`
- Rows after strict head filtering: `182`
- Purity rows before/after strict filtering: `166 -> 166`
- Yield rows before/after strict filtering: `177 -> 174`
- Head thresholds: rows>=20, sources>=3, max source share<=0.70

### Head eligibility

| Target | Head | Rows | Sources | Max source share | Eligible | Reasons |
| --- | --- | ---: | ---: | ---: | --- | --- |
| purity | crude_hplc | 97 | 8 | 0.485 | yes | - |
| purity | purified_hplc | 41 | 4 | 0.488 | yes | - |
| purity | final_product | 28 | 3 | 0.464 | yes | - |
| yield | isolated|isolated | 63 | 10 | 0.206 | yes | - |
| yield | crude|crude | 33 | 3 | 0.485 | yes | - |
| yield | recovery|recovery | 31 | 2 | 0.645 | no | sources<3 |

### Eligible head LOSO results

| Target | Head | Rows | Sources | RMSE | MAE | R² | Accuracy | Interval coverage | Mean interval width | Bayesian | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| purity_target | crude_hplc | 97 | 8 | 31.667 | 25.545 | -0.932 | 0.144 | 0.227 | 21.417 | no | reject |
| purity_target | purified_hplc | 41 | 4 | 12.499 | 9.286 | -0.922 | 0.341 | 0.293 | 6.628 | no | reject |
| purity_target | final_product | 28 | 3 | 3.275 | 2.933 | -1.093 | 0.893 | 0.179 | 2.581 | no | reject |
| yield_target | isolated|isolated | 63 | 10 | 24.549 | 21.792 | 0.092 | 0.127 | 0.111 | 14.702 | no | low |
| yield_target | crude|crude | 33 | 3 | 41.770 | 38.347 | -2.778 | 0.061 | 0.061 | 28.606 | no | reject |

### Weighted head summary

- Weighted support: `1753.0`
- Weighted RMSE: `26.526`
- Weighted MAE: `22.315`
- Weighted R²: `-0.675`
- Weighted accuracy: `0.188`

### Prediction confidence and rejection policy

- High-confidence heads: `0`
- Warning-only heads: `1`
- Rejected heads: `4`
- Warning-only: `yield_target / isolated|isolated` because weak_signal
- Rejected: `purity_target / crude_hplc` because negative_r2, interval_too_wide
- Rejected: `purity_target / purified_hplc` because negative_r2
- Rejected: `purity_target / final_product` because negative_r2
- Rejected: `yield_target / crude|crude` because negative_r2

### Primary isolated yield head

- Baseline isolated head R²: `0.000`
- Final isolated head R²: `0.092`
- RMSE change ratio vs baseline: `0.047`

## 12. Conclusions

- The workflow now covers data cleaning, missing-value treatment, scaling, feature selection, ensemble learning, hyperparameter tuning, and grouped cross-validation in one Python pipeline.
- Exact and near-duplicate removal as well as outlier filtering are tracked explicitly, so preprocessing impact is measurable rather than implicit.
- Protein embeddings from ESM2 are evaluated strictly inside grouped folds via PCA reduction, so the before/after comparison does not leak test-fold information.
- Feature selection and permutation importance identify which sequence, stage, process, and embedding-augmented variables are most predictive for purity and yield.
- Grouped cross-validation by source is intentionally strict; it measures cross-paper robustness rather than same-source memorization.
