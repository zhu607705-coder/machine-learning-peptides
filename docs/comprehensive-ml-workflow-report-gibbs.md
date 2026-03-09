# Comprehensive ML Workflow Report

## 1. Dataset and objective

- Dataset: `data/real/final_purity_yield_literature.csv`
- Initial usable rows with sequence + purity + yield: `126`
- Final rows after cleaning: `113`
- Unique source groups used for grouped CV: `8`

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
| baseline_raw | 126 | 19.394 | 15.007 | -0.073 | 0.222 | 33.583 | 27.760 | -0.257 | 0.214 | 0.048 |
| after_exact_dedup | 126 | 19.394 | 15.007 | -0.073 | 0.222 | 33.583 | 27.760 | -0.257 | 0.214 | 0.048 |
| after_sequence_dedup | 122 | 20.154 | 15.504 | -0.153 | 0.254 | 36.305 | 29.507 | -0.484 | 0.213 | 0.041 |
| after_outlier_removal | 113 | 19.622 | 15.045 | -0.077 | 0.257 | 37.197 | 30.070 | -0.477 | 0.230 | 0.035 |
| after_imputation_and_scaling | 113 | 19.753 | 15.143 | -0.091 | 0.257 | 36.205 | 29.592 | -0.399 | 0.239 | 0.027 |
| after_feature_selection | 113 | 20.160 | 14.960 | -0.137 | 0.274 | 34.928 | 28.969 | -0.302 | 0.230 | 0.044 |
| final_tuned_ensemble | 113 | 19.620 | 14.636 | -0.076 | 0.257 | 35.433 | 29.317 | -0.340 | 0.248 | 0.044 |
| final_tuned_ensemble_with_embeddings | 113 | 20.609 | 15.397 | -0.188 | 0.239 | 34.099 | 29.019 | -0.241 | 0.195 | 0.044 |
| bayesian_calibrated_residual_ensemble | 113 | 18.654 | 15.256 | 0.027 | 0.142 | 34.617 | 29.080 | -0.279 | 0.186 | 0.044 |

## 5. Final tuned stage comparison

- Baseline tuned ensemble purity RMSE: `19.620`
- Embedding tuned ensemble purity RMSE: `20.609`
- Baseline tuned ensemble yield RMSE: `35.433`
- Embedding tuned ensemble yield RMSE: `34.099`
- Purity RMSE delta (embedding - baseline): `0.989`
- Yield RMSE delta (embedding - baseline): `-1.335`

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
  "selector__threshold": "mean",
  "selector__estimator__max_depth": null,
  "regressor__weights": [
    2,
    1,
    2,
    1
  ],
  "regressor__ridge__alpha": 0.1,
  "regressor__rf__n_estimators": 300,
  "regressor__rf__min_samples_leaf": 1,
  "regressor__rf__max_depth": 8,
  "regressor__gbr__n_estimators": 180,
  "regressor__gbr__max_depth": 2,
  "regressor__gbr__learning_rate": 0.05,
  "regressor__et__n_estimators": 500,
  "regressor__et__min_samples_leaf": 1,
  "regressor__et__max_depth": 8
}
```

### Embedding purity model

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

### Embedding yield model

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

- Final stage: `bayesian_calibrated_residual_ensemble`
- Final purity RMSE: `18.654`
- Final purity MAE: `15.256`
- Final purity R²: `0.027`
- Final purity accuracy within ±5%: `0.142`
- Final yield RMSE: `34.617`
- Final yield MAE: `29.080`
- Final yield R²: `-0.279`
- Final yield accuracy within ±10%: `0.186`
- Final joint tolerance accuracy: `0.044`

## 8. Semantic subgroup evaluation

### Purity by stage (embedding tuned ensemble)

| Stage | n | RMSE | MAE | R² | Acc ±5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude_hplc | 65 | 25.799 | 20.683 | -0.743 | 0.169 |
| crude_hplc_214nm | 17 | 13.943 | 12.722 | -2.589 | 0.118 |
| final_product | 13 | 4.132 | 3.375 | -10.827 | 0.769 |
| purified_hplc | 18 | 8.185 | 7.520 | -26.705 | 0.222 |

### Yield by stage (embedding tuned ensemble)

| Stage | n | RMSE | MAE | R² | Acc ±10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude | 76 | 33.586 | 27.643 | -0.675 | 0.250 |
| isolated | 28 | 37.524 | 35.240 | -0.790 | 0.000 |
| recovery | 9 | 26.312 | 21.279 | -1.047 | 0.333 |

### Yield by label basis (embedding tuned ensemble)

| Basis | n | RMSE | MAE | R² | Acc ±10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude | 26 | 45.044 | 40.986 | -3.598 | 0.115 |
| isolated | 28 | 37.524 | 35.240 | -0.790 | 0.000 |
| other | 50 | 25.681 | 20.705 | -0.037 | 0.320 |
| recovery | 9 | 26.312 | 21.279 | -1.047 | 0.333 |

## 9. Feature importance

### Top permutation importances for purity

| Feature | Mean Importance | Std |
| --- | ---: | ---: |
| sequence_norm | 5.2744 | 0.5972 |
| molecular_weight | 4.2653 | 0.5578 |
| yield_stage_canon | 2.8823 | 0.5243 |
| yield_stage | 2.6102 | 0.4945 |
| length | 2.3566 | 0.3569 |
| bulky_ratio | 1.0618 | 0.2528 |
| source_size_log | 1.0315 | 0.2309 |
| avg_hydrophobicity | 0.8803 | 0.2244 |
| longest_hydrophobic_run_norm | 0.6706 | 0.2495 |
| coupling_reagent | 0.6365 | 0.1917 |

### Top permutation importances for yield

| Feature | Mean Importance | Std |
| --- | ---: | ---: |
| purity_stage_canon | 17.2802 | 1.3196 |
| molecular_weight | 13.8910 | 1.4726 |
| length | 7.0283 | 0.8425 |
| sequence_norm | 5.5621 | 0.4535 |
| purity_stage | 1.4507 | 0.2834 |
| avg_hydrophobicity | 1.2631 | 0.2566 |
| aromatic_ratio | 0.8598 | 0.1927 |
| total_charge | 0.6512 | 0.1669 |
| unique_ratio | 0.6206 | 0.2404 |
| c_term_acidic | 0.5353 | 0.1581 |

## 10. Heterogeneity calibration under LOSO

- Raw LOSO purity R²: `-0.059`
- Bayesian calibrated LOSO purity R²: `0.131`
- Bayesian + residual LOSO purity R²: `0.027`
- Raw LOSO yield R²: `-0.180`
- Bayesian calibrated LOSO yield R²: `-0.204`
- Bayesian + residual LOSO yield R²: `-0.279`
- Purity tolerance accuracy (final LOSO): `0.142`
- Yield tolerance accuracy (final LOSO): `0.186`
- Joint tolerance accuracy (final LOSO): `0.044`

### Heterogeneity decomposition (posterior means)

| Target | sigma² | tau_source² | tau_stage² | ICC_source | ICC_stage |
| --- | ---: | ---: | ---: | ---: | ---: |
| purity_target | 198.316 | 32.455 | 60.883 | 0.107 | 0.191 |
| yield_target | 374.669 | 87.514 | 86.480 | 0.148 | 0.137 |

## 11. Conclusions

- The workflow now covers data cleaning, missing-value treatment, scaling, feature selection, ensemble learning, hyperparameter tuning, and grouped cross-validation in one Python pipeline.
- Exact and near-duplicate removal as well as outlier filtering are tracked explicitly, so preprocessing impact is measurable rather than implicit.
- Protein embeddings from ESM2 are evaluated strictly inside grouped folds via PCA reduction, so the before/after comparison does not leak test-fold information.
- Feature selection and permutation importance identify which sequence, stage, process, and embedding-augmented variables are most predictive for purity and yield.
- Grouped cross-validation by source is intentionally strict; it measures cross-paper robustness rather than same-source memorization.
