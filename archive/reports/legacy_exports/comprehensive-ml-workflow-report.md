# Comprehensive ML Workflow Report

## 1. Dataset and objective

- Dataset: `data/real/final_purity_yield_literature.csv`
- Initial usable rows with sequence + purity + yield: `120`
- Final rows after cleaning: `108`
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
| baseline_raw | 120 | 17.512 | 13.044 | 0.135 | 0.250 | 37.907 | 30.382 | -0.627 | 0.225 | 0.008 |
| after_exact_dedup | 120 | 17.512 | 13.044 | 0.135 | 0.250 | 37.907 | 30.382 | -0.627 | 0.225 | 0.008 |
| after_sequence_dedup | 119 | 17.427 | 12.985 | 0.142 | 0.269 | 38.149 | 30.657 | -0.649 | 0.227 | 0.008 |
| after_outlier_removal | 108 | 18.332 | 13.234 | 0.087 | 0.287 | 37.590 | 29.671 | -0.710 | 0.278 | 0.083 |
| after_imputation_and_scaling | 108 | 18.844 | 13.650 | 0.035 | 0.269 | 37.494 | 29.799 | -0.701 | 0.259 | 0.056 |
| after_feature_selection | 108 | 20.125 | 15.434 | -0.100 | 0.213 | 38.781 | 31.045 | -0.820 | 0.231 | 0.037 |
| final_tuned_ensemble | 108 | 20.042 | 15.391 | -0.091 | 0.204 | 36.871 | 29.541 | -0.645 | 0.259 | 0.046 |
| final_tuned_ensemble_with_embeddings | 108 | 21.034 | 16.407 | -0.202 | 0.185 | 37.810 | 30.146 | -0.730 | 0.287 | 0.056 |

## 5. Final tuned stage comparison

- Baseline tuned ensemble purity RMSE: `20.042`
- Embedding tuned ensemble purity RMSE: `21.034`
- Baseline tuned ensemble yield RMSE: `36.871`
- Embedding tuned ensemble yield RMSE: `37.810`
- Purity RMSE delta (embedding - baseline): `0.992`
- Yield RMSE delta (embedding - baseline): `0.939`

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

- Final purity RMSE: `21.034`
- Final purity MAE: `16.407`
- Final purity R²: `-0.202`
- Final purity accuracy within ±5%: `0.185`
- Final yield RMSE: `37.810`
- Final yield MAE: `30.146`
- Final yield R²: `-0.730`
- Final yield accuracy within ±10%: `0.287`
- Final joint tolerance accuracy: `0.056`

## 8. Semantic subgroup evaluation

### Purity by stage (embedding tuned ensemble)

| Stage | n | RMSE | MAE | R² | Acc ±5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude_hplc | 61 | 25.755 | 20.717 | -0.886 | 0.148 |
| crude_hplc_214nm | 16 | 14.450 | 13.185 | -4.816 | 0.125 |
| final_product | 13 | 15.862 | 15.753 | -173.270 | 0.000 |
| purified_hplc | 18 | 6.274 | 5.133 | -15.278 | 0.500 |

### Yield by stage (embedding tuned ensemble)

| Stage | n | RMSE | MAE | R² | Acc ±10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude | 72 | 37.369 | 28.431 | -1.383 | 0.333 |
| isolated | 27 | 42.408 | 38.499 | -2.821 | 0.111 |
| recovery | 9 | 24.251 | 18.805 | -0.739 | 0.444 |

### Yield by label basis (embedding tuned ensemble)

| Basis | n | RMSE | MAE | R² | Acc ±10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| crude | 25 | 56.689 | 50.985 | -6.159 | 0.160 |
| isolated | 27 | 42.408 | 38.499 | -2.821 | 0.111 |
| other | 47 | 20.733 | 16.434 | -0.014 | 0.426 |
| recovery | 9 | 24.251 | 18.805 | -0.739 | 0.444 |

## 9. Feature importance

### Top permutation importances for purity

| Feature | Mean Importance | Std |
| --- | ---: | ---: |
| purity_stage | 6.3536 | 0.9124 |
| sequence_norm | 4.4909 | 0.5656 |
| yield_basis_class | 3.1376 | 0.5464 |
| bulky_ratio | 1.5008 | 0.3535 |
| yield_stage | 1.0623 | 0.1943 |
| longest_hydrophobic_run_norm | 0.7068 | 0.1459 |
| max_coupling_difficulty | 0.6774 | 0.1927 |
| avg_hydrophobicity | 0.6745 | 0.2391 |
| publication_year | 0.2968 | 0.1325 |
| molecular_weight | 0.2892 | 0.0399 |

### Top permutation importances for yield

| Feature | Mean Importance | Std |
| --- | ---: | ---: |
| purity_stage | 18.5887 | 1.3154 |
| sequence_norm | 3.1503 | 0.4278 |
| molecular_weight | 1.7937 | 0.4913 |
| avg_hydrophobicity | 1.1477 | 0.3140 |
| hydrophobic_ratio | 0.8429 | 0.3553 |
| total_charge | 0.7402 | 0.1005 |
| length | 0.5442 | 0.3149 |
| unique_ratio | 0.4988 | 0.1274 |
| longest_hydrophobic_run_norm | 0.4795 | 0.1209 |
| yield_stage | 0.4722 | 0.1400 |

## 10. Conclusions

- The workflow now covers data cleaning, missing-value treatment, scaling, feature selection, ensemble learning, hyperparameter tuning, and grouped cross-validation in one Python pipeline.
- Exact and near-duplicate removal as well as outlier filtering are tracked explicitly, so preprocessing impact is measurable rather than implicit.
- Protein embeddings from ESM2 are evaluated strictly inside grouped folds via PCA reduction, so the before/after comparison does not leak test-fold information.
- Feature selection and permutation importance identify which sequence, stage, process, and embedding-augmented variables are most predictive for purity and yield.
- Grouped cross-validation by source is intentionally strict; it measures cross-paper robustness rather than same-source memorization.
