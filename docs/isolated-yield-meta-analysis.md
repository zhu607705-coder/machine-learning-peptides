# Isolated Yield Meta-Style Analysis

## Dataset scope
- Source-tracked rows in full literature table: 135
- Sequence-resolved isolated-yield rows used here: 29
- Unique studies: 5
- Canonical/simple linear rows: 19
- Modified-chemistry rows: 10
- Stapled rows: 4

## Study-level summary
| Study | Year | Family | n | Mean isolated yield | Mean purity | Mean length |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| amyloid_hydrophobic_2008 | 2008 | canonical_or_simple_modified | 14 | 7.81 | 96.21 | 9.79 |
| ram_spps_2025 | 2025 | canonical_or_simple_modified | 3 | 69.33 | 94.33 | 6.33 |
| stapled_bcl9_2025 | 2025 | stapled | 4 | 8.55 | 96.85 | 23.25 |
| sulfotyrosine_psc_2016 | 2016 | sulfated | 6 | 21.95 | 98.78 | 6.17 |
| trityl_anchor_2013 | 2013 | canonical_or_simple_modified | 2 | 49.50 | 59.00 | 6.00 |

## Random-effects meta summary
- Study count: 5
- Fixed-effect pooled mean isolated yield: 17.89
- Random-effect pooled mean isolated yield: 30.53
- Random-effect 95% CI: 13.76 to 47.30
- Tau^2: 353.744
- I^2: 0.986

## Leave-one-study-out predictive comparison
- Grouped Ridge RMSE: 47.218
- Grouped Ridge MAE: 38.657
- Grouped Ridge R^2: -4.082
- MixedLM RMSE: 48.428
- MixedLM MAE: 40.290
- MixedLM R^2: -4.346

## Full mixed-effects fit
- In-sample RMSE: 28.448
- In-sample MAE: 24.714
- In-sample R^2: -0.845
- Random intercept variance: 593.311
- Residual variance: 27.298
- Intraclass correlation (study-level ICC): 0.956

Top fixed effects by absolute standardized coefficient:
- avg_hydrophobicity: 15.649
- hydrophobic_ratio: -6.811
- aromatic_ratio: 5.441
- length: -5.213
- total_charge: 2.891
- max_coupling_difficulty: 0.403

Study random intercepts:
- amyloid_hydrophobic_2008: -35.330
- ram_spps_2025: 36.619
- stapled_bcl9_2025: -8.671
- sulfotyrosine_psc_2016: -7.260
- trityl_anchor_2013: 14.641

## Sensitivity analysis
- all_sequence_isolated: rows=29, sources=5, ridge_rmse=47.218, mixedlm_rmse=48.428, meta_mean=30.53, I^2=0.986
- purity_observed_only: rows=29, sources=5, ridge_rmse=47.218, mixedlm_rmse=48.428, meta_mean=30.53, I^2=0.986
- canonical_linear_only: rows=19, sources=3, ridge_rmse=63.272, mixedlm_rmse=205.393, meta_mean=41.79, I^2=0.991
- exclude_stapled: rows=25, sources=4, ridge_rmse=50.640, mixedlm_rmse=50.748, meta_mean=36.47, I^2=0.987

## Interpretation
- The random-effects meta summary quantifies between-study heterogeneity in mean isolated yield.
- The mixed-effects model treats `source_id` as a random intercept, so chemistry and reporting conventions that cluster within studies are partially separated from sequence-level signals.
- Sensitivity subsets show whether the sequence-yield relationship survives after excluding specialized chemistries such as stapled or sulfated peptides.
