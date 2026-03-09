# Bayesian Hierarchical Isolated-Yield Analysis

## Dataset scope
- Full literature rows: 135
- Sequence-resolved isolated-yield rows: 29
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

## Meta heterogeneity
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
- Bayesian RMSE: 33.070
- Bayesian MAE: 30.011
- Bayesian R^2: -1.493
- Bayesian interval coverage: 0.552
- Bayesian mean 95% interval width: 88.945

## Full Bayesian hierarchical fit
- In-sample RMSE: 5.072
- In-sample MAE: 3.770
- In-sample R^2: 0.941
- In-sample coverage: 0.828
- Mean interval width: 12.132
- Posterior ICC mean: 0.897
- Posterior ICC 95% CI: 0.746 to 0.978

Top posterior fixed effects by absolute posterior mean:
- length: mean=-3.302, P(>0)=0.106, 95% CI=-8.822 to 1.880
- max_coupling_difficulty: mean=-2.562, P(>0)=0.167, 95% CI=-7.562 to 3.004
- avg_hydrophobicity: mean=2.071, P(>0)=0.703, 95% CI=-5.880 to 9.381
- total_charge: mean=1.799, P(>0)=0.822, 95% CI=-2.312 to 5.889
- aromatic_ratio: mean=1.516, P(>0)=0.678, 95% CI=-4.996 to 8.169
- hydrophobic_ratio: mean=-0.077, P(>0)=0.487, 95% CI=-5.950 to 5.410

Posterior random intercepts:
- amyloid_hydrophobic_2008: mean=-27.323, 95% CI=-45.613 to -8.322
- ram_spps_2025: mean=35.531, 95% CI=18.232 to 55.460
- stapled_bcl9_2025: mean=-9.262, 95% CI=-29.286 to 10.349
- sulfotyrosine_psc_2016: mean=-9.701, 95% CI=-31.089 to 12.508
- trityl_anchor_2013: mean=14.227, 95% CI=-5.743 to 36.824

## Sensitivity analysis
- all_sequence_isolated: rows=29, sources=5, ridge_rmse=47.218, bayes_rmse=33.070, bayes_coverage=0.552, meta_mean=30.53, I^2=0.986
- purity_observed_only: rows=29, sources=5, ridge_rmse=47.218, bayes_rmse=33.070, bayes_coverage=0.552, meta_mean=30.53, I^2=0.986
- canonical_linear_only: rows=19, sources=3, ridge_rmse=63.272, bayes_rmse=48.735, bayes_coverage=0.632, meta_mean=41.79, I^2=0.991
- exclude_stapled: rows=25, sources=4, ridge_rmse=50.640, bayes_rmse=39.455, bayes_coverage=0.560, meta_mean=36.47, I^2=0.987

## Interpretation
- This model keeps study-specific random intercepts but estimates them in a Bayesian way, which is more stable on tiny multi-study datasets than relying only on asymptotic MixedLM summaries.
- Leave-one-study-out prediction for a new study integrates over an unseen-study random intercept with mean zero, so point predictions stay conservative.
- Posterior ICC close to 1 means study/source effects dominate the shared sequence-level signal.
