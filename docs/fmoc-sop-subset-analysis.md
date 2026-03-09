# Fmoc SOP Subset Analysis

## Filter definition
- Keep rows whose `condition_summary` explicitly contains `Fmoc`.
- Keep linear/blank-topology rows only.
- Exclude rows and sources containing: boc, afps, stapled, sulfotyrosine, sulfated, peg23, pegyl, palmitoyl, hopo, solvent-less, ram , ligation, fragment synthesis, microwave
- Keep only sequence-resolved rows after sequence normalization.

## Dataset summary
- Rows: 66
- Sources: 3 -> amyloid_hydrophobic_2008, teabags_fmoc_2021, trityl_anchor_2013
- Yield stages: {'isolated': 16, 'crude': 50}
- Purity stages: {'final_product': 14, 'crude_hplc': 52}

## Source summary
| Source | n | Mean length | Mean purity | Mean yield |
| --- | ---: | ---: | ---: | ---: |
| amyloid_hydrophobic_2008 | 14 | 9.79 | 96.21 | 7.81 |
| teabags_fmoc_2021 | 50 | 13.98 | 67.11 | 54.08 |
| trityl_anchor_2013 | 2 | 6.00 | 59.00 | 49.50 |

## Grouped evaluation
- Purity grouped Ridge RMSE: 30.781
- Purity mean-baseline RMSE: 30.621
- Yield grouped Ridge RMSE: 34.040
- Yield mean-baseline RMSE: 47.106

## Within-source centered correlations
### Yield
- total_charge: 0.066
- length: -0.063
- avg_hydrophobicity: -0.063
- max_coupling_difficulty: 0.053
- molecular_weight: -0.049
- longest_hydrophobic_run_norm: -0.042
- hydrophobic_ratio: -0.015
- aromatic_ratio: 0.000

### Purity
- longest_hydrophobic_run_norm: -0.116
- avg_hydrophobicity: -0.080
- hydrophobic_ratio: -0.066
- length: 0.064
- total_charge: 0.064
- aromatic_ratio: -0.061
- molecular_weight: 0.056
- max_coupling_difficulty: -0.007

## Interpretation
- Relative to the SOP-like mean baseline, grouped Ridge improves yield prediction noticeably, indicating that shrinking the chemistry space toward the current Fmoc SOP does recover some stable signal.
- Purity remains difficult because this subset still mixes `crude_hplc` and `final_product` semantics across different studies.
- After centering by source mean, no single sequence feature dominates; this suggests the current SOP-aligned literature subset is directionally useful but still too small for strong mechanistic claims.
