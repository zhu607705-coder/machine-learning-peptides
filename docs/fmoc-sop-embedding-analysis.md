# Fmoc SOP Embedding Analysis

## Dataset summary
- Rows: 66
- Sources: 3 -> amyloid_hydrophobic_2008, teabags_fmoc_2021, trityl_anchor_2013
- Purity rows: 66
- Yield rows: 66

## Embedding configuration
- Model: facebook/esm2_t6_8M_UR50D
- Device: mps
- Cache dir: /Users/zhuhangcheng/Downloads/星辰计划/机器学习多肽/peptide-synthesis-predictor/data/cache/protein_embeddings
- Raw embedding dimension: 320
- Requested PCA components per fold: 8

## Grouped evaluation
- Purity manual Ridge RMSE: 30.781
- Purity manual + embedding Ridge RMSE: 29.281
- Yield manual Ridge RMSE: 34.040
- Yield manual + embedding Ridge RMSE: 30.867

## Delta
- Purity RMSE delta (embedding - manual): -1.500
- Yield RMSE delta (embedding - manual): -3.173
- Purity MAE delta (embedding - manual): -1.294
- Yield MAE delta (embedding - manual): -2.740

## Interpretation
- The embedding features come from a pretrained ESM2 protein language model and are reduced within each grouped fold before concatenation with manual chemistry features.
- A negative RMSE delta means the embedding-augmented model improves on the manual-feature baseline.
- Given the small SOP-aligned literature subset, any gain should be interpreted as feature-engineering evidence rather than a claim of broad external generalization.
