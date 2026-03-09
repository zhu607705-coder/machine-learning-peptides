# Machine Learning Development Log

Last updated: 2026-03-02

## 1. Project objective

The original UI goal was to predict peptide synthesis quality with two
user-facing outputs:

- predicted purity
- predicted yield

The main engineering constraint was that publicly accessible, sequence-resolved
datasets for final peptide purity and isolated yield are small and fragmented.
Because of that, the project evolved along two parallel tracks:

- a reproducible synthetic sequence-to-purity/yield model for the UI
- a real-data proxy model trained on public fast-flow synthesis measurements

## 2. Initial baseline and refactor

The starting point used fixed handwritten heuristics. That was replaced with a
reproducible feature pipeline so training and inference share the same logic.

Core files:

- `src/lib/peptideCore.ts`
- `python/peptide_core.py`
- `src/lib/neuralModel.ts`
- `python/neural_model.py`

Key feature set used by the synthetic predictor:

- normalized sequence length
- bulky residue ratio
- difficult residue ratio
- hydrophobic ratio
- charged ratio
- breaker ratio
- sulfur ratio
- aspartimide risk
- reagent, topology, temperature, solvent, cleavage scores
- sequence complexity
- longest hydrophobic run
- cysteine cyclization fit

## 3. Synthetic benchmark generation

Because no large public final purity/yield dataset was available at the start,
a chemistry-informed synthetic benchmark was created in
`python/optimize_model.py`.

Design choices:

- balanced scenario generation
- explicit random seeds for reproducibility
- train/validation/test split separated from training seed
- target simulation with chemically meaningful penalties and bonuses

Scenario families:

- `easy_linear`
- `hydrophobic_long`
- `steric_rich`
- `charged_difficult`
- `cysteine_rich`
- `cyclized`

Current synthetic dataset design:

- 1,080 total records
- 180 records per scenario
- split sizes: 750 train / 162 validation / 168 test
- dataset seed: `20260301`
- split seed: `20260302`
- training seed: `20260303`

## 4. Synthetic neural model tuning

The synthetic model is a small fully connected neural network with:

- 16 engineered input features
- 1 hidden layer
- `tanh` activation
- 2 outputs: purity and yield

The optimizer performs a grid search over:

- hidden size
- learning rate
- L2 penalty
- early stopping patience

Best synthetic configuration from
`artifacts/peptide-model-report.json`:

- hidden size: `8`
- learning rate: `0.12`
- L2: `0.0015`
- max epochs: `420`
- patience: `55`

Best synthetic validation metrics:

- purity RMSE: `3.4950`
- yield RMSE: `4.5176`
- combined RMSE: `4.0063`

Best synthetic test metrics:

- purity RMSE: `3.5636`
- yield RMSE: `4.6527`
- combined RMSE: `4.1082`

Exported artifacts:

- `src/lib/modelArtifacts.ts`
- `artifacts/peptide-model-report.json`

Result:

- the app no longer relies on static handwritten weights
- browser inference and Python inference use the same learned parameters
- the synthetic grid search now runs through NumPy-matrix training and parallel
  candidate evaluation, which materially reduces optimization time

## 5. Python parity and CLI inference

To make retraining and inspection easier, the same synthetic model was
implemented in Python.

Files:

- `python/predictor.py`
- `python/predict.py`
- `python/neural_model.py`

Outcome:

- model training can be rerun from Python
- CLI inference can be executed without the frontend
- the shared synthetic artifact stays aligned with the TypeScript runtime

## 6. Real public data pipeline

The next phase moved from synthetic labels to real public measurements using
MIT's `peptimizer` dataset:

- source repo: `learningmatter-mit/peptimizer`
- raw URL:
  `https://raw.githubusercontent.com/learningmatter-mit/peptimizer/master/dataset/data_synthesis/synthesis_data.csv`

Pipeline files:

- `python/real_data.py`
- `python/train_real_model.py`
- `python/predict_real_step.py`

Important scope decision:

- this dataset does not contain final purity or isolated yield
- it contains real step-level proxy targets:
  `first_area`, `first_height`, `first_width`, `first_diff`

To reduce leakage, splitting is grouped by synthesis `serial`.

Current real dataset stats:

- 12,600 usable rows
- 769 unique serials
- mean pre-chain length: `11.88`
- max pre-chain length: `49`

Split sizes:

- train: `8907`
- validation: `1791`
- test: `1902`

Serial split sizes:

- train: `538`
- validation: `115`
- test: `116`

## 7. Real-data architecture search

Three sequence encoders were compared in `python/train_real_model.py`:

- `gru`
- `cnn`
- `hybrid`

The current winning architecture is `gru_residual_small` with:

- architecture: `gru`
- max length: `40`
- embedding dim: `16`
- sequence hidden: `32`
- numeric hidden: `32`
- trunk hidden: `96`
- dropout: `0.12`

Model structure:

- token embedding for pre-chain sequence
- bidirectional GRU sequence encoder
- next-amino-acid embedding
- coupling-agent embedding
- normalized numeric feature branch
- residual MLP trunk
- 4-output regression head

Current real-data result from `artifacts/real-synthesis-report.json`:

- validation combined RMSE: `0.1732`
- test combined RMSE: `0.1608`

Exported artifacts:

- `artifacts/real-synthesis-model.pt`
- `artifacts/real-synthesis-report.json`

## 8. Search for public final purity / yield data

Because the real MIT dataset is only a proxy target, a literature-mining track
was added to find public sequence-resolved final purity/yield examples closer to
the UI target.

Source scan file:

- `data/real/final_purity_yield_sources.md`

High-value primary sources currently incorporated include:

- amyloid beta C-terminal hydrophobic peptides
- peptide alpha-thioester volatilizable support
- RAM SPPS Green Chemistry 2025
- THP backbone protection
- trityl side-chain anchoring
- PDAC PEGylation peptide synthesis
- Tea Bags for Fmoc SPPS
- HOPO protocol paper
- MYC PTM AFPS paper
- recifin A synthesis

## 9. Manual literature dataset construction

The manually curated dataset lives in:

- `data/real/final_purity_yield_literature.csv`
- `data/real/final_purity_yield_literature.md`

Current state:

- 125 source-tracked rows
- 10 primary sources
- 15 rows still missing sequence fields

Current label-stage distribution:

- `crude_hplc`: 72
- `final_product`: 18
- `crude_hplc_214nm`: 17
- `purified_hplc`: 14
- `unknown`: 4

Yield-stage distribution:

- `crude`: 83
- `isolated`: 30
- `recovery`: 12

Important caveat:

- this is still a heterogeneous benchmark
- crude, purified, recovery, and isolated labels are not identical tasks

## 10. Literature baseline modeling

To test whether final purity/yield contains any learnable signal at current
scale, a conservative tabular baseline was added in:

- `python/train_literature_baseline.py`

Model choice:

- ridge regression on normalized engineered sequence features
- optional stage features for mixed-purity experiments
- LOOCV for within-dataset signal
- leave-one-source-out for cross-paper generalization

Recent robustness fixes:

- parsing support for labels like `>95` and `>99`
- additional task splits for `recovery` and `purified_hplc`
- correction of CSV row alignment issues in newly added literature rows

Current results from `artifacts/literature-baseline-report.json`:

`isolated_yield_sequence_only`

- 19 rows
- LOOCV RMSE: `18.69`
- leave-one-source-out R²: `-3.71`

`crude_yield_sequence_only`

- 79 rows
- LOOCV RMSE: `23.71`
- baseline RMSE: `26.16`
- leave-one-source-out R²: `-0.20`

`purity_mixed_sequence_plus_stage`

- 110 rows
- LOOCV RMSE: `15.72`
- baseline RMSE: `18.94`
- leave-one-source-out R²: `-0.204`

Interpretation:

- weak within-source signal exists
- cross-source generalization is still not stable
- the main bottleneck remains data heterogeneity and label mismatch

## 11. Current production posture

There are now three distinct modeling assets in the repository:

1. Synthetic UI model
   Purpose: stable local purity/yield prediction for the app

2. Real fast-flow proxy model
   Purpose: step-level prediction on public experimental measurements

3. Literature baseline
   Purpose: evidence-gathering on whether final purity/yield has enough stable
   signal to justify a supervised model

## 12. Current conclusion

As of 2026-03-02:

- the synthetic sequence-to-purity/yield model is stable and fully deployable
- the real-data fast-flow model is the strongest scientifically grounded neural
  model in the repo, but it predicts step-level proxy signals rather than final
  product quality
- the final purity/yield literature dataset is now large enough to detect weak
  signal, but not yet strong enough to claim robust cross-paper generalization

## 13. Recommended next steps

- continue expanding sequence-resolved isolated-yield sources
- add source-family grouped evaluation beyond leave-one-source-out
- convert `condition_summary` into structured process features
- decide whether the frontend should remain synthetic for purity/yield while
  exposing the real fast-flow model as a separate step-risk panel
