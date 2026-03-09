# autoresearch program

This repository uses `autoresearch` as a first-class research loop for peptide literature modeling. The goal is not generic nanoGPT optimization; the goal is to improve the **strict semantic head** pipeline for final peptide synthesis results while preserving the credibility controls already established in this codebase.

## Files in scope

Read these files before making any change:

- `README.md`
- `autoresearch/prepare.py`
- `autoresearch/train.py`
- `python/comprehensive_ml_workflow.py`
- `python/source_head_calibration.py`
- `docs/comprehensive-ml-workflow-report.md`

## Editable surface

Only this file is meant to be iterated by the autoresearch loop:

- `autoresearch/train.py`

Treat `autoresearch/prepare.py` as fixed orchestration and reporting glue.

## System contract

The autoresearch loop must respect the modeling contract already established in this repository:

1. `purity` and `yield` stay separated.
2. `stage` and `basis` semantics must not be remixed into a single head.
3. `source_id` heterogeneity controls must remain enabled.
4. `predictionPolicy` and reject / low-confidence routing must remain enabled.
5. Evaluation for result heads must remain LOSO or grouped by source. Never use random row splits.

## Objective

Primary objective, in order:

1. Improve `yield_target / isolated|isolated` LOSO `R²`
2. Reduce `yield_target / isolated|isolated` LOSO `RMSE`
3. Increase the number of accepted high-confidence heads
4. Improve weighted head summary only if the primary head does not regress

Current baseline to beat:

- `strictSemanticFilter.outputRows = 185`
- `yield_target / isolated|isolated`
  - `rows = 63`
  - `sources = 10`
  - `RMSE = 24.549335`
  - `R² = 0.092445`
  - `accuracyWithinTolerance = 0.126984`
  - `decision = low`
- `acceptedHighConfidenceHeads = 0`

## Allowed moves

Inside `autoresearch/train.py`, you may:

1. Tune source-drift thresholds and compatible-subset logic
2. Add source-aware weighting, balancing, or filtering logic
3. Tune strict-head training hyperparameters
4. Change which strict-head subsets are emphasized during training
5. Adjust keep / discard policy used for experimental selection

## Forbidden moves

Do not:

1. Modify raw literature CSV files
2. Disable semantic head splitting
3. Disable source-aware cleaning or calibration
4. Disable prediction rejection / confidence policy
5. Recast step-proxy metrics as evidence for final purity or yield performance

## Baseline command

```bash
PYTHONPATH=python archive/tooling/.venv-pymc/bin/python autoresearch/train.py > autoresearch/run.log 2>&1
```

## Fast import-only check

Use this before a long run when changing entrypoint plumbing:

```bash
AUTORESEARCH_IMPORT_ONLY=1 PYTHONPATH=python archive/tooling/.venv-pymc/bin/python autoresearch/train.py
```

## Keep / discard rule

Keep an experiment only if at least one of the following is true:

1. `isolated|isolated` `R²` increases and RMSE does not materially worsen
2. `isolated|isolated` RMSE decreases and `R²` does not materially worsen
3. `acceptedHighConfidenceHeads` increases

If weighted aggregate metrics improve while the primary head regresses, discard the experiment.
