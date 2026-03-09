# Conditional GRU Attention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a conditional interaction architecture that predicts target deltas and benchmark it against the current GRU baseline.

**Architecture:** Reuse the existing dataset and training pipeline, but introduce a new `conditional_gru_attention` branch inside the model. This branch uses sequence-level GRU states, a context-conditioned attention query, target-specific heads, and `prev_*` offsets for delta prediction.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Add failing tests for the new architecture

**Files:**
- Modify: `tests/test_train_real_huber_delta.py`

**Step 1: Write the failing test**

Assert that:
- candidate config expansion includes `conditional_gru_attention`
- seed offsets stay stable
- forward pass returns `[batch, 4]`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python .venv-pymc/bin/python -m pytest -q tests/test_train_real_huber_delta.py`

**Step 3: Write minimal implementation**

Add config plumbing and model path.

**Step 4: Run test to verify it passes**

Run the same pytest command.

### Task 2: Implement delta-target dataset and evaluation support

**Files:**
- Modify: `python/train_real_model.py`

**Step 1: Extend dataset**

Add optional `include_prev_targets` and `predict_delta` config-driven behavior.

**Step 2: Extend evaluation**

If the model predicts deltas, invert normalization and add `prev_*` offsets back before metric computation.

### Task 3: Implement conditional interaction model branch

**Files:**
- Modify: `python/train_real_model.py`

**Step 1: Add architecture**

Use `BiGRU` token states, condition query projection, additive attention, shared trunk, and target-specific heads.

**Step 2: Keep existing architectures intact**

Avoid changing current GRU / CNN / hybrid / rnn_attention behavior.

### Task 4: Run benchmark and collect evidence

**Files:**
- Use existing benchmark scripts
- Update artifacts under `artifacts/`

**Step 1: Run focused grouped benchmark**

Compare `gru` vs `conditional_gru_attention`.

**Step 2: Save outputs**

Keep JSON/Markdown artifacts for direct report citation.

### Task 5: Update study reports

**Files:**
- Modify: `docs/多肽合成机器学习学习报告.md`
- Modify: `/Users/zhuhangcheng/Downloads/星辰计划/报告/多肽合成机器学习学习报告.md`

**Step 1: Add a new subsection**

State what changed, why it was tried, and what the verified results show.

**Step 2: Reference artifact paths and exact metrics**

Only write conclusions supported by fresh benchmark output.
