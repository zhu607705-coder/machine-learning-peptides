# 2026-03-07 文件整理清单

## 模型迁移

- 默认 annotated 模型迁入 `models/annotated/`
- 默认 deploy 模型迁入 `models/deploy/`
- 顶层旧 baseline 模型迁入 `models/legacy/root_baseline/`
- `artifacts/annotated_training/models/` 迁入 `models/legacy/annotated_training/`
- 重复模型 `yield_cleaned_gradientboosting.joblib` 迁入 `archive/recovery/duplicate_models/`

## 报告与导出

- `报告/` 只保留正式学习报告 `.md/.docx/.pdf`
- `output/doc` 与 `output/pdf` 的重复导出迁入 `archive/output/export_duplicates/`
- `报告/` 中的中间导出和技术副本迁入 `archive/reports/legacy_exports/`
- 旧修正版提交件迁入 `archive/reports/legacy_submission/`

## 实验产物

- `artifacts/annotated_grouped_review/` 迁入 `artifacts/experiments/annotated_grouped_review/`
- `reports/` 中训练文本与统计迁入 `artifacts/reports/training/`
- `artifacts/annotated_training/training_summary.json` 迁入 `artifacts/reports/annotated_training/`
- 当前 deploy 报告迁入 `artifacts/reports/deploy/`

## 脚本与缓存

- `complete_experiment7.py`
- `generate_experiment7_pdf.py`

以上一次性脚本迁入 `scripts/legacy/`

已清理：

- `.DS_Store`
- `.pytest_cache/`
- `python/.pytest_cache/`
- `.playwright-cli/`
- `dist/`
- `tmp/`
- `python/__pycache__/`
- `tests/__pycache__/`

## 待后续重建或迁移

- `.venv-pymc/`
- `node_modules/`

这两项仍可用，但后续会迁入 `archive/tooling/` 或按需重建。
