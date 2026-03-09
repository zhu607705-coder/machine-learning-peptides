# 仓库结构约定

本仓库按“源码、数据、模型、实验产物、正式交付、回收区”六层管理。

## 目录职责

- `src/`、`python/`、`tests/`、`scripts/`
  - 仅放源码、训练入口、测试和长期维护脚本。
- `data/`
  - `raw/`、`real/`、`external/` 保存正式数据。
  - `cache/` 只允许保存可重建缓存，不允许作为正式结果输入。
- `models/`
  - `annotated/`：annotated 数据系默认模型。
  - `deploy/`：默认部署模型。
  - `legacy/`：历史模型与非默认副本。
- `artifacts/`
  - `experiments/`：实验中间结果、清洗数据、评估快照。
  - `reports/`：训练报告、部署报告、自动导出统计。
- `docs/`
  - 可编辑源文档、方法说明、技术报告、图表说明。
- `报告/`
  - 正式交付版 `.md/.docx/.pdf`。
- `archive/`
  - 本次整理移出的旧导出、旧脚本、重复模型、工具目录。

## 约束

- 默认推理入口不得再硬编码引用 `artifacts/.../models`。
- `artifacts/` 中的文件默认视为实验记录，不作为主入口唯一依赖。
- `报告/` 不放技术草稿与素材索引。
- `output/`、`tmp/`、缓存目录不放唯一真本。

## 当前默认模型

- `models/annotated/purity_full_sourcerouted.joblib`
- `models/annotated/yield_val_full_sourcerouted.joblib`
- `models/deploy/final-deploy-model.pt`
