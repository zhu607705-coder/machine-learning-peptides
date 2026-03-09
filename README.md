<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Peptide Synthesis Predictor

多肽合成预测系统，当前按“源码 / 数据 / 模型 / 实验产物 / 正式报告 / 回收区”分层管理。

## 当前目录约定

- `src/`、`python/`、`tests/`、`scripts/`: 源码、训练入口、测试、长期维护脚本
- `data/`: 正式数据集与缓存
- `models/annotated/`: 当前默认 annotated 结果级模型
- `models/deploy/`: 当前默认步骤级部署模型
- `models/legacy/`: 历史模型和兼容副本
- `artifacts/experiments/`: 实验中间结果和清洗数据
- `artifacts/reports/`: 自动生成的训练/部署报告
- `docs/`: 可编辑技术文档与学习报告源稿
- `报告/`: 正式提交版 Markdown / DOCX / PDF
- `archive/`: 被整理出的旧导出、旧脚本、重复副本、工具目录

详细约定见 [docs/repo-structure.md](docs/repo-structure.md)。

## 环境准备

这次整理后，工作区内不再保留可执行依赖目录；如果要重新运行命令，需要先重建环境。

### 前端

```bash
npm install
```

### Python

仓库当前没有 `requirements.txt`，推荐直接新建虚拟环境并安装核心依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn torch joblib markdown beautifulsoup4 python-docx reportlab pymc arviz pytest
```

如果只是临时复用上次整理前的环境，可参考 `archive/tooling/.venv-pymc/`，但不建议把它作为长期默认环境。

## 前端运行

```bash
npm run dev
npm run build
npm run preview
npm run lint
```

## 当前默认模型

### 结果级 annotated 默认模型

- purity: `models/annotated/purity_full_sourcerouted.joblib`
- yield: `models/annotated/yield_val_full_sourcerouted.joblib`

对应推理入口：

```bash
python python/predict.py \
  --sequence "H-Gly-Ala-Val-Leu-Ile-OH" \
  --topology "Linear" \
  --coupling-reagent "HATU" \
  --solvent "DMF" \
  --temperature "Room Temperature" \
  --cleavage-time "2 hours"
```

### 步骤级 deploy 默认模型

- model: `models/deploy/final-deploy-model.pt`
- weights: `models/deploy/final-deploy-weights.json`

对应推理入口：

```bash
python python/predict_real_step.py \
  --pre-chain CKLFSG \
  --amino-acid I
```

## 主要训练命令

### 1. 结果级 annotated grouped 复核与重训

输出：
- 模型写入 `models/annotated/`
- 报告写入 `artifacts/experiments/annotated_grouped_review/`

```bash
python python/train_annotated_grouped_multiseed.py
```

主要产物：
- `models/annotated/purity_full_sourcerouted.joblib`
- `models/annotated/yield_val_full_sourcerouted.joblib`
- `artifacts/experiments/annotated_grouped_review/annotated_grouped_multiseed_review.json`

### 2. 步骤级多种子部署训练

输出：
- 模型写入 `models/deploy/`
- 报告写入 `artifacts/reports/deploy/`

```bash
python python/train_data_first_gru_multiseed.py
```

主要产物：
- `models/deploy/final-deploy-model.pt`
- `models/deploy/final-deploy-weights.json`
- `artifacts/reports/deploy/final-deploy-report.json`

### 3. 综合文献工作流

```bash
python python/comprehensive_ml_workflow.py
```

输出：
- `artifacts/comprehensive-ml-workflow-report.json`
- `docs/comprehensive-ml-workflow-report.md`

### 4. 架构与基准实验

```bash
python python/train_real_model.py
python python/train_data_first_gru.py
python python/architecture_grouped_cv_benchmark.py
python python/rnn_attention_grid_benchmark.py
python python/optimize_model.py
```

### 5. autoresearch 有机集成实验

`autoresearch/` 现在是仓库内一级研究子系统，不再以 `python/autoresearch_train.py` 这种外置 wrapper 作为主入口。

固定文件：
- `autoresearch/prepare.py`
- `program.md`

唯一实验入口：
- `autoresearch/train.py`

快速检查入口是否可导入：

```bash
AUTORESEARCH_IMPORT_ONLY=1 PYTHONPATH=python archive/tooling/.venv-pymc/bin/python autoresearch/train.py
```

运行一轮基线：

```bash
PYTHONPATH=python archive/tooling/.venv-pymc/bin/python autoresearch/train.py > autoresearch/run.log 2>&1
```

查看结果：

```bash
tail -n 40 autoresearch/run.log
cat artifacts/autoresearch/latest-run.json
tail -n 5 results.tsv
```

## 三层任务栈训练入口

### result_head

```bash
python -m python.pipelines.train diagnose \
  --input=data/real/final_purity_yield_literature.csv

python -m python.pipelines.train result_head \
  --input=data/real/final_purity_yield_literature.csv \
  --eval=loso

python -m python.pipelines.train result_head \
  --input=data/real/final_purity_yield_literature.csv \
  --head=final_product_isolated_isolated_mass \
  --eval=groupkfold
```

### step_proxy

```bash
python -m python.pipelines.train step_proxy \
  --input=data/external/peptimizer_cpp_predictor_dataset.csv \
  --target=first_area \
  --loss=mse
```

## 数据处理与分析

### 数据预处理

```bash
python python/enhance_dataset.py \
  --input=data/real/final_purity_yield_literature.csv \
  --output=data/enhanced_training_data.csv

python python/data_preprocessor.py \
  --input=data/real/final_purity_yield_literature.csv \
  --output=data/processed_literature_data.csv

python python/download_public_peptide_data.py
```

### 元分析与贝叶斯分析

```bash
python python/meta_isolated_yield_analysis.py
python python/bayesian_isolated_yield_analysis.py
python python/visualize_bayes_likelihood_ablation.py
python python/visualize_gibbs_vs_pymc.py
```

### Fmoc SOP 相关分析

```bash
python python/fmoc_sop_embedding_analysis.py
python python/fmoc_sop_subset_analysis.py
```

## 报告导出

学习报告源稿与交付件已分离：

- 源稿：`docs/多肽合成机器学习学习报告.md`
- 正式交付：`报告/多肽合成机器学习学习报告.md`
- 导出件：`报告/多肽合成机器学习学习报告.docx`
- 导出件：`报告/多肽合成机器学习学习报告.pdf`

重新导出命令：

```bash
python python/export_report_documents.py
```

## 测试

```bash
pytest
pytest python/tests/test_architecture.py -v
pytest tests/test_predictor_annotated.py -v
pytest tests/test_predict_real_step.py -v
```

## 常用输出路径

### 模型

- `models/annotated/purity_full_sourcerouted.joblib`
- `models/annotated/yield_val_full_sourcerouted.joblib`
- `models/deploy/final-deploy-model.pt`
- `models/deploy/final-deploy-weights.json`

### 实验与报告

- `artifacts/experiments/annotated_grouped_review/annotated_grouped_multiseed_review.json`
- `artifacts/reports/deploy/final-deploy-report.json`
- `artifacts/reports/training/training_statistics.json`
- `artifacts/comprehensive-ml-workflow-report.json`
- `docs/comprehensive-ml-workflow-report.md`

### 正式交付

- `报告/多肽合成机器学习学习报告.md`
- `报告/多肽合成机器学习学习报告.docx`
- `报告/多肽合成机器学习学习报告.pdf`

## 备注

- `artifacts/` 只视为实验记录，不再作为默认推理入口硬编码依赖。
- `archive/` 中的内容默认不参与主流程，但都可以回收。
- 旧的一次性脚本已移入 `scripts/legacy/`。
