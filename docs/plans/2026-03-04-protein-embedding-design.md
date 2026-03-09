# Protein Embedding Feature Engineering Design

## Goal

在当前 `Fmoc SOP-like` 文献子集上引入预训练蛋白语言模型 embedding，用于补充现有手工理化特征，并检验其是否能在分组评估下改善 purity / yield 回归结果。

## Scope

- 新增一个独立的 Python embedding 模块，负责模型加载、序列编码、缓存与降维。
- 新增一个专门的 `Fmoc SOP + embedding` 分析脚本，对比“手工特征”与“手工特征 + embedding”。
- 更新学习报告与素材索引，补入方法与结果。

## Non-Goals

- 不改动前端推理逻辑。
- 不把 embedding 直接灌入所有已有建模脚本。
- 不在当前阶段把 embedding 并入贝叶斯分层模型。

## Design

### 1. Embedding backend

采用 ESM2 家族的 Hugging Face checkpoint 作为预训练蛋白模型来源。默认使用较小的 `facebook/esm2_t6_8M_UR50D`，原因是当前数据量很小，且本地需要可重复运行。虽然 checkpoint 较小，但其训练范式属于大型预训练蛋白语言模型，可作为这一阶段的 embedding backend。

### 2. Caching

对每条归一化后的序列做 SHA1 命名缓存，单条保存为 `.npy` 文件。缓存键包含 `model_name + sequence`，避免后续模型切换时污染。

### 3. Dimensionality control

原始 embedding 维度较高，直接拼接到几十条样本的文献数据上会放大过拟合风险。因此评估时只追加训练折上 PCA 降维后的 embedding 主成分，默认 8 维，并在每个 `source_id` 留一折内单独拟合 PCA，避免把测试折分布提前泄露给降维器。

### 4. Evaluation target

优先接入 `Fmoc SOP-like` 子集分析。原因是这条数据线最贴近当前实验 SOP，且前一轮结果已经表明在 yield 任务上存在可恢复的稳定信号，更适合作为 embedding 增益试验台。

### 5. Testing

不依赖真实 Hugging Face 下载做单元测试。测试覆盖：

- 缓存命中后不重复编码
- PCA 降维输出维度受样本数和目标维度共同限制
- 重复序列在特征表中得到一致的 embedding 特征

## Expected Outputs

- `python/protein_embeddings.py`
- `tests/test_protein_embeddings.py`
- `python/fmoc_sop_embedding_analysis.py`
- `artifacts/fmoc-sop-embedding-analysis.json`
- `docs/fmoc-sop-embedding-analysis.md`
- 学习报告与素材索引的新增小节
