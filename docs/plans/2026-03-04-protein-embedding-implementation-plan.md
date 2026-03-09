# Protein Embedding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为当前多肽文献分析增加一个可缓存、可降维、可比较增益的预训练蛋白 embedding 特征工程模块。

**Architecture:** 新增独立 embedding 模块，将原始 ESM2 mean-pooled embedding 缓存到本地，再在分组交叉验证折内做 PCA 降维后与现有手工特征拼接。评估脚本只先落到 `Fmoc SOP-like` 子集，避免一次性把高维特征扩散到整个项目。

**Tech Stack:** Python, PyTorch, transformers, NumPy, scikit-learn, pytest

---

### Task 1: Write the failing embedding utility tests

**Files:**
- Create: `tests/test_protein_embeddings.py`
- Create: `python/protein_embeddings.py`

**Step 1: Write the failing test**

编写以下测试：
- 缓存命中后不再调用底层编码函数
- PCA 降维输出维度受 `n_components` 与 `n_samples - 1` 共同限制
- 重复序列生成的 embedding 特征一致

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_protein_embeddings.py -q`

Expected: FAIL，因为目标模块与函数尚不存在。

**Step 3: Write minimal implementation**

实现最小可用的 embedding 提取、缓存与降维逻辑，并支持通过 fake extractor 做测试。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_protein_embeddings.py -q`

Expected: PASS

### Task 2: Add Fmoc SOP embedding analysis script

**Files:**
- Create: `python/fmoc_sop_embedding_analysis.py`
- Modify: `python/fmoc_sop_subset_analysis.py` only if reuse helper import is needed

**Step 1: Write the failing test**

这里不额外写端到端测试，改为脚本级运行验证。

**Step 2: Run analysis to verify current gap**

Run: `python3 python/fmoc_sop_embedding_analysis.py`

Expected: 首次运行若缺依赖或模型下载失败，则据实暴露阻塞。

**Step 3: Write minimal implementation**

基于已有 `Fmoc SOP-like` 子集，输出：
- 手工特征 baseline
- 手工特征 + embedding 的 grouped Ridge 结果
- PCA 维度、模型名、缓存信息

**Step 4: Run analysis to verify it passes**

Run: `python3 python/fmoc_sop_embedding_analysis.py`

Expected: 成功生成 JSON 与 Markdown 报告。

### Task 3: Update documentation

**Files:**
- Modify: `docs/多肽合成机器学习学习报告.md`
- Modify: `docs/多肽合成机器学习学习报告-素材索引.md`

**Step 1: Add a new subsection**

补充 embedding 方法、参数、结果和解释。

**Step 2: Verify references and exports**

Run:
- `python3 python/audit_report_citations.py`
- `python3 python/export_report_documents.py`

Expected:
- 引用编号仍一致
- Word/PDF 成功导出

