# 机器学习模型开发日志

最后更新：2026-03-02

## 1. 项目目标

本项目最初的界面目标是对多肽合成质量进行预测，输出两个用户可理解的结果：

- 预测纯度
- 预测收率

开发过程中很快遇到一个核心约束：公开可获得、且带有序列信息的
`final purity / isolated yield` 数据非常少，而且分散在不同论文中，
标签定义也不完全一致。基于这个现实，项目最终分成了两条并行路线：

- 一条是用于当前前端 UI 的可复现合成数据模型
- 一条是基于真实公开实验数据训练的步骤级代理信号模型

## 2. 初始基线与代码重构

最开始的实现主要依赖手工规则和固定权重，不利于复现、调参和统一推理。
因此首先完成了特征工程和推理层的重构，使训练端与推理端共用同一套逻辑。

核心文件：

- `src/lib/peptideCore.ts`
- `python/peptide_core.py`
- `src/lib/neuralModel.ts`
- `python/neural_model.py`

这一阶段确定的主要输入特征包括：

- 序列长度归一化
- 大位阻氨基酸比例
- 难偶联氨基酸比例
- 疏水残基比例
- 带电残基比例
- breaker 残基比例
- 含硫残基比例
- 天冬酰亚胺风险
- 缩合试剂评分
- 拓扑复杂度
- 温度评分
- 溶剂评分
- 裂解条件评分
- 序列复杂度
- 最长疏水连续片段
- 半胱氨酸环化适配度

## 3. 合成数据集构建

由于一开始没有足够大的公开 `final purity / yield` 数据集，因此先在
`python/optimize_model.py` 中构建了一个化学启发式的合成训练集。

这一阶段的设计原则如下：

- 每类场景样本数平衡
- 数据生成、数据切分、模型训练均使用固定随机种子
- 目标值由可解释的化学惩罚项和奖励项组成
- 训练、验证、测试集严格分离

构建的 6 类场景为：

- `easy_linear`
- `hydrophobic_long`
- `steric_rich`
- `charged_difficult`
- `cysteine_rich`
- `cyclized`

当前合成数据集规模：

- 总样本数：`1080`
- 每类场景：`180`
- 划分：`750 train / 162 validation / 168 test`
- dataset seed：`20260301`
- split seed：`20260302`
- training seed：`20260303`

## 4. 合成模型调参与优化

用于前端的合成模型是一个小型前馈神经网络，其结构为：

- 输入特征数：`16`
- 隐藏层：`1` 层
- 激活函数：`tanh`
- 输出维度：`2`
  分别对应 purity 和 yield

在 `python/optimize_model.py` 中对以下超参数进行了搜索：

- hidden size
- learning rate
- L2 正则
- early stopping patience

当前最优配置来自 `artifacts/peptide-model-report.json`：

- hidden size：`8`
- learning rate：`0.12`
- L2：`0.0015`
- max epochs：`420`
- patience：`55`

当前最优验证集指标：

- purity RMSE：`3.4950`
- yield RMSE：`4.5176`
- combined RMSE：`4.0063`

当前最优测试集指标：

- purity RMSE：`3.5636`
- yield RMSE：`4.6527`
- combined RMSE：`4.1082`

这一轮训练输出了两个关键产物：

- `src/lib/modelArtifacts.ts`
- `artifacts/peptide-model-report.json`

这意味着当前前端展示的 purity / yield 预测已经不再依赖手写固定权重，
而是基于可重复训练得到的模型参数。

另外，这一阶段后续又补做了一轮性能优化：

- 将合成模型训练内核改为 NumPy 矩阵化
- 将候选超参数搜索改为并行执行

这样做的目的不是改变模型定义，而是缩短反复调参时的等待时间。

## 5. Python 版本实现与推理对齐

为了便于后续训练、批量实验和命令行推理，又补充了完整的 Python 实现。

相关文件：

- `python/predictor.py`
- `python/predict.py`
- `python/neural_model.py`

这一阶段的目标有三个：

- 让模型训练不依赖前端环境
- 让推理可以直接通过命令行完成
- 确保 Python 端与 TypeScript 端使用相同的模型权重和特征定义

## 6. 真实公开数据训练管线

在合成模型稳定之后，下一步转向真实公开实验数据。当前仓库中的真实数据
训练管线基于 MIT `peptimizer` 项目公开的数据集：

- source repo：`learningmatter-mit/peptimizer`
- raw url：
  `https://raw.githubusercontent.com/learningmatter-mit/peptimizer/master/dataset/data_synthesis/synthesis_data.csv`

对应实现文件：

- `python/real_data.py`
- `python/train_real_model.py`
- `python/predict_real_step.py`

这一阶段必须明确一个边界：

- 该数据集不是 final purity / isolated yield 数据集
- 它是 fast-flow peptide synthesis 的步骤级实验信号数据
- 预测目标是：
  `first_area`、`first_height`、`first_width`、`first_diff`

这些目标更接近去保护/聚集相关代理信号，而不是最终产物质量。

为了减少信息泄漏，数据切分不是随机按行切分，而是按 synthesis `serial`
做 grouped split。

当前真实数据统计如下：

- 可用样本数：`12600`
- 唯一 serial 数：`769`
- 平均 pre-chain 长度：`11.88`
- 最大 pre-chain 长度：`49`

当前数据划分：

- train：`8907`
- validation：`1791`
- test：`1902`

对应 serial 划分：

- train：`538`
- validation：`115`
- test：`116`

## 7. 真实数据神经网络架构搜索

在 `python/train_real_model.py` 中，对三类序列编码架构进行了比较：

- `gru`
- `cnn`
- `hybrid`

当前最优真实数据架构为：

- 名称：`gru_residual_small`
- architecture：`gru`
- max length：`40`
- embedding dim：`16`
- sequence hidden：`32`
- numeric hidden：`32`
- trunk hidden：`96`
- dropout：`0.12`

该模型的结构由以下部分组成：

- pre-chain token embedding
- 双向 GRU 序列编码器
- 下一步氨基酸 embedding
- coupling agent embedding
- 数值特征分支
- residual MLP trunk
- 4 维回归 head

当前结果来自 `artifacts/real-synthesis-report.json`：

- validation combined RMSE：`0.1732`
- test combined RMSE：`0.1608`

对应产物：

- `artifacts/real-synthesis-model.pt`
- `artifacts/real-synthesis-report.json`

## 8. 面向 final purity / yield 的公开数据检索

由于 MIT 这套真实数据并不直接对应当前 UI 的最终 purity / yield 目标，
因此又单独启动了文献检索和人工抽表路线，专门寻找更贴近界面需求的
公开一手数据。

来源扫描文件：

- `data/real/final_purity_yield_sources.md`

目前已整理并核实过的高价值来源包括：

- Amyloid beta C 端疏水肽合成论文
- Peptide alpha-thioester volatilizable support 论文
- Green Chemistry 2025 的 RAM SPPS 论文
- THP backbone protection 论文
- Trityl side-chain anchoring 论文
- PDAC PEGylation 论文
- Tea Bags for Fmoc SPPS 论文
- HOPO protocol 论文
- MYC PTM AFPS 论文
- recifin A 合成论文

## 9. 文献数据集人工构建

人工抽取后的本地数据集位于：

- `data/real/final_purity_yield_literature.csv`
- `data/real/final_purity_yield_literature.md`

当前数据集状态：

- source-tracked 记录数：`125`
- primary sources：`10`
- 缺失 sequence 的记录数：`15`

当前 purity stage 分布：

- `crude_hplc`：`72`
- `final_product`：`18`
- `crude_hplc_214nm`：`17`
- `purified_hplc`：`14`
- `unknown`：`4`

当前 yield stage 分布：

- `crude`：`83`
- `isolated`：`30`
- `recovery`：`12`

这一阶段的关键结论是：

- 数据量相比最初已经显著扩大
- 但标签语义仍不统一
- `crude purity`、`purified purity`、`isolated yield`、`recovery`
  不能被简单当作同一个监督任务

## 10. 文献数据 baseline 建模

为了验证当前文献数据是否已经足以支撑稳定的 final purity / yield 学习，
新增了一个保守的 baseline：

- 文件：`python/train_literature_baseline.py`

当前 baseline 方案是：

- 基于 engineered sequence features 的 ridge regression
- 对 mixed purity 任务可加入 stage features
- 用 LOOCV 评估同数据集内部可学习性
- 用 leave-one-source-out 评估跨论文泛化能力

在最近的迭代中，还修复了两个关键问题：

- 支持解析 `>95`、`>99` 这类文献数值标签
- 修复新扩表时部分 CSV 行的列错位问题

当前结果见 `artifacts/literature-baseline-report.json`。

主要结果如下：

`isolated_yield_sequence_only`

- 样本数：`19`
- LOOCV RMSE：`18.69`
- leave-one-source-out R²：`-3.71`

`crude_yield_sequence_only`

- 样本数：`79`
- LOOCV RMSE：`23.71`
- baseline RMSE：`26.16`
- leave-one-source-out R²：`-0.20`

`purity_mixed_sequence_plus_stage`

- 样本数：`110`
- LOOCV RMSE：`15.72`
- baseline RMSE：`18.94`
- leave-one-source-out R²：`-0.204`

解释如下：

- 在同来源内部，已经能看到一定弱信号
- 但跨来源泛化仍然不稳定
- 当前瓶颈依然主要是数据异质性和标签不统一，而不是神经网络结构本身

## 11. 当前仓库中的三类模型资产

截至目前，仓库里实际上存在三类不同用途的模型资产：

1. 合成数据 purity / yield 模型
   用途：服务当前前端 UI

2. 真实 fast-flow 步骤级代理信号模型
   用途：预测真实实验的 step-level proxy signals

3. 文献数据 baseline
   用途：验证 final purity / yield 是否已经能学到稳定信号

## 12. 当前总体结论

截至 2026-03-02，可以给出如下结论：

- 当前前端使用的 synthetic purity / yield 模型已经稳定，可复现，可部署
- 当前仓库中最有科学依据的真实神经网络模型是 `gru_residual_small`
- 但该真实模型预测的是步骤级代理信号，不是最终 purity / yield
- 当前文献数据已经足以观察到弱可学习信号
- 但还不足以支持“final purity / yield 已具备稳定跨论文泛化能力”这一结论

## 13. 后续建议

下一步最合理的工作包括：

- 继续扩充带序列的 isolated yield 数据来源
- 从 leave-one-source-out 进一步收紧到按序列家族分组留出
- 将 `condition_summary` 结构化为工艺特征
- 评估是否需要在前端中将 synthetic purity/yield 模型与真实步骤级风险模型拆分展示
