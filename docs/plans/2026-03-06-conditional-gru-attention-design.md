# Conditional GRU Attention Design

## Goal

在不扩大整体模型规模太多的前提下，尝试一条“交互前移”的候选结构：让 `pre-chain` 编码结果在序列层面与 `next amino acid`、偶联类别和数值工艺特征发生条件注意力交互，并把输出拆成目标专属 head，同时用 `delta = first - prev` 作为训练目标。

## Why This Design

当前 `gru` 与 `rnn_attention` 都属于晚融合结构：序列先被单独编码，后续再与 `next amino acid` 和工艺特征拼接。这样的归纳偏置并不直接表达“下一个氨基酸在当前链段上下文中的局部困难度”。

本次设计只改三件事：

1. 用 `BiGRU` 输出整条 `pre-chain` 的 token-level states。
2. 用 `next amino acid + coupling + numeric branch` 生成条件 query，对序列 states 做 additive attention。
3. 对四个目标使用独立 head，并改为预测 `delta`，最后用 `prev_*` 还原到绝对量。

## Scope

- 修改 `python/train_real_model.py`
- 修改相关测试
- 跑最小对照实验
- 将验证后的结论同步到两份学习报告

## Success Criteria

- 新候选结构可以纳入现有训练与基准脚本
- 训练和评估流程不破坏旧架构
- 能得到一组和当前 `gru` 可直接对照的指标
