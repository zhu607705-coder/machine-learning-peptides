# RNN+Attention 架构对照结论（与现有模型）

数据来源：`artifacts/real-synthesis-report.json`（2026-03-05 训练轮次），同一数据切分、同一训练流程、同一 Huber `delta=1.0`。

| 架构 | validation combined RMSE | test combined RMSE |
| --- | ---: | ---: |
| `gru_residual_small` | `0.171968` | `0.162649` |
| `rnn_attention_residual_medium` | `0.172909` | `0.162632` |
| `hybrid_residual_medium` | `0.181533` | `0.150742` |
| `cnn_residual_medium` | `0.184438` | `0.140532` |

## 与当前主基线（GRU）的直接对照

- `rnn_attention` 相比 `gru`：
  - validation RMSE：`+0.000941`（变差）
  - test RMSE：`-0.000017`（几乎持平，统计意义很弱）

## 结论（是否替换主架构）

当前不建议直接把主架构从 `gru` 切到 `rnn_attention`。原因：

1. `rnn_attention` 在验证集上没有超过 GRU，且优势不稳定。
2. 当前最佳历史结果仍是 `gru + Huber(delta=0.75)`（`val=0.171534`, `test=0.162049`），新架构未超过该参考点。
3. 单次切分下 `cnn/hybrid` 出现“验证差、测试好”的分化，提示方差较高，不能据此直接换主架构。

## 建议的下一步判定标准

- 在 grouped CV（或多随机种子重复）下比较 `GRU` 与 `RNN+Attention` 的均值与方差。
- 若 `RNN+Attention` 在主指标均值提升且方差不恶化，再考虑替换主架构。

## 更新：3×3 稳定性复测 + 小网格（hidden/dropout/delta）

已追加一轮 `3 seeds × 3 folds` 的小网格复测（共 8 组 `rnn_attention` 组合，72 次训练）：

- `hidden={32,40}`
- `dropout={0.12,0.18}`
- `delta={0.75,1.00}`

结果文件：

- `artifacts/rnn-attention-grid-benchmark.json`
- `docs/rnn-attention-grid-benchmark.md`

关键结论：

1. `win-rate >= 0.6` 目标已达到。  
   - 最佳胜率组合：`h40/do0.12/d0.75`，对 GRU 配对胜率 `0.778`，test RMSE 均值 `0.170971`（优于 GRU 的 `0.173258`）。
2. 但最高胜率组合的方差略高于 GRU（`0.004530 > 0.003721`），严格“均值更优且方差不劣化”标准下不算稳定替代。
3. 同时满足“均值更优 + 方差不劣化 + 胜率≥0.6”的组合为：  
   - `h40/do0.18/d1.00`：胜率 `0.667`，test RMSE 均值 `0.172445`，std `0.003651`。  
   这是当前最接近“可替换候选”的配置。

结论更新：`rnn_attention` 已从“无稳定优势”推进到“存在满足阈值的可替换候选”，但整体提升幅度仍小，建议下一步先扩大 seed 数（如 5~7 个）再做最终架构替换决策。

## 最终复核：仅 `h40/do0.18/d1.00` vs GRU（5 seeds）

已按更严格要求做 5 seeds 复核（`20260305, 20260315, 20260325, 20260335, 20260345`，每个 3 folds，共 15 组配对）：

- GRU 基线（同 seeds/folds）：  
  test RMSE mean/std = `0.173455 / 0.003233`
- `rnn_attn_h40_do0.18_delta1.00`：  
  test RMSE mean/std = `0.172406 / 0.005069`  
  配对胜率 vs GRU = `0.600`

对应文件：

- `artifacts/grouped-cv-architecture-benchmark-5seeds-gru.json`
- `artifacts/rnn-attention-grid-benchmark-5seeds-h40-do018-d1.json`
- `docs/grouped-cv-architecture-benchmark-5seeds-gru.md`
- `docs/rnn-attention-grid-benchmark-5seeds-h40-do018-d1.md`

最终判定：当前**不建议正式替换主架构**。  
理由是虽然胜率达到了目标阈值（0.6），且均值略优，但标准差明显高于 GRU，稳定性仍不足。
