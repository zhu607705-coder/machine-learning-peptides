# RNN Attention Grid Benchmark (3x3 Stability)

- Seeds: `[20260305, 20260315, 20260325, 20260335, 20260345]`
- Folds: `3`
- Validation ratio: `0.2`
- Hidden grid: `[40]`
- Dropout grid: `[0.18]`
- Delta grid: `[1.0]`
- Target win-rate threshold: `0.6`

- GRU baseline mean/std: `0.173455` / `0.003233`

| Combo | Runs | Val RMSE mean | Val RMSE std | Test RMSE mean | Test RMSE std | Pairwise win rate vs GRU | Stable better than GRU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rnn_attn_h40_do0.18_delta1.00 | 15 | 0.168317 | 0.013526 | 0.172406 | 0.005069 | 0.600 | no |

## Best by win rate: `rnn_attn_h40_do0.18_delta1.00`
## Best by test RMSE mean: `rnn_attn_h40_do0.18_delta1.00`