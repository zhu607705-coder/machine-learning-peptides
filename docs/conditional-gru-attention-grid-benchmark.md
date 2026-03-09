# Conditional GRU Attention Grid Benchmark

- Seeds: `[20260305, 20260315, 20260325]`
- Folds: `3`

| Combo | Runs | Val RMSE mean | Val RMSE std | Test RMSE mean | Test RMSE std | Win rate vs GRU | Stable better than GRU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conditional_h32_do0.12_delta0.75 | 9 | 0.158057 | 0.015122 | 0.171320 | 0.006542 | 0.667 | no |
| conditional_h40_do0.12_delta0.75 | 9 | 0.158021 | 0.015186 | 0.170987 | 0.003821 | 0.667 | yes |
| conditional_h40_do0.15_delta0.75 | 9 | 0.157965 | 0.014914 | 0.172347 | 0.005540 | 0.444 | no |
| conditional_h40_do0.15_delta1.00 | 9 | 0.158043 | 0.015165 | 0.172090 | 0.005541 | 0.556 | no |
| conditional_h40_do0.18_delta0.75 | 9 | 0.157898 | 0.014911 | 0.170666 | 0.003896 | 0.667 | yes |
| conditional_h48_do0.15_delta0.75 | 9 | 0.157900 | 0.014135 | 0.173993 | 0.004189 | 0.444 | no |
| gru_h32_do0.12_delta0.75 | 9 | 0.161560 | 0.013599 | 0.173094 | 0.005298 | 0.000 | no |

## Best by test RMSE mean: `conditional_h40_do0.18_delta0.75`
## Stable better combos: `['conditional_h40_do0.12_delta0.75', 'conditional_h40_do0.18_delta0.75']`