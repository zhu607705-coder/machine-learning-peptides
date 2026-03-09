# Conditional GRU Attention 5-Seed Confirmation

- Seeds: `[20260305, 20260315, 20260325, 20260335, 20260345]`
- Folds: `3`

| Combo | Runs | Val RMSE mean | Val RMSE std | Test RMSE mean | Test RMSE std | Win rate vs GRU | Stable better than GRU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gru_h32_do0.12_delta0.75 | 15 | 0.167930 | 0.014037 | 0.173169 | 0.004832 | 0.000 | no |
| conditional_h40_do0.12_delta0.75 | 15 | 0.164660 | 0.015310 | 0.170604 | 0.004618 | 0.733 | yes |
| conditional_h40_do0.18_delta0.75 | 15 | 0.164395 | 0.015004 | 0.170592 | 0.004255 | 0.667 | yes |

## Best by test RMSE mean: `conditional_h40_do0.18_delta0.75`