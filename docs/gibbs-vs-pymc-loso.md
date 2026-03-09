# Gibbs vs PyMC (LOSO) 对照

| 指标 | Gibbs | PyMC/NUTS | Δ(PyMC-Gibbs) |
| --- | ---: | ---: | ---: |
| Purity RMSE (↓) | 18.6537 | 31.3486 | +12.6949 |
| Purity R² (↑) | 0.0269 | -1.7482 | -1.7751 |
| Yield RMSE (↓) | 34.6167 | 35.6848 | +1.0681 |
| Yield R² (↑) | -0.2793 | -0.3594 | -0.0802 |
| Purity ±5% Acc (↑) | 0.1416 | 0.1681 | +0.0265 |
| Yield ±10% Acc (↑) | 0.1858 | 0.1858 | +0.0000 |
| Joint Acc (↑) | 0.0442 | 0.0354 | -0.0088 |
| Purity 95% Coverage (≈) | 0.8850 | 0.8053 | -0.0796 |
| Yield 95% Coverage (≈) | 0.7434 | 0.8142 | +0.0708 |
| Purity Interval Width (↓) | 65.5504 | 68.7515 | +3.2012 |
| Yield Interval Width (↓) | 91.5954 | 103.0038 | +11.4084 |

