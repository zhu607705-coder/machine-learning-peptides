# Grouped CV + Multi-Seed Architecture Benchmark

- Seeds: `[20260305, 20260315, 20260325, 20260335, 20260345]`
- Folds: `3`
- Validation ratio inside training fold: `0.2`
- Huber delta: `1.0`
- Epoch scale: `1.0`

| Architecture | Runs | Val RMSE mean | Val RMSE std | Test RMSE mean | Test RMSE std | vs GRU mean Δ | vs GRU std Δ | Win rate vs GRU | Stable better than GRU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gru | 15 | 0.168148 | 0.013651 | 0.173455 | 0.003233 | +0.000000 | +0.000000 | 0.000 | no |
