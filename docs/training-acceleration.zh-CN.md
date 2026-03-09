# Windows GPU 与 macOS CPU 训练加速说明

最后更新：2026-03-02

## 1. 当前加速策略

`python/train_real_model.py` 现已支持平台自适应训练加速：

- Windows：优先自动选择 `CUDA`
- macOS：默认优先选择优化后的 `CPU`
- 其他平台：优先 `CUDA`，否则回退到 `MPS` 或 `CPU`

脚本新增的优化点包括：

- 自动设备选择
- Windows CUDA 混合精度训练
- CUDA 下 `pin_memory` 与 `non_blocking` 传输
- DataLoader 多进程加载
- 将数据集预先物化为 tensor，减少 `__getitem__` 开销
- macOS CPU 线程数与 interop 线程数优化
- 可选 `torch.compile`

## 2. 默认行为

直接运行：

```bash
python3 python/train_real_model.py
```

脚本会输出当前运行时配置，例如：

- device
- 设备选择原因
- CPU 线程数
- DataLoader worker 数
- 是否启用 AMP
- 是否启用 `torch.compile`

训练完成后，运行时配置也会写入：

- `artifacts/real-synthesis-report.json`

## 3. Windows 中的 GPU 加速

### 环境要求

- 安装带 CUDA 支持的 PyTorch
- 机器上有可用 NVIDIA GPU
- `torch.cuda.is_available()` 返回 `True`

### 推荐运行方式

在 Windows 命令行中：

```bat
set PEPTIDE_DEVICE=cuda
set PEPTIDE_NUM_WORKERS=4
set PEPTIDE_COMPILE=1
python python/train_real_model.py
```

### 当前实现中的加速项

- 自动使用 `CUDA`
- 自动启用 `torch.amp.autocast(..., dtype=float16)`
- 自动启用 `GradScaler`
- 自动启用 `pin_memory=True`
- 自动启用 `non_blocking=True`
- 自动启用 `cudnn.benchmark=True`

### 可调环境变量

- `PEPTIDE_DEVICE`
  可选：`auto` / `cpu` / `cuda` / `mps`
- `PEPTIDE_NUM_WORKERS`
  控制 DataLoader worker 数
- `PEPTIDE_COMPILE`
  `1` 表示启用 `torch.compile`
- `PEPTIDE_COMPILE_MODE`
  默认 `reduce-overhead`

## 4. macOS 中的 CPU 加速

当前策略遵循“mac 中 CPU 加速”要求，因此在 macOS 上默认不会自动切到 MPS，
而是优先使用优化后的 CPU 路径。

### 推荐运行方式

```bash
export PEPTIDE_DEVICE=cpu
export PEPTIDE_CPU_THREADS=8
export PEPTIDE_INTEROP_THREADS=2
export PEPTIDE_NUM_WORKERS=4
python3 python/train_real_model.py
```

如果你的 Mac 是高核心数机器，可以进一步尝试：

```bash
export PEPTIDE_CPU_THREADS=10
export PEPTIDE_INTEROP_THREADS=3
```

### 当前实现中的加速项

- 自动设置 `torch.set_num_threads(...)`
- 自动设置 `torch.set_num_interop_threads(...)`
- 自动设置 `torch.set_float32_matmul_precision(\"high\")`
- DataLoader 多 worker
- 数据预先转为 tensor，减少 Python 端样本构造开销

### 建议调参方法

在 macOS CPU 训练中，最重要的是三个变量：

- `PEPTIDE_CPU_THREADS`
- `PEPTIDE_INTEROP_THREADS`
- `PEPTIDE_NUM_WORKERS`

推荐从下面组合开始尝试：

1. `8 / 2 / 4`
2. `10 / 2 / 4`
3. `8 / 3 / 6`

观察总训练时间和 CPU 占用后再定型。

## 5. 手动覆盖设备

如果你明确要切换设备，可以这样指定：

```bash
export PEPTIDE_DEVICE=cpu
```

```bash
export PEPTIDE_DEVICE=cuda
```

```bash
export PEPTIDE_DEVICE=mps
```

注意：

- `mps` 不是 macOS 默认路径
- 只有显式设置 `PEPTIDE_DEVICE=mps` 时才会启用

## 6. 建议的实际使用方式

### Windows + NVIDIA

优先使用：

```bat
set PEPTIDE_DEVICE=cuda
set PEPTIDE_NUM_WORKERS=4
set PEPTIDE_COMPILE=1
python python/train_real_model.py
```

### MacBook / Mac Studio

优先使用：

```bash
export PEPTIDE_DEVICE=cpu
export PEPTIDE_CPU_THREADS=8
export PEPTIDE_INTEROP_THREADS=2
export PEPTIDE_NUM_WORKERS=4
python3 python/train_real_model.py
```

## 7. 当前边界

这套加速目前主要覆盖的是：

- `python/train_real_model.py`

也就是 PyTorch 真实数据训练管线。

当前的 `python/optimize_model.py` 仍然是纯 Python + NumPy 风格的小模型搜索，
它的主要瓶颈不在 GPU，而在 Python 循环本身。如果后续需要，还可以继续做：

- NumPy 向量化
- PyTorch 化
- 多进程搜索

## 8. 验证方法

运行训练后，可以在输出里检查：

- device 是否符合预期
- useAmp 是否为 `true`（Windows CUDA）
- numWorkers 是否生效

训练报告里也会记录运行时信息：

- `artifacts/real-synthesis-report.json`
