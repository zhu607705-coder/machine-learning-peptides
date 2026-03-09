# PyMC 环境配置说明

## 1. 环境位置

- 项目虚拟环境：`/Users/zhuhangcheng/Downloads/星辰计划/机器学习多肽/peptide-synthesis-predictor/.venv-pymc`
- Python 版本：`3.12.12`

## 2. 已安装核心依赖

- `pymc==5.28.1`
- `arviz==0.23.4`
- `pytensor==2.38.1`
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- 工作流额外依赖：`torch==2.10.0`, `transformers==5.3.0`

## 3. 使用方式

```bash
cd /Users/zhuhangcheng/Downloads/星辰计划/机器学习多肽/peptide-synthesis-predictor
source .venv-pymc/bin/activate
python --version
```

## 4. 快速校验

```bash
cd /Users/zhuhangcheng/Downloads/星辰计划/机器学习多肽/peptide-synthesis-predictor
.venv-pymc/bin/python -c "import pymc,arviz,pytensor; print(pymc.__version__, arviz.__version__, pytensor.__version__)"
.venv-pymc/bin/python python/comprehensive_ml_workflow.py --help
```

## 5. 运行建议

- 后续训练请优先使用 `.venv-pymc/bin/python`，避免落回系统 Python 3.14。
- 若要复现 LOSO 实验，建议先从低采样参数开始（如 `--bayes-draws 80 --bayes-tune 80 --bayes-chains 1`），确认流程通后再放大采样。
