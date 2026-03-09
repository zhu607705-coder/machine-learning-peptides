# 多肽合成数据预处理管道使用指南

## 概述

`data_preprocessor.py` 是一个完整的数据预处理管道，用于将各种格式的原始多肽合成数据转换为模型可识别和训练的标准格式。

## 功能特性

### 1. 支持的数据源
- **CSV/Excel 文件** - 实验记录、文献数据
- **JSON 数据** - API响应、配置文件
- **手动输入** - 单条数据快速处理
- **实时合成仪数据** - 仪器导出数据

### 2. 数据验证
- 序列格式验证（支持单字母和三字母代码）
- 纯度值范围检查 [0-100]
- 收率值范围检查 [0-100]
- 缺失值处理

### 3. 特征工程

#### 序列特征
- 序列长度
- One-hot编码（50个氨基酸位置 × 20种氨基酸）

#### 物理化学特征
- 平均疏水性（Kyte-Doolittle标度）
- 总电荷数
- 分子量
- 平均氨基酸体积
- 最大偶联难度评分

#### 结构特征
- 疏水氨基酸比例（V, I, L, F, W, Y）
- 极性氨基酸比例（S, T, N, Q, C, Y）
- 带电氨基酸比例（D, E, R, H, K）
- 芳香族氨基酸比例（F, W, Y, H）
- 含硫氨基酸比例（C, M）

#### 合成条件特征
- 试剂评分（HATU=1.0, PyBOP=0.84, HBTU=0.66）
- 溶剂评分（NMP=0.86, DMF/DCM=0.74, DMF=0.55）
- 温度评分（90°C=1.0, 75°C=0.72, RT=0.25）
- 裂解时间评分（4h=1.0, 3h=0.72, 2h=0.44）

#### 归一化特征向量（15维）
```python
[长度/50, 疏水性/5, 电荷/10, 分子量/5000, 体积/250, 难度/3,
 疏水比例, 极性比例, 带电比例, 芳香比例, 含硫比例,
 试剂评分, 溶剂评分, 温度评分, 裂解评分]
```

## 使用方法

### 1. 基本使用

```python
from data_preprocessor import DataPreprocessor

# 初始化预处理器
preprocessor = DataPreprocessor()

# 处理CSV文件
df = preprocessor.process_csv_file(
    file_path='data.csv',
    column_mapping={'seq': 'sequence', 'purity_pct': 'purity'},
    source_name='my_experiment'
)

# 保存处理后的数据
preprocessor.save_processed_data(df, 'output.csv')
```

### 2. 处理CSV文件

```python
# 定义列名映射（如果CSV列名与标准不同）
column_mapping = {
    'seq': 'sequence',           # 序列列
    'purity_pct': 'purity',      # 纯度列
    'yield_pct': 'yield_val',    # 收率列
    'reagent': 'coupling_reagent', # 试剂列
    'sol': 'solvent',            # 溶剂列
    'temp': 'temperature',       # 温度列
    'cleavage': 'cleavage_time', # 裂解时间列
}

# 处理文件
df = preprocessor.process_csv_file(
    file_path='../data/my_data.csv',
    column_mapping=column_mapping,
    source_name='laboratory'
)
```

### 3. 处理JSON数据

```python
import json

# 从API获取的JSON数据
json_data = [
    {
        "sequence": "H-Gly-Ala-Val-OH",
        "purity": 92.5,
        "yield": 78.0,
        "coupling_reagent": "HATU",
        "solvent": "DMF",
        "temperature": "Room Temperature"
    },
    {
        "sequence": "RWS",
        "purity": 88.0,
        "coupling_reagent": "HBTU"
    }
]

# 处理JSON数据
df = preprocessor.process_json_data(json_data, source_name='api')
```

### 4. 手动输入单条数据

```python
# 处理单条记录
result = preprocessor.process_manual_input(
    sequence='H-Gly-Ala-Val-Leu-Ile-OH',
    purity=85.5,
    yield_val=72.3,
    coupling_reagent='HATU',
    solvent='DMF',
    temperature='Room Temperature',
    cleavage_time='2 hours',
    topology='Linear'
)

# 查看特征
print(f"序列: {result.sequence}")
print(f"特征向量: {result.feature_vector}")
print(f"One-hot编码维度: {len(result.sequence_onehot)}")
```

### 5. 批量处理多个文件

```python
import glob

# 批量处理多个CSV文件
all_data = []
for file_path in glob.glob('../data/experiments/*.csv'):
    df = preprocessor.process_csv_file(file_path, source_name='batch')
    all_data.append(df)

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)
preprocessor.save_processed_data(combined_df, 'combined_data.csv')
```

## 输出格式

处理后的数据包含以下列：

| 列名 | 说明 | 类型 |
|------|------|------|
| sequence | 肽序列 | string |
| length | 序列长度 | int |
| avg_hydrophobicity | 平均疏水性 | float |
| total_charge | 总电荷 | float |
| molecular_weight | 分子量 | float |
| avg_volume | 平均体积 | float |
| max_coupling_difficulty | 最大偶联难度 | float |
| bulky_ratio | 疏水氨基酸比例 | float |
| polar_ratio | 极性氨基酸比例 | float |
| charged_ratio | 带电氨基酸比例 | float |
| aromatic_ratio | 芳香族氨基酸比例 | float |
| sulfur_ratio | 含硫氨基酸比例 | float |
| reagent_score | 试剂评分 | float |
| solvent_score | 溶剂评分 | float |
| temperature_score | 温度评分 | float |
| cleavage_score | 裂解评分 | float |
| purity | 纯度（目标变量） | float |
| yield_val | 收率（目标变量） | float |
| source | 数据来源 | string |
| feature_0 - feature_14 | 归一化特征向量 | float |

## 错误处理

```python
# 获取错误报告
error_df = preprocessor.get_error_report()
print(error_df)

# 错误报告包含：
# - sequence: 出错的序列
# - error: 错误信息
# - source: 数据来源
```

## 完整示例

```python
from data_preprocessor import DataPreprocessor
import pandas as pd

# 初始化
preprocessor = DataPreprocessor()

# 1. 处理文献数据
print("处理文献数据...")
df_literature = preprocessor.process_csv_file(
    '../data/real/final_purity_yield_literature.csv',
    column_mapping={'purity_pct': 'purity', 'yield_pct': 'yield_val'},
    source_name='literature'
)

# 2. 处理实验数据
print("处理实验数据...")
df_experiment = preprocessor.process_csv_file(
    '../data/experiments/2024_01.csv',
    source_name='experiment'
)

# 3. 处理API数据
api_data = [
    {"sequence": "GAVL", "purity": 95.0, "coupling_reagent": "HATU"},
    {"sequence": "RWSK", "purity": 89.5, "coupling_reagent": "HBTU"}
]
df_api = preprocessor.process_json_data(api_data, source_name='api')

# 4. 合并所有数据
combined_df = pd.concat([df_literature, df_experiment, df_api], ignore_index=True)

# 5. 保存
preprocessor.save_processed_data(combined_df, '../data/training_data.csv')

# 6. 查看统计信息
print(f"\n数据统计:")
print(f"总记录数: {len(combined_df)}")
print(f"特征数: {len(combined_df.columns)}")
print(f"\n纯度统计:")
print(combined_df['purity'].describe())
```

## 注意事项

1. **序列格式**: 支持以下格式
   - `H-Gly-Ala-Val-OH`（三字母，带修饰符）
   - `Gly-Ala-Val`（三字母，无修饰符）
   - `GAV`（单字母连续）
   - `G-A-V`（单字母分隔）

2. **缺失值处理**:
   - 序列: 必须提供，否则记录会被丢弃
   - 纯度/收率: 可以为空（用于预测场景）
   - 合成条件: 使用默认值

3. **数据验证**:
   - 纯度必须在 [0, 100] 范围内
   - 收率必须在 [0, 100] 范围内
   - 序列必须包含至少一个有效氨基酸

## 扩展功能

### 自定义特征提取

```python
class CustomFeatureExtractor(FeatureExtractor):
    @classmethod
    def extract_custom_features(cls, sequence: str) -> Dict:
        # 添加自定义特征
        return {
            'my_feature': calculate_my_feature(sequence),
        }
```

### 自定义验证规则

```python
class CustomValidator(DataValidator):
    @staticmethod
    def validate_custom(data: Dict) -> Tuple[bool, str]:
        # 添加自定义验证
        if data.get('custom_field') < 0:
            return False, "自定义字段必须大于0"
        return True, "验证通过"
```

## 参考文献

- Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. Journal of molecular biology, 157(1), 105-132.
- Merrifield, R. B. (1963). Solid phase peptide synthesis. I. The synthesis of a tetrapeptide. Journal of the American Chemical Society, 85(14), 2149-2154.
