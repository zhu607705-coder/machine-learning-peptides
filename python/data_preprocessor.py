"""
多肽合成数据预处理管道
将各种格式的原始数据转换为模型可识别和训练的标准格式

支持的数据源：
1. CSV/Excel 文件（实验记录）
2. JSON 数据（API响应）
3. 文献数据（PubMed格式）
4. 实时合成仪数据
5. 手动输入数据
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# 导入已有的氨基酸性质数据
from enhance_dataset import (
    AMINO_ACID_PROPERTIES, 
    COUPLING_DIFFICULTY, 
    THREE_TO_ONE,
    parse_sequence,
    calculate_peptide_features
)


@dataclass
class RawDataInput:
    """原始数据输入结构"""
    sequence: str
    purity: Optional[float] = None
    yield_val: Optional[float] = None
    coupling_reagent: Optional[str] = None
    solvent: Optional[str] = None
    temperature: Optional[str] = None
    cleavage_time: Optional[str] = None
    topology: Optional[str] = None
    source: str = "unknown"
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessedFeatures:
    """处理后的特征结构"""
    # 序列特征
    sequence: str
    sequence_onehot: np.ndarray
    length: int
    
    # 物理化学特征
    avg_hydrophobicity: float
    total_charge: float
    molecular_weight: float
    avg_volume: float
    max_coupling_difficulty: float
    
    # 结构特征
    bulky_ratio: float
    polar_ratio: float
    charged_ratio: float
    aromatic_ratio: float
    sulfur_ratio: float
    
    # 合成条件特征
    reagent_score: float
    solvent_score: float
    temperature_score: float
    cleavage_score: float
    
    # 目标变量
    purity: Optional[float] = None
    yield_val: Optional[float] = None
    
    # 元数据
    feature_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    source: str = "unknown"


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_sequence(sequence: str) -> Tuple[bool, str]:
        """验证序列格式"""
        if not sequence or pd.isna(sequence):
            return False, "序列为空"
        
        # 移除修饰符
        clean_seq = sequence.replace('H-', '').replace('-OH', '').replace('-NH2', '')
        
        # 检查是否包含有效氨基酸
        parts = [p.strip() for p in clean_seq.split('-') if p.strip()]
        valid_count = 0
        
        for part in parts:
            if part in THREE_TO_ONE or (len(part) == 1 and part in AMINO_ACID_PROPERTIES):
                valid_count += 1
            elif len(part) > 1:
                # 检查是否为连续单字母
                for char in part:
                    if char in AMINO_ACID_PROPERTIES:
                        valid_count += 1
        
        if valid_count == 0:
            return False, "序列不包含有效氨基酸"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_purity(purity: float) -> Tuple[bool, str]:
        """验证纯度值"""
        if purity is None or pd.isna(purity):
            return True, "纯度为空（允许）"
        
        if not isinstance(purity, (int, float)):
            return False, "纯度必须是数值"
        
        if purity < 0 or purity > 100:
            return False, f"纯度值 {purity} 超出有效范围 [0, 100]"
        
        return True, "验证通过"
    
    @staticmethod
    def validate_yield(yield_val: float) -> Tuple[bool, str]:
        """验证收率值"""
        if yield_val is None or pd.isna(yield_val):
            return True, "收率为空（允许）"
        
        if not isinstance(yield_val, (int, float)):
            return False, "收率必须是数值"
        
        if yield_val < 0 or yield_val > 100:
            return False, f"收率值 {yield_val} 超出有效范围 [0, 100]"
        
        return True, "验证通过"


class FeatureExtractor:
    """特征提取器"""
    
    # 氨基酸分类
    BULKY_RESIDUES = {'V', 'I', 'L', 'F', 'W', 'Y'}
    POLAR_RESIDUES = {'S', 'T', 'N', 'Q', 'C', 'Y'}
    CHARGED_RESIDUES = {'D', 'E', 'R', 'H', 'K'}
    AROMATIC_RESIDUES = {'F', 'W', 'Y', 'H'}
    SULFUR_RESIDUES = {'C', 'M'}
    
    @classmethod
    def extract_sequence_features(cls, sequence: str) -> Dict:
        """提取序列相关特征"""
        residues = parse_sequence(sequence)
        
        if not residues:
            return {
                'length': 0,
                'bulky_ratio': 0,
                'polar_ratio': 0,
                'charged_ratio': 0,
                'aromatic_ratio': 0,
                'sulfur_ratio': 0,
            }
        
        length = len(residues)
        
        return {
            'length': length,
            'bulky_ratio': sum(1 for r in residues if r in cls.BULKY_RESIDUES) / length,
            'polar_ratio': sum(1 for r in residues if r in cls.POLAR_RESIDUES) / length,
            'charged_ratio': sum(1 for r in residues if r in cls.CHARGED_RESIDUES) / length,
            'aromatic_ratio': sum(1 for r in residues if r in cls.AROMATIC_RESIDUES) / length,
            'sulfur_ratio': sum(1 for r in residues if r in cls.SULFUR_RESIDUES) / length,
        }
    
    @classmethod
    def extract_condition_features(
        cls,
        coupling_reagent: Optional[str] = None,
        solvent: Optional[str] = None,
        temperature: Optional[str] = None,
        cleavage_time: Optional[str] = None
    ) -> Dict:
        """提取合成条件特征"""
        # 试剂评分
        reagent_score = 0.5
        if coupling_reagent:
            if 'HATU' in coupling_reagent:
                reagent_score = 1.0
            elif 'PyBOP' in coupling_reagent:
                reagent_score = 0.84
            elif 'HBTU' in coupling_reagent:
                reagent_score = 0.66
        
        # 溶剂评分
        solvent_score = 0.55
        if solvent:
            if 'NMP' in solvent:
                solvent_score = 0.86
            elif 'DMF/DCM' in solvent:
                solvent_score = 0.74
            elif 'DMF' in solvent:
                solvent_score = 0.55
        
        # 温度评分
        temperature_score = 0.25
        if temperature:
            if '90' in temperature:
                temperature_score = 1.0
            elif '75' in temperature:
                temperature_score = 0.72
            elif 'Room' in temperature or '25' in temperature:
                temperature_score = 0.25
        
        # 裂解时间评分
        cleavage_score = 0.44
        if cleavage_time:
            if '4' in cleavage_time:
                cleavage_score = 1.0
            elif '3' in cleavage_time:
                cleavage_score = 0.72
            elif '2' in cleavage_time:
                cleavage_score = 0.44
        
        return {
            'reagent_score': reagent_score,
            'solvent_score': solvent_score,
            'temperature_score': temperature_score,
            'cleavage_score': cleavage_score,
        }
    
    @classmethod
    def create_onehot_encoding(cls, sequence: str, max_length: int = 50) -> np.ndarray:
        """创建序列的one-hot编码"""
        residues = parse_sequence(sequence)
        amino_acids = sorted(AMINO_ACID_PROPERTIES.keys())
        
        # 初始化one-hot矩阵
        onehot = np.zeros((max_length, len(amino_acids)))
        
        # 填充one-hot编码
        for i, residue in enumerate(residues[:max_length]):
            if residue in amino_acids:
                idx = amino_acids.index(residue)
                onehot[i, idx] = 1
        
        return onehot.flatten()
    
    @classmethod
    def create_feature_vector(cls, processed: ProcessedFeatures) -> np.ndarray:
        """创建完整的特征向量"""
        features = [
            processed.length / 50.0,  # 归一化长度
            processed.avg_hydrophobicity / 5.0,  # 归一化疏水性
            processed.total_charge / 10.0,  # 归一化电荷
            processed.molecular_weight / 5000.0,  # 归一化分子量
            processed.avg_volume / 250.0,  # 归一化体积
            processed.max_coupling_difficulty / 3.0,  # 归一化难度
            processed.bulky_ratio,
            processed.polar_ratio,
            processed.charged_ratio,
            processed.aromatic_ratio,
            processed.sulfur_ratio,
            processed.reagent_score,
            processed.solvent_score,
            processed.temperature_score,
            processed.cleavage_score,
        ]
        
        return np.array(features)


class DataPreprocessor:
    """数据预处理器主类"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.extractor = FeatureExtractor()
        self.processed_data: List[ProcessedFeatures] = []
        self.errors: List[Dict] = []
    
    def process_single_record(self, input_data: RawDataInput) -> Optional[ProcessedFeatures]:
        """处理单条记录"""
        # 验证序列
        is_valid, msg = self.validator.validate_sequence(input_data.sequence)
        if not is_valid:
            self.errors.append({
                'sequence': input_data.sequence,
                'error': msg,
                'source': input_data.source
            })
            return None
        
        # 验证目标变量
        if input_data.purity is not None:
            is_valid, msg = self.validator.validate_purity(input_data.purity)
            if not is_valid:
                self.errors.append({
                    'sequence': input_data.sequence,
                    'error': msg,
                    'source': input_data.source
                })
                input_data.purity = None
        
        if input_data.yield_val is not None:
            is_valid, msg = self.validator.validate_yield(input_data.yield_val)
            if not is_valid:
                self.errors.append({
                    'sequence': input_data.sequence,
                    'error': msg,
                    'source': input_data.source
                })
                input_data.yield_val = None
        
        # 提取肽特征
        peptide_features = calculate_peptide_features(input_data.sequence)
        
        # 提取序列特征
        seq_features = self.extractor.extract_sequence_features(input_data.sequence)
        
        # 提取条件特征
        cond_features = self.extractor.extract_condition_features(
            input_data.coupling_reagent,
            input_data.solvent,
            input_data.temperature,
            input_data.cleavage_time
        )
        
        # 创建one-hot编码
        onehot = self.extractor.create_onehot_encoding(input_data.sequence)
        
        # 创建处理后的特征对象
        processed = ProcessedFeatures(
            sequence=input_data.sequence,
            sequence_onehot=onehot,
            purity=input_data.purity,
            yield_val=input_data.yield_val,
            source=input_data.source,
            **seq_features,
            **cond_features,
            avg_hydrophobicity=peptide_features.avg_hydrophobicity,
            total_charge=peptide_features.total_charge,
            molecular_weight=peptide_features.molecular_weight,
            avg_volume=peptide_features.avg_volume,
            max_coupling_difficulty=peptide_features.max_coupling_difficulty,
        )
        
        # 创建特征向量
        processed.feature_vector = self.extractor.create_feature_vector(processed)
        
        return processed
    
    def process_csv_file(
        self, 
        file_path: str,
        column_mapping: Optional[Dict[str, str]] = None,
        source_name: str = "csv"
    ) -> pd.DataFrame:
        """
        处理CSV文件
        
        Args:
            file_path: CSV文件路径
            column_mapping: 列名映射，例如 {'seq': 'sequence', 'purity_pct': 'purity'}
            source_name: 数据来源标识
        """
        print(f"正在处理CSV文件: {file_path}")
        
        # 读取CSV
        df = pd.read_csv(file_path)
        print(f"读取到 {len(df)} 条记录")
        
        # 应用列名映射
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # 处理每条记录
        processed_records = []
        for idx, row in df.iterrows():
            try:
                input_data = RawDataInput(
                    sequence=str(row.get('sequence', '')),
                    purity=row.get('purity'),
                    yield_val=row.get('yield_val') or row.get('yield'),
                    coupling_reagent=row.get('coupling_reagent'),
                    solvent=row.get('solvent'),
                    temperature=row.get('temperature'),
                    cleavage_time=row.get('cleavage_time'),
                    topology=row.get('topology'),
                    source=source_name,
                    metadata={'row_index': idx}
                )
                
                processed = self.process_single_record(input_data)
                if processed:
                    processed_records.append(processed)
                    
            except Exception as e:
                self.errors.append({
                    'row_index': idx,
                    'error': str(e),
                    'source': source_name
                })
        
        print(f"成功处理 {len(processed_records)} 条记录")
        print(f"失败记录数: {len(self.errors)}")
        
        return self._to_dataframe(processed_records)
    
    def process_json_data(
        self, 
        json_data: Union[str, Dict, List],
        source_name: str = "json"
    ) -> pd.DataFrame:
        """处理JSON数据"""
        print(f"正在处理JSON数据...")
        
        # 解析JSON
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # 确保是列表
        if isinstance(data, dict):
            data = [data]
        
        print(f"读取到 {len(data)} 条记录")
        
        # 处理每条记录
        processed_records = []
        for idx, item in enumerate(data):
            try:
                input_data = RawDataInput(
                    sequence=item.get('sequence', ''),
                    purity=item.get('purity'),
                    yield_val=item.get('yield') or item.get('yield_val'),
                    coupling_reagent=item.get('coupling_reagent'),
                    solvent=item.get('solvent'),
                    temperature=item.get('temperature'),
                    cleavage_time=item.get('cleavage_time'),
                    topology=item.get('topology'),
                    source=source_name,
                    metadata={'index': idx}
                )
                
                processed = self.process_single_record(input_data)
                if processed:
                    processed_records.append(processed)
                    
            except Exception as e:
                self.errors.append({
                    'index': idx,
                    'error': str(e),
                    'source': source_name
                })
        
        print(f"成功处理 {len(processed_records)} 条记录")
        
        return self._to_dataframe(processed_records)
    
    def process_manual_input(
        self,
        sequence: str,
        purity: Optional[float] = None,
        yield_val: Optional[float] = None,
        **kwargs
    ) -> Optional[ProcessedFeatures]:
        """处理手动输入数据"""
        input_data = RawDataInput(
            sequence=sequence,
            purity=purity,
            yield_val=yield_val,
            source="manual_input",
            **kwargs
        )
        
        return self.process_single_record(input_data)
    
    def _to_dataframe(self, processed_records: List[ProcessedFeatures]) -> pd.DataFrame:
        """将处理后的记录转换为DataFrame"""
        if not processed_records:
            return pd.DataFrame()
        
        data = []
        for record in processed_records:
            row = {
                'sequence': record.sequence,
                'length': record.length,
                'avg_hydrophobicity': record.avg_hydrophobicity,
                'total_charge': record.total_charge,
                'molecular_weight': record.molecular_weight,
                'avg_volume': record.avg_volume,
                'max_coupling_difficulty': record.max_coupling_difficulty,
                'bulky_ratio': record.bulky_ratio,
                'polar_ratio': record.polar_ratio,
                'charged_ratio': record.charged_ratio,
                'aromatic_ratio': record.aromatic_ratio,
                'sulfur_ratio': record.sulfur_ratio,
                'reagent_score': record.reagent_score,
                'solvent_score': record.solvent_score,
                'temperature_score': record.temperature_score,
                'cleavage_score': record.cleavage_score,
                'purity': record.purity,
                'yield_val': record.yield_val,
                'source': record.source,
            }
            
            # 添加特征向量（展开为单独列）
            for i, val in enumerate(record.feature_vector):
                row[f'feature_{i}'] = val
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_error_report(self) -> pd.DataFrame:
        """获取错误报告"""
        return pd.DataFrame(self.errors)
    
    def save_processed_data(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        format: str = 'csv'
    ):
        """保存处理后的数据"""
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        
        print(f"数据已保存到: {output_path}")
        print(f"总记录数: {len(df)}")
        print(f"特征数: {len(df.columns)}")


def create_training_pipeline_example():
    """创建训练数据管道的示例"""
    
    print("="*60)
    print("多肽合成数据预处理管道示例")
    print("="*60)
    
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 示例1: 处理CSV文件
    print("\n示例1: 处理文献数据CSV")
    print("-"*40)
    
    column_mapping = {
        'purity_pct': 'purity',
        'yield_pct': 'yield_val',
    }
    
    try:
        df_literature = preprocessor.process_csv_file(
            '../data/real/final_purity_yield_literature.csv',
            column_mapping=column_mapping,
            source_name='literature'
        )
        
        if len(df_literature) > 0:
            preprocessor.save_processed_data(
                df_literature,
                '../data/processed_literature_data.csv'
            )
    except Exception as e:
        print(f"处理文献数据失败: {e}")
    
    # 示例2: 处理手动输入
    print("\n示例2: 处理手动输入")
    print("-"*40)
    
    manual_record = preprocessor.process_manual_input(
        sequence='H-Gly-Ala-Val-Leu-Ile-OH',
        purity=85.5,
        yield_val=72.3,
        coupling_reagent='HATU',
        solvent='DMF',
        temperature='Room Temperature',
        cleavage_time='2 hours',
        topology='Linear'
    )
    
    if manual_record:
        print(f"序列: {manual_record.sequence}")
        print(f"长度: {manual_record.length}")
        print(f"平均疏水性: {manual_record.avg_hydrophobicity:.2f}")
        print(f"分子量: {manual_record.molecular_weight:.2f}")
        print(f"特征向量维度: {len(manual_record.feature_vector)}")
    
    # 示例3: 处理JSON数据
    print("\n示例3: 处理JSON数据")
    print("-"*40)
    
    json_data = [
        {
            "sequence": "RWS",
            "purity": 92.0,
            "yield": 78.5,
            "coupling_reagent": "HATU",
            "solvent": "NMP",
            "temperature": "Microwave 75°C"
        },
        {
            "sequence": "Gly-Ala-Ser",
            "purity": 88.0,
            "coupling_reagent": "HBTU"
        }
    ]
    
    df_json = preprocessor.process_json_data(json_data, source_name='api')
    print(f"处理完成: {len(df_json)} 条记录")
    
    # 输出错误报告
    if preprocessor.errors:
        print("\n错误报告:")
        print("-"*40)
        error_df = preprocessor.get_error_report()
        print(error_df.head())
    
    print("\n" + "="*60)
    print("预处理管道示例完成!")
    print("="*60)


if __name__ == '__main__':
    create_training_pipeline_example()
