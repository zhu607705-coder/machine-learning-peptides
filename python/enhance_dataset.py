"""
数据集增强和整合脚本
用于整合现有数据集并生成增强数据以优化模型训练
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass

# 氨基酸物理化学性质数据
# 来源: Kyte & Doolittle (1982), Zimmerman et al. (1968)
AMINO_ACID_PROPERTIES = {
    'G': {'hydrophobicity': -0.4, 'pI': 5.97, 'mw': 57.05, 'volume': 60.1, 'charge': 0},
    'A': {'hydrophobicity': 1.8, 'pI': 6.01, 'mw': 71.08, 'volume': 88.6, 'charge': 0},
    'V': {'hydrophobicity': 4.2, 'pI': 5.97, 'mw': 99.13, 'volume': 140.0, 'charge': 0},
    'I': {'hydrophobicity': 4.5, 'pI': 6.05, 'mw': 113.16, 'volume': 166.7, 'charge': 0},
    'L': {'hydrophobicity': 3.8, 'pI': 5.98, 'mw': 113.16, 'volume': 166.7, 'charge': 0},
    'F': {'hydrophobicity': 2.8, 'pI': 5.48, 'mw': 147.18, 'volume': 189.9, 'charge': 0},
    'W': {'hydrophobicity': -0.9, 'pI': 5.89, 'mw': 186.21, 'volume': 227.8, 'charge': 0},
    'Y': {'hydrophobicity': -1.3, 'pI': 5.66, 'mw': 163.18, 'volume': 193.6, 'charge': 0},
    'D': {'hydrophobicity': -3.5, 'pI': 2.77, 'mw': 115.09, 'volume': 111.1, 'charge': -1},
    'E': {'hydrophobicity': -3.5, 'pI': 3.22, 'mw': 129.12, 'volume': 138.4, 'charge': -1},
    'R': {'hydrophobicity': -4.5, 'pI': 10.76, 'mw': 156.19, 'volume': 202.1, 'charge': 1},
    'H': {'hydrophobicity': -3.2, 'pI': 7.59, 'mw': 137.14, 'volume': 153.2, 'charge': 0.5},
    'K': {'hydrophobicity': -3.9, 'pI': 9.74, 'mw': 128.17, 'volume': 168.6, 'charge': 1},
    'S': {'hydrophobicity': -0.8, 'pI': 5.68, 'mw': 87.08, 'volume': 89.0, 'charge': 0},
    'T': {'hydrophobicity': -0.7, 'pI': 5.60, 'mw': 101.11, 'volume': 116.1, 'charge': 0},
    'C': {'hydrophobicity': 2.5, 'pI': 5.07, 'mw': 103.14, 'volume': 108.5, 'charge': 0},
    'M': {'hydrophobicity': 1.9, 'pI': 5.74, 'mw': 131.19, 'volume': 162.9, 'charge': 0},
    'N': {'hydrophobicity': -3.5, 'pI': 5.41, 'mw': 114.10, 'volume': 114.1, 'charge': 0},
    'Q': {'hydrophobicity': -3.5, 'pI': 5.65, 'mw': 128.13, 'volume': 143.8, 'charge': 0},
    'P': {'hydrophobicity': -1.6, 'pI': 6.30, 'mw': 97.12, 'volume': 112.7, 'charge': 0},
}

# 氨基酸偶联难度评分 (基于文献和实验经验)
COUPLING_DIFFICULTY = {
    'A': 1.0, 'G': 1.0, 'S': 1.0,  # 容易
    'V': 1.5, 'L': 1.5, 'I': 1.8,  # 中等
    'F': 2.0, 'Y': 2.0, 'W': 2.5,  # 较难
    'P': 2.5,  # 脯氨酸特殊
    'D': 2.0, 'E': 2.0,  # 酸性氨基酸
    'K': 1.8, 'R': 2.8, 'H': 2.5,  # 碱性氨基酸 (Arg最难)
    'N': 2.0, 'Q': 2.0,  # 酰胺
    'C': 2.2, 'M': 2.0, 'T': 1.5,  # 含硫/羟基
}

# 三字母到单字母的转换
THREE_TO_ONE = {
    'Gly': 'G', 'Ala': 'A', 'Val': 'V', 'Ile': 'I', 'Leu': 'L',
    'Phe': 'F', 'Trp': 'W', 'Tyr': 'Y', 'Asp': 'D', 'Glu': 'E',
    'Arg': 'R', 'His': 'H', 'Lys': 'K', 'Ser': 'S', 'Thr': 'T',
    'Cys': 'C', 'Met': 'M', 'Asn': 'N', 'Gln': 'Q', 'Pro': 'P'
}


@dataclass
class PeptideFeatures:
    """肽特征数据结构"""
    sequence: str
    length: int
    avg_hydrophobicity: float
    total_charge: float
    molecular_weight: float
    avg_volume: float
    max_coupling_difficulty: float
    n_term_basic: bool
    c_term_acidic: bool
    

def parse_sequence(sequence: str) -> List[str]:
    """解析肽序列，支持单字母和三字母格式"""
    # 移除常见的修饰符
    sequence = sequence.replace('H-', '').replace('-OH', '').replace('-NH2', '')
    
    residues = []
    parts = [p.strip() for p in sequence.split('-') if p.strip()]
    
    for part in parts:
        if part in THREE_TO_ONE:
            residues.append(THREE_TO_ONE[part])
        elif len(part) == 1 and part in AMINO_ACID_PROPERTIES:
            residues.append(part)
        elif len(part) > 1:
            # 尝试解析连续的单字母序列
            for char in part:
                if char in AMINO_ACID_PROPERTIES:
                    residues.append(char)
    
    return residues


def calculate_peptide_features(sequence: str) -> PeptideFeatures:
    """计算肽的物理化学特征"""
    residues = parse_sequence(sequence)
    
    if not residues:
        return PeptideFeatures(sequence, 0, 0, 0, 0, 0, 0, False, False)
    
    # 计算平均性质
    hydrophobicities = [AMINO_ACID_PROPERTIES[r]['hydrophobicity'] for r in residues]
    charges = [AMINO_ACID_PROPERTIES[r]['charge'] for r in residues]
    mws = [AMINO_ACID_PROPERTIES[r]['mw'] for r in residues]
    volumes = [AMINO_ACID_PROPERTIES[r]['volume'] for r in residues]
    difficulties = [COUPLING_DIFFICULTY.get(r, 1.5) for r in residues]
    
    # N端和C端特征
    n_term_basic = residues[0] in ['R', 'K', 'H']
    c_term_acidic = residues[-1] in ['D', 'E']
    
    return PeptideFeatures(
        sequence=sequence,
        length=len(residues),
        avg_hydrophobicity=np.mean(hydrophobicities),
        total_charge=sum(charges),
        molecular_weight=sum(mws) - 18.015 * (len(residues) - 1),  # 减去水分子
        avg_volume=np.mean(volumes),
        max_coupling_difficulty=max(difficulties),
        n_term_basic=n_term_basic,
        c_term_acidic=c_term_acidic
    )


def augment_sequence(sequence: str, augmentation_factor: int = 3) -> List[str]:
    """生成序列变体用于数据增强"""
    residues = parse_sequence(sequence)
    if len(residues) < 3:
        return [sequence]
    
    variants = [sequence]
    
    for _ in range(augmentation_factor):
        variant = residues.copy()
        
        # 随机选择一种增强方法
        aug_type = random.choice(['swap', 'substitute', 'reverse'])
        
        if aug_type == 'swap' and len(variant) >= 4:
            # 交换相邻的相似性质氨基酸
            idx = random.randint(1, len(variant) - 2)
            if abs(AMINO_ACID_PROPERTIES[variant[idx]]['hydrophobicity'] - 
                   AMINO_ACID_PROPERTIES[variant[idx+1]]['hydrophobicity']) < 1.0:
                variant[idx], variant[idx+1] = variant[idx+1], variant[idx]
                
        elif aug_type == 'substitute':
            # 替换为性质相似的氨基酸
            idx = random.randint(0, len(variant) - 1)
            original = variant[idx]
            candidates = [
                aa for aa, props in AMINO_ACID_PROPERTIES.items()
                if abs(props['hydrophobicity'] - AMINO_ACID_PROPERTIES[original]['hydrophobicity']) < 0.5
                and aa != original
            ]
            if candidates:
                variant[idx] = random.choice(candidates)
                
        elif aug_type == 'reverse' and len(variant) >= 5:
            # 反转内部片段
            start = random.randint(1, len(variant) - 3)
            end = random.randint(start + 2, len(variant) - 1)
            variant[start:end] = reversed(variant[start:end])
        
        variants.append('-'.join(variant))
    
    return variants


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """生成合成训练数据"""
    data = []
    amino_acids = list(AMINO_ACID_PROPERTIES.keys())
    
    for _ in range(n_samples):
        # 随机生成序列长度 (5-20个氨基酸)
        length = random.randint(5, 20)
        sequence = '-'.join(random.choices(amino_acids, k=length))
        
        # 计算特征
        features = calculate_peptide_features(sequence)
        
        # 基于特征计算合成难度和预期纯度/收率
        difficulty_score = (
            features.max_coupling_difficulty * 0.3 +
            abs(features.total_charge) * 0.2 +
            (features.length / 20) * 0.3 +
            (1 if features.n_term_basic else 0) * 0.1 +
            (1 if features.c_term_acidic else 0) * 0.1
        )
        
        # 添加噪声
        noise = np.random.normal(0, 0.1)
        
        # 计算纯度和收率 (基于难度)
        base_purity = max(30, 100 - difficulty_score * 20 + noise * 10)
        base_yield = max(20, base_purity * 0.8 + noise * 5)
        
        # 随机选择合成条件
        coupling_reagent = random.choice(['HATU', 'HBTU', 'PyBOP', 'DIC/Oxyma'])
        solvent = random.choice(['DMF', 'NMP', 'DMF/DCM'])
        temperature = random.choice(['Room Temperature', 'Microwave 75°C', 'Microwave 90°C'])
        
        # 条件对结果的影响
        if coupling_reagent == 'HATU':
            base_purity += 5
            base_yield += 3
        if solvent == 'NMP':
            base_purity += 2
        if temperature == 'Microwave 90°C':
            base_purity += 3
            base_yield += 2
        
        data.append({
            'sequence': sequence,
            'length': features.length,
            'avg_hydrophobicity': features.avg_hydrophobicity,
            'total_charge': features.total_charge,
            'molecular_weight': features.molecular_weight,
            'max_coupling_difficulty': features.max_coupling_difficulty,
            'coupling_reagent': coupling_reagent,
            'solvent': solvent,
            'temperature': temperature,
            'predicted_purity': min(99.5, max(10, base_purity)),
            'predicted_yield': min(95, max(5, base_yield)),
            'difficulty_score': difficulty_score,
            'data_source': 'synthetic'
        })
    
    return pd.DataFrame(data)


def integrate_datasets():
    """整合所有可用数据集"""
    print("开始数据集整合...")
    
    datasets = []
    
    # 1. 加载文献数据
    try:
        literature_df = pd.read_csv('../data/real/final_purity_yield_literature.csv')
        print(f"加载文献数据: {len(literature_df)} 条记录")
        
        # 提取特征
        literature_features = []
        for _, row in literature_df.iterrows():
            features = calculate_peptide_features(row['sequence'])
            literature_features.append({
                'sequence': row['sequence'],
                'length': features.length,
                'avg_hydrophobicity': features.avg_hydrophobicity,
                'total_charge': features.total_charge,
                'molecular_weight': features.molecular_weight,
                'max_coupling_difficulty': features.max_coupling_difficulty,
                'coupling_reagent': 'HBTU',  # 默认值
                'solvent': 'NMP',
                'temperature': 'Room Temperature',
                'predicted_purity': row['purity_pct'],
                'predicted_yield': row['yield_pct'],
                'data_source': 'literature'
            })
        
        datasets.append(pd.DataFrame(literature_features))
    except Exception as e:
        print(f"文献数据加载失败: {e}")
    
    # 2. 加载实时合成数据
    try:
        synthesis_df = pd.read_csv('../data/real/synthesis_data.csv')
        print(f"加载合成数据: {len(synthesis_df)} 条记录")
        
        # 简化处理，取前1000条
        synthesis_sample = synthesis_df.head(1000)
        synthesis_features = []
        
        for _, row in synthesis_sample.iterrows():
            pre_chain = row['pre-chain']
            amino_acid = row['amino_acid']
            full_sequence = pre_chain + amino_acid
            
            features = calculate_peptide_features(full_sequence)
            synthesis_features.append({
                'sequence': full_sequence,
                'length': features.length,
                'avg_hydrophobicity': features.avg_hydrophobicity,
                'total_charge': features.total_charge,
                'molecular_weight': features.molecular_weight,
                'max_coupling_difficulty': features.max_coupling_difficulty,
                'coupling_reagent': row['coupling_agent'],
                'flow_rate': row['flow_rate'],
                'temp_coupling': row['temp_coupling'],
                'first_area': row['first_area'],
                'data_source': 'experimental'
            })
        
        datasets.append(pd.DataFrame(synthesis_features))
    except Exception as e:
        print(f"合成数据加载失败: {e}")
    
    # 3. 生成合成数据
    print("生成合成数据...")
    synthetic_df = generate_synthetic_data(n_samples=2000)
    datasets.append(synthetic_df)
    
    # 合并所有数据集
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"\n整合完成!")
    print(f"总记录数: {len(combined_df)}")
    print(f"数据来源分布:\n{combined_df['data_source'].value_counts()}")
    
    # 保存整合后的数据集
    output_path = '../data/enhanced_training_data.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\n已保存到: {output_path}")
    
    return combined_df


def generate_data_statistics(df: pd.DataFrame):
    """生成数据统计报告"""
    print("\n" + "="*60)
    print("数据集统计报告")
    print("="*60)
    
    print(f"\n总样本数: {len(df)}")
    print(f"特征数: {len(df.columns)}")
    
    print("\n数值特征统计:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    print("\n类别特征分布:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'sequence':
            print(f"\n{col}:")
            print(df[col].value_counts().head(10))
    
    print("\n序列长度分布:")
    print(df['length'].value_counts().sort_index().head(10))


if __name__ == '__main__':
    # 整合数据集
    combined_data = integrate_datasets()
    
    # 生成统计报告
    generate_data_statistics(combined_data)
    
    print("\n" + "="*60)
    print("数据集增强完成!")
    print("="*60)
