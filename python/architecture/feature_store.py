"""
特征生产线 - Feature Store
===========================
统一特征生产，不再让训练脚本各自算特征

特征分组:
1. sequence_global: 全局序列特征
2. sequence_local_risk: 局部风险特征
3. process_context: 过程上下文特征
4. semantic_meta: 语义元特征

版本: 2.0.0
"""

import os
import sys
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

# 导入核心架构
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.core import (
    DataContract, TaskLine, TaskLineConfig, SemanticHead
)


# ============================================================================
# 氨基酸性质数据
# ============================================================================

AMINO_ACID_PROPERTIES = {
    'G': {'hydrophobicity': -0.4, 'pI': 5.97, 'mw': 57.05, 'volume': 60.1, 'charge': 0, 'bulky': False, 'polar': False, 'aromatic': False, 'sulfur': False},
    'A': {'hydrophobicity': 1.8, 'pI': 6.01, 'mw': 71.08, 'volume': 88.6, 'charge': 0, 'bulky': False, 'polar': False, 'aromatic': False, 'sulfur': False},
    'V': {'hydrophobicity': 4.2, 'pI': 5.97, 'mw': 99.13, 'volume': 140.0, 'charge': 0, 'bulky': True, 'polar': False, 'aromatic': False, 'sulfur': False},
    'I': {'hydrophobicity': 4.5, 'pI': 6.05, 'mw': 113.16, 'volume': 166.7, 'charge': 0, 'bulky': True, 'polar': False, 'aromatic': False, 'sulfur': False},
    'L': {'hydrophobicity': 3.8, 'pI': 5.98, 'mw': 113.16, 'volume': 166.7, 'charge': 0, 'bulky': True, 'polar': False, 'aromatic': False, 'sulfur': False},
    'F': {'hydrophobicity': 2.8, 'pI': 5.48, 'mw': 147.18, 'volume': 189.9, 'charge': 0, 'bulky': True, 'polar': False, 'aromatic': True, 'sulfur': False},
    'W': {'hydrophobicity': -0.9, 'pI': 5.89, 'mw': 186.21, 'volume': 227.8, 'charge': 0, 'bulky': True, 'polar': False, 'aromatic': True, 'sulfur': False},
    'Y': {'hydrophobicity': -1.3, 'pI': 5.66, 'mw': 163.18, 'volume': 193.6, 'charge': 0, 'bulky': True, 'polar': True, 'aromatic': True, 'sulfur': False},
    'D': {'hydrophobicity': -3.5, 'pI': 2.77, 'mw': 115.09, 'volume': 111.1, 'charge': -1, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'E': {'hydrophobicity': -3.5, 'pI': 3.22, 'mw': 129.12, 'volume': 138.4, 'charge': -1, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'R': {'hydrophobicity': -4.5, 'pI': 10.76, 'mw': 156.19, 'volume': 202.1, 'charge': 1, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'H': {'hydrophobicity': -3.2, 'pI': 7.59, 'mw': 137.14, 'volume': 153.2, 'charge': 0.5, 'bulky': False, 'polar': True, 'aromatic': True, 'sulfur': False},
    'K': {'hydrophobicity': -3.9, 'pI': 9.74, 'mw': 128.17, 'volume': 168.6, 'charge': 1, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'S': {'hydrophobicity': -0.8, 'pI': 5.68, 'mw': 87.08, 'volume': 89.0, 'charge': 0, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'T': {'hydrophobicity': -0.7, 'pI': 5.60, 'mw': 101.11, 'volume': 116.1, 'charge': 0, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'C': {'hydrophobicity': 2.5, 'pI': 5.07, 'mw': 103.14, 'volume': 108.5, 'charge': 0, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': True},
    'M': {'hydrophobicity': 1.9, 'pI': 5.74, 'mw': 131.19, 'volume': 162.9, 'charge': 0, 'bulky': False, 'polar': False, 'aromatic': False, 'sulfur': True},
    'N': {'hydrophobicity': -3.5, 'pI': 5.41, 'mw': 114.10, 'volume': 114.1, 'charge': 0, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'Q': {'hydrophobicity': -3.5, 'pI': 5.65, 'mw': 128.13, 'volume': 143.8, 'charge': 0, 'bulky': False, 'polar': True, 'aromatic': False, 'sulfur': False},
    'P': {'hydrophobicity': -1.6, 'pI': 6.30, 'mw': 97.12, 'volume': 112.7, 'charge': 0, 'bulky': False, 'polar': False, 'aromatic': False, 'sulfur': False},
}

# 三字母到单字母映射
THREE_TO_ONE = {
    'Gly': 'G', 'Ala': 'A', 'Val': 'V', 'Ile': 'I', 'Leu': 'L',
    'Phe': 'F', 'Trp': 'W', 'Tyr': 'Y', 'Asp': 'D', 'Glu': 'E',
    'Arg': 'R', 'His': 'H', 'Lys': 'K', 'Ser': 'S', 'Thr': 'T',
    'Cys': 'C', 'Met': 'M', 'Asn': 'N', 'Gln': 'Q', 'Pro': 'P'
}

# 偶联难度评分
COUPLING_DIFFICULTY = {
    'A': 1.0, 'G': 1.0, 'S': 1.0,
    'V': 1.5, 'L': 1.5, 'I': 1.8,
    'F': 2.0, 'Y': 2.0, 'W': 2.5,
    'P': 2.5,
    'D': 2.0, 'E': 2.0,
    'K': 1.8, 'R': 2.8, 'H': 2.5,
    'N': 2.0, 'Q': 2.0,
    'C': 2.2, 'M': 2.0, 'T': 1.5,
}

# 困难二肽/三肽motif
DIFFICULT_MOTIFS = {
    # 位阻冲突
    'VV': 1.5, 'II': 1.8, 'LL': 1.5, 'FF': 2.0, 'WW': 2.5,
    'VI': 1.5, 'IV': 1.5, 'LF': 1.8, 'FL': 1.8,
    # 脯氨酸相关
    'PP': 2.5, 'GP': 1.8, 'PG': 1.8,
    # Asp相关副反应风险
    'DG': 1.5, 'DN': 1.5, 'DA': 1.3,
    # Cys/Met敏感位点
    'CC': 2.0, 'CM': 1.8, 'MC': 1.8, 'MM': 1.5,
}


# ============================================================================
# 序列解析
# ============================================================================

def parse_sequence(sequence: str) -> List[str]:
    """解析肽序列，支持单字母和三字母格式"""
    sequence = sequence.replace('H-', '').replace('-OH', '').replace('-NH2', '')
    
    residues = []
    parts = [p.strip() for p in sequence.split('-') if p.strip()]
    
    for part in parts:
        if part in THREE_TO_ONE:
            residues.append(THREE_TO_ONE[part])
        elif len(part) == 1 and part in AMINO_ACID_PROPERTIES:
            residues.append(part)
        elif len(part) > 1:
            for char in part:
                if char in AMINO_ACID_PROPERTIES:
                    residues.append(char)
    
    return residues


# ============================================================================
# 特征组定义
# ============================================================================

@dataclass
class SequenceGlobalFeatures:
    """全局序列特征 (V1 - 已有)"""
    length: int
    avg_hydrophobicity: float
    total_charge: float
    molecular_weight: float
    avg_volume: float
    hydrophobic_ratio: float
    bulky_ratio: float
    polar_ratio: float
    charged_ratio: float
    aromatic_ratio: float
    sulfur_ratio: float
    max_coupling_difficulty: float
    longest_hydrophobic_run_norm: float
    n_term_basic: bool
    c_term_acidic: bool
    
    def to_dict(self) -> Dict:
        return {
            'length': self.length,
            'avg_hydrophobicity': self.avg_hydrophobicity,
            'total_charge': self.total_charge,
            'molecular_weight': self.molecular_weight,
            'avg_volume': self.avg_volume,
            'hydrophobic_ratio': self.hydrophobic_ratio,
            'bulky_ratio': self.bulky_ratio,
            'polar_ratio': self.polar_ratio,
            'charged_ratio': self.charged_ratio,
            'aromatic_ratio': self.aromatic_ratio,
            'sulfur_ratio': self.sulfur_ratio,
            'max_coupling_difficulty': self.max_coupling_difficulty,
            'longest_hydrophobic_run_norm': self.longest_hydrophobic_run_norm,
            'n_term_basic': self.n_term_basic,
            'c_term_acidic': self.c_term_acidic,
        }


@dataclass
class SequenceLocalRiskFeatures:
    """局部风险特征 (V2 - 新增)"""
    difficult_motif_count: int           # 困难二肽/三肽motif计数
    difficult_motif_max_score: float     # 最困难motif得分
    hydrophobic_cluster_count: int       # 疏水聚集窗口计数
    hydrophobic_cluster_positions: List[int]  # 疏水聚集位置
    asp_risk_score: float                # Asp相关副反应风险
    sulfur_sensitive_count: int          # Cys/Met敏感位点计数
    steric_conflict_windows: int         # 位阻冲突窗口计数
    charge_clustering_score: float       # 局部带电聚集评分
    
    def to_dict(self) -> Dict:
        return {
            'difficult_motif_count': self.difficult_motif_count,
            'difficult_motif_max_score': self.difficult_motif_max_score,
            'hydrophobic_cluster_count': self.hydrophobic_cluster_count,
            'asp_risk_score': self.asp_risk_score,
            'sulfur_sensitive_count': self.sulfur_sensitive_count,
            'steric_conflict_windows': self.steric_conflict_windows,
            'charge_clustering_score': self.charge_clustering_score,
        }


@dataclass
class ProcessContextFeatures:
    """过程上下文特征"""
    reagent_score: float
    solvent_score: float
    temperature_score: float
    cleavage_score: float
    
    def to_dict(self) -> Dict:
        return {
            'reagent_score': self.reagent_score,
            'solvent_score': self.solvent_score,
            'temperature_score': self.temperature_score,
            'cleavage_score': self.cleavage_score,
        }


@dataclass
class SemanticMetaFeatures:
    """语义元特征"""
    head_id: str
    source_id: str
    purity_stage: str
    yield_stage: str
    yield_basis_class: str
    topology: str
    
    def to_dict(self) -> Dict:
        return {
            'head_id': self.head_id,
            'source_id': self.source_id,
            'purity_stage': self.purity_stage,
            'yield_stage': self.yield_stage,
            'yield_basis_class': self.yield_basis_class,
            'topology': self.topology,
        }


@dataclass
class FeatureBundle:
    """完整特征包"""
    sequence_global: SequenceGlobalFeatures
    sequence_local_risk: SequenceLocalRiskFeatures
    process_context: ProcessContextFeatures
    semantic_meta: SemanticMetaFeatures
    
    # 特征向量
    feature_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 元数据
    sequence_hash: str = ""
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        result = {}
        result.update(self.sequence_global.to_dict())
        result.update(self.sequence_local_risk.to_dict())
        result.update(self.process_context.to_dict())
        result.update(self.semantic_meta.to_dict())
        result['sequence_hash'] = self.sequence_hash
        result['created_at'] = self.created_at
        return result


# ============================================================================
# 特征提取器
# ============================================================================

class FeatureExtractor:
    """特征提取器"""
    
    @staticmethod
    def compute_sequence_hash(sequence: str) -> str:
        """计算序列哈希"""
        return hashlib.md5(sequence.encode()).hexdigest()[:12]
    
    @classmethod
    def extract_global_features(cls, sequence: str) -> SequenceGlobalFeatures:
        """提取全局序列特征"""
        residues = parse_sequence(sequence)
        
        if not residues:
            return SequenceGlobalFeatures(
                length=0, avg_hydrophobicity=0, total_charge=0,
                molecular_weight=0, avg_volume=0, hydrophobic_ratio=0,
                bulky_ratio=0, polar_ratio=0, charged_ratio=0,
                aromatic_ratio=0, sulfur_ratio=0, max_coupling_difficulty=0,
                longest_hydrophobic_run_norm=0, n_term_basic=False, c_term_acidic=False
            )
        
        length = len(residues)
        
        # 计算平均性质
        hydrophobicities = [AMINO_ACID_PROPERTIES[r]['hydrophobicity'] for r in residues]
        charges = [AMINO_ACID_PROPERTIES[r]['charge'] for r in residues]
        mws = [AMINO_ACID_PROPERTIES[r]['mw'] for r in residues]
        volumes = [AMINO_ACID_PROPERTIES[r]['volume'] for r in residues]
        difficulties = [COUPLING_DIFFICULTY.get(r, 1.5) for r in residues]
        
        # 计算比例
        bulky_count = sum(1 for r in residues if AMINO_ACID_PROPERTIES[r]['bulky'])
        polar_count = sum(1 for r in residues if AMINO_ACID_PROPERTIES[r]['polar'])
        charged_count = sum(1 for r in residues if AMINO_ACID_PROPERTIES[r]['charge'] != 0)
        aromatic_count = sum(1 for r in residues if AMINO_ACID_PROPERTIES[r]['aromatic'])
        sulfur_count = sum(1 for r in residues if AMINO_ACID_PROPERTIES[r]['sulfur'])
        
        # 计算最长疏水段
        hydrophobic_residues = {'A', 'V', 'I', 'L', 'F', 'W', 'M', 'P'}
        max_run = 0
        current_run = 0
        for r in residues:
            if r in hydrophobic_residues:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        # N端和C端特征
        n_term_basic = residues[0] in ['R', 'K', 'H'] if residues else False
        c_term_acidic = residues[-1] in ['D', 'E'] if residues else False
        
        return SequenceGlobalFeatures(
            length=length,
            avg_hydrophobicity=np.mean(hydrophobicities),
            total_charge=sum(charges),
            molecular_weight=sum(mws) - 18.015 * (length - 1),
            avg_volume=np.mean(volumes),
            hydrophobic_ratio=sum(1 for r in residues if r in hydrophobic_residues) / length,
            bulky_ratio=bulky_count / length,
            polar_ratio=polar_count / length,
            charged_ratio=charged_count / length,
            aromatic_ratio=aromatic_count / length,
            sulfur_ratio=sulfur_count / length,
            max_coupling_difficulty=max(difficulties),
            longest_hydrophobic_run_norm=max_run / length if length > 0 else 0,
            n_term_basic=n_term_basic,
            c_term_acidic=c_term_acidic
        )
    
    @classmethod
    def extract_local_risk_features(cls, sequence: str) -> SequenceLocalRiskFeatures:
        """提取局部风险特征"""
        residues = parse_sequence(sequence)
        
        if len(residues) < 2:
            return SequenceLocalRiskFeatures(
                difficult_motif_count=0, difficult_motif_max_score=0,
                hydrophobic_cluster_count=0, hydrophobic_cluster_positions=[],
                asp_risk_score=0, sulfur_sensitive_count=0,
                steric_conflict_windows=0, charge_clustering_score=0
            )
        
        # 困难motif检测
        motif_count = 0
        max_motif_score = 0
        for i in range(len(residues) - 1):
            dipeptide = residues[i] + residues[i+1]
            if dipeptide in DIFFICULT_MOTIFS:
                motif_count += 1
                max_motif_score = max(max_motif_score, DIFFICULT_MOTIFS[dipeptide])
        
        # 疏水聚集窗口检测 (窗口大小=5)
        hydrophobic_residues = {'A', 'V', 'I', 'L', 'F', 'W', 'M'}
        cluster_count = 0
        cluster_positions = []
        window_size = 5
        threshold = 0.6
        
        for i in range(len(residues) - window_size + 1):
            window = residues[i:i+window_size]
            hydro_ratio = sum(1 for r in window if r in hydrophobic_residues) / window_size
            if hydro_ratio >= threshold:
                cluster_count += 1
                cluster_positions.append(i)
        
        # Asp相关副反应风险
        asp_risk = 0
        for i in range(len(residues) - 1):
            if residues[i] == 'D' and residues[i+1] in ['G', 'N', 'A']:
                asp_risk += 1
        
        # Cys/Met敏感位点
        sulfur_count = sum(1 for r in residues if r in ['C', 'M'])
        
        # 位阻冲突窗口 (连续大体积氨基酸)
        bulky_residues = {'V', 'I', 'L', 'F', 'W', 'Y'}
        steric_windows = 0
        for i in range(len(residues) - 2):
            window = residues[i:i+3]
            if sum(1 for r in window if r in bulky_residues) >= 2:
                steric_windows += 1
        
        # 局部带电聚集
        charge_clustering = 0
        for i in range(len(residues) - 3):
            window = residues[i:i+4]
            charges = [abs(AMINO_ACID_PROPERTIES[r]['charge']) for r in window]
            charge_clustering += sum(charges) / 4
        
        return SequenceLocalRiskFeatures(
            difficult_motif_count=motif_count,
            difficult_motif_max_score=max_motif_score,
            hydrophobic_cluster_count=cluster_count,
            hydrophobic_cluster_positions=cluster_positions,
            asp_risk_score=asp_risk / len(residues) if residues else 0,
            sulfur_sensitive_count=sulfur_count,
            steric_conflict_windows=steric_windows,
            charge_clustering_score=charge_clustering / len(residues) if residues else 0
        )
    
    @classmethod
    def extract_process_features(
        cls,
        coupling_reagent: str = 'HBTU',
        solvent: str = 'DMF',
        temperature: str = 'Room Temperature',
        cleavage_time: str = '2 hours'
    ) -> ProcessContextFeatures:
        """提取过程上下文特征"""
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
        
        return ProcessContextFeatures(
            reagent_score=reagent_score,
            solvent_score=solvent_score,
            temperature_score=temperature_score,
            cleavage_score=cleavage_score
        )
    
    @classmethod
    def create_feature_bundle(
        cls,
        contract: DataContract,
        embedding: Optional[np.ndarray] = None
    ) -> FeatureBundle:
        """创建完整特征包"""
        global_features = cls.extract_global_features(contract.sequence)
        local_risk_features = cls.extract_local_risk_features(contract.sequence)
        process_features = cls.extract_process_features(
            contract.coupling_reagent,
            contract.solvent,
            contract.temperature or 'Room Temperature',
            contract.cleavage_time or '2 hours'
        )
        semantic_features = SemanticMetaFeatures(
            head_id=contract.head_id,
            source_id=contract.source_id,
            purity_stage=contract.purity_stage.value,
            yield_stage=contract.yield_stage.value,
            yield_basis_class=contract.yield_basis_class.value,
            topology=contract.topology
        )
        
        # 构建特征向量
        feature_dict = {}
        feature_dict.update(global_features.to_dict())
        feature_dict.update(local_risk_features.to_dict())
        feature_dict.update(process_features.to_dict())
        
        # 排除非数值特征
        numeric_keys = [k for k, v in feature_dict.items() if isinstance(v, (int, float))]
        feature_vector = np.array([feature_dict[k] for k in numeric_keys])
        
        return FeatureBundle(
            sequence_global=global_features,
            sequence_local_risk=local_risk_features,
            process_context=process_features,
            semantic_meta=semantic_features,
            feature_vector=feature_vector,
            sequence_hash=cls.compute_sequence_hash(contract.sequence),
            created_at=datetime.now().isoformat()
        )


# ============================================================================
# Feature Store
# ============================================================================

class FeatureStore:
    """
    特征存储
    
    统一建设feature_store，特征分4组:
    - sequence_global
    - sequence_local_risk
    - process_context
    - semantic_meta
    """
    
    def __init__(self, store_path: str = '../data/feature_store'):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.cache: Dict[str, FeatureBundle] = {}
        self.version = "2.0.0"
    
    def get_or_create(
        self, 
        contract: DataContract,
        use_cache: bool = True
    ) -> FeatureBundle:
        """获取或创建特征"""
        cache_key = f"{contract.sequence}_{contract.head_id}"
        
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        features = FeatureExtractor.create_feature_bundle(contract)
        
        if use_cache:
            self.cache[cache_key] = features
        
        return features
    
    def batch_create(
        self, 
        contracts: List[DataContract],
        show_progress: bool = True
    ) -> List[FeatureBundle]:
        """批量创建特征"""
        features = []
        
        for i, contract in enumerate(contracts):
            if show_progress and (i + 1) % 100 == 0:
                print(f"  处理进度: {i+1}/{len(contracts)}")
            
            features.append(self.get_or_create(contract))
        
        return features
    
    def save(self, features: List[FeatureBundle], filename: str):
        """保存特征到文件"""
        data = [f.to_dict() for f in features]
        
        output_path = self.store_path / f"{filename}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'version': self.version,
                'created_at': datetime.now().isoformat(),
                'count': len(data),
                'features': data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 特征已保存: {output_path}")
    
    def load(self, filename: str) -> List[Dict]:
        """从文件加载特征"""
        input_path = self.store_path / f"{filename}.json"
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 特征已加载: {input_path} ({data['count']} 条)")
        return data['features']
    
    def to_dataframe(self, features: List[FeatureBundle]) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame([f.to_dict() for f in features])


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'FeatureExtractor',
    'FeatureStore',
    'FeatureBundle',
    'SequenceGlobalFeatures',
    'SequenceLocalRiskFeatures',
    'ProcessContextFeatures',
    'SemanticMetaFeatures',
    'parse_sequence',
    'AMINO_ACID_PROPERTIES',
    'COUPLING_DIFFICULTY',
    'DIFFICULT_MOTIFS',
]
