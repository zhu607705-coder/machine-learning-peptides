"""
多肽合成预测系统 - 核心架构定义
=====================================
三层任务栈架构 + 数据合同层 + 评估协议

任务线定义:
1. step_proxy: 步骤级代理信号预测 (面向peptimizer)
2. result_head: 结果级语义head回归 (面向文献数据)
3. sop_case: Fmoc SOP收缩子集 (补充验证案例)

作者: Peptide Synthesis Predictor Team
版本: 2.0.0
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import json
import hashlib


# ============================================================================
# 任务线定义
# ============================================================================

class TaskLine(Enum):
    """三层任务栈定义"""
    STEP_PROXY = "step_proxy"       # 步骤级代理信号预测
    RESULT_HEAD = "result_head"     # 结果级语义head回归
    SOP_CASE = "sop_case"           # Fmoc SOP收缩子集


class TaskLineConfig:
    """任务线配置"""
    
    CONFIGS = {
        TaskLine.STEP_PROXY: {
            "description": "面向peptimizer的步骤级代理信号预测",
            "target_variables": ["first_area", "first_height", "first_width", "first_diff"],
            "eval_protocol": "grouped_by_serial",
            "embedding_mode": "optional",
            "min_rows": 100,
            "min_sources": 1,
            "max_source_share": 1.0,
            "deployable": "demo_only",
            "note": "只负责过程风险感知，不是终点指标预测"
        },
        TaskLine.RESULT_HEAD: {
            "description": "面向文献结果级数据的严格语义head回归",
            "target_variables": ["purity", "yield"],
            "eval_protocol": "loso_or_groupkfold",
            "embedding_mode": "conditional",
            "min_rows": 20,
            "min_sources": 3,
            "max_source_share": 0.70,
            "deployable": "trainable",
            "note": "只负责可定义head的LOSO评估"
        },
        TaskLine.SOP_CASE: {
            "description": "面向Fmoc SOP收缩子集的补充案例",
            "target_variables": ["purity", "yield"],
            "eval_protocol": "groupkfold",
            "embedding_mode": "on",
            "min_rows": 50,
            "min_sources": 1,
            "max_source_share": 1.0,
            "deployable": "validation_only",
            "note": "只用于验证任务边界收缩后的模型行为"
        }
    }
    
    @classmethod
    def get_config(cls, task_line: TaskLine) -> Dict:
        return cls.CONFIGS.get(task_line, {})
    
    @classmethod
    def is_valid_target(cls, task_line: TaskLine, target: str) -> bool:
        config = cls.get_config(task_line)
        return target in config.get("target_variables", [])


# ============================================================================
# 语义Head定义
# ============================================================================

class PurityStage(Enum):
    """纯度测量阶段"""
    CRUDE = "crude"                           # 粗品
    CRUDE_HPLC_214NM = "crude_hplc_214nm"     # 粗品HPLC 214nm
    CRUDE_HPLC_280NM = "crude_hplc_280nm"     # 粗品HPLC 280nm
    PURIFIED_HPLC = "purified_hplc"           # 纯化后HPLC
    FINAL_PRODUCT = "final_product"           # 最终产品
    UNKNOWN = "unknown"


class YieldStage(Enum):
    """收率测量阶段"""
    CRUDE = "crude"                           # 粗品收率
    ISOLATED = "isolated"                     # 分离收率
    RECOVERY = "recovery"                     # 回收率
    PURIFIED = "purified"                     # 纯化后收率
    FINAL_PRODUCT = "final_product"           # 最终产品收率
    UNKNOWN = "unknown"


class YieldBasisClass(Enum):
    """收率计算基准"""
    ISOLATED_MASS = "isolated_mass"           # 分离质量
    CRUDE_WEIGHT = "crude_weight"             # 粗品重量
    RESIN_SUBSTITUTION = "resin_substitution" # 树脂取代量
    THEORETICAL = "theoretical"               # 理论量
    UNKNOWN = "unknown"


@dataclass
class SemanticHead:
    """语义Head定义"""
    purity_stage: PurityStage
    yield_stage: YieldStage
    yield_basis_class: YieldBasisClass
    
    @property
    def head_id(self) -> str:
        """生成唯一的head标识符"""
        return f"{self.purity_stage.value}_{self.yield_stage.value}_{self.yield_basis_class.value}"
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticHead':
        return cls(
            purity_stage=PurityStage(data.get("purity_stage", "unknown")),
            yield_stage=YieldStage(data.get("yield_stage", "unknown")),
            yield_basis_class=YieldBasisClass(data.get("yield_basis_class", "unknown"))
        )
    
    def to_dict(self) -> Dict:
        return {
            "purity_stage": self.purity_stage.value,
            "yield_stage": self.yield_stage.value,
            "yield_basis_class": self.yield_basis_class.value,
            "head_id": self.head_id
        }


# ============================================================================
# 数据合同层
# ============================================================================

@dataclass
class DataContract:
    """
    数据合同 - 显式固化字段定义
    
    所有训练脚本不得直接读取原始CSV自由拼接标签，
    必须先经过canonical mapping
    """
    source_id: str                              # 数据来源ID
    purity_stage: PurityStage                   # 纯度测量阶段
    yield_stage: YieldStage                     # 收率测量阶段
    yield_basis_class: YieldBasisClass          # 收率计算基准
    topology: str                               # 拓扑结构
    coupling_reagent: str                       # 偶联试剂
    solvent: str                                # 溶剂
    sequence: str                               # 肽序列
    purity: Optional[float] = None              # 纯度值
    yield_val: Optional[float] = None           # 收率值
    
    # 元数据
    record_id: Optional[str] = None
    temperature: Optional[str] = None
    cleavage_time: Optional[str] = None
    
    def __post_init__(self):
        """验证数据合同"""
        self._validate()
    
    def _validate(self):
        """验证字段有效性"""
        # 验证纯度范围
        if self.purity is not None:
            if not (0 <= self.purity <= 100):
                raise ValueError(f"纯度值 {self.purity} 超出有效范围 [0, 100]")
        
        # 验证收率范围
        if self.yield_val is not None:
            if not (0 <= self.yield_val <= 100):
                raise ValueError(f"收率值 {self.yield_val} 超出有效范围 [0, 100]")
        
        # 验证序列非空
        if not self.sequence or len(self.sequence.strip()) == 0:
            raise ValueError("序列不能为空")
    
    @property
    def semantic_head(self) -> SemanticHead:
        """获取语义Head"""
        return SemanticHead(
            purity_stage=self.purity_stage,
            yield_stage=self.yield_stage,
            yield_basis_class=self.yield_basis_class
        )
    
    @property
    def head_id(self) -> str:
        """获取head标识符"""
        return self.semantic_head.head_id
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "source_id": self.source_id,
            "purity_stage": self.purity_stage.value,
            "yield_stage": self.yield_stage.value,
            "yield_basis_class": self.yield_basis_class.value,
            "topology": self.topology,
            "coupling_reagent": self.coupling_reagent,
            "solvent": self.solvent,
            "sequence": self.sequence,
            "purity": self.purity,
            "yield_val": self.yield_val,
            "record_id": self.record_id,
            "temperature": self.temperature,
            "cleavage_time": self.cleavage_time,
            "head_id": self.head_id
        }


# ============================================================================
# Canonical Mapping
# ============================================================================

class CanonicalMapper:
    """
    Canonical映射器 - 统一字段映射
    
    所有训练脚本必须通过此映射器处理数据
    """
    
    # 字段名映射表
    FIELD_MAPPINGS = {
        'purity_pct': 'purity',
        'yield_pct': 'yield_val',
        'yield': 'yield_val',
        'purity_stage': 'purity_stage',
        'yield_stage': 'yield_stage',
        'yield_basis': 'yield_basis_class',
        'yield_basis_class': 'yield_basis_class',
    }
    
    # 默认值映射
    DEFAULT_VALUES = {
        'purity_stage': PurityStage.UNKNOWN,
        'yield_stage': YieldStage.UNKNOWN,
        'yield_basis_class': YieldBasisClass.UNKNOWN,
        'topology': 'linear',
        'coupling_reagent': 'HBTU',
        'solvent': 'DMF',
    }
    
    # 标签语义映射 - 分离不同类型的映射
    PURITY_STAGE_MAPPINGS = {
        'final_product': PurityStage.FINAL_PRODUCT,
        'final': PurityStage.FINAL_PRODUCT,
        'purified': PurityStage.PURIFIED_HPLC,
        'purified_hplc': PurityStage.PURIFIED_HPLC,
        'crude': PurityStage.CRUDE,
        'crude_hplc': PurityStage.CRUDE_HPLC_214NM,
        'crude_hplc_214nm': PurityStage.CRUDE_HPLC_214NM,
        'crude_hplc_280nm': PurityStage.CRUDE_HPLC_280NM,
    }
    
    YIELD_STAGE_MAPPINGS = {
        'isolated': YieldStage.ISOLATED,
        'crude': YieldStage.CRUDE,
        'recovery': YieldStage.RECOVERY,
        'purified': YieldStage.PURIFIED,
        'final_product': YieldStage.FINAL_PRODUCT,
    }
    
    YIELD_BASIS_MAPPINGS = {
        'isolated_mass': YieldBasisClass.ISOLATED_MASS,
        'isolated': YieldBasisClass.ISOLATED_MASS,
        'crude_weight': YieldBasisClass.CRUDE_WEIGHT,
        'resin_substitution': YieldBasisClass.RESIN_SUBSTITUTION,
        'theoretical': YieldBasisClass.THEORETICAL,
    }
    
    @classmethod
    def map_field_name(cls, field_name: str) -> str:
        """映射字段名"""
        normalized = field_name.lower().strip()
        return cls.FIELD_MAPPINGS.get(normalized, field_name)
    
    @classmethod
    def map_stage(cls, value: str, stage_type: str) -> Any:
        """映射阶段值"""
        normalized = value.lower().strip().replace(' ', '_')
        
        if stage_type == 'purity':
            return cls.PURITY_STAGE_MAPPINGS.get(normalized, PurityStage.UNKNOWN)
        elif stage_type == 'yield':
            return cls.YIELD_STAGE_MAPPINGS.get(normalized, YieldStage.UNKNOWN)
        elif stage_type == 'yield_basis':
            return cls.YIELD_BASIS_MAPPINGS.get(normalized, YieldBasisClass.UNKNOWN)
        
        return value
    
    @classmethod
    def create_contract(cls, row: Dict, source_id: str) -> DataContract:
        """
        从原始数据行创建数据合同
        
        Args:
            row: 原始数据行
            source_id: 数据来源ID
            
        Returns:
            DataContract实例
        """
        # 映射字段名
        mapped_row = {}
        for key, value in row.items():
            mapped_key = cls.map_field_name(key)
            mapped_row[mapped_key] = value
        
        # 解析阶段值
        purity_stage_raw = str(mapped_row.get('purity_stage', 'unknown')).lower()
        yield_stage_raw = str(mapped_row.get('yield_stage', 'unknown')).lower()
        yield_basis_raw = str(mapped_row.get('yield_basis_class', mapped_row.get('yield_basis', 'unknown'))).lower()
        
        purity_stage = cls.map_stage(purity_stage_raw, 'purity')
        yield_stage = cls.map_stage(yield_stage_raw, 'yield')
        yield_basis_class = cls.map_stage(yield_basis_raw, 'yield_basis')
        
        # 创建数据合同
        return DataContract(
            source_id=source_id,
            purity_stage=purity_stage if isinstance(purity_stage, PurityStage) else PurityStage.UNKNOWN,
            yield_stage=yield_stage if isinstance(yield_stage, YieldStage) else YieldStage.UNKNOWN,
            yield_basis_class=yield_basis_class if isinstance(yield_basis_class, YieldBasisClass) else YieldBasisClass.UNKNOWN,
            topology=str(mapped_row.get('topology', 'linear')),
            coupling_reagent=str(mapped_row.get('coupling_reagent', 'HBTU')),
            solvent=str(mapped_row.get('solvent', 'DMF')),
            sequence=str(mapped_row.get('sequence', '')),
            purity=mapped_row.get('purity'),
            yield_val=mapped_row.get('yield_val'),
            record_id=mapped_row.get('record_id'),
            temperature=mapped_row.get('temperature'),
            cleavage_time=mapped_row.get('cleavage_time')
        )


# ============================================================================
# 来源诊断
# ============================================================================

@dataclass
class SourceDiagnostics:
    """来源诊断结果"""
    source_id: str
    sample_count: int
    label_variance: Dict[str, float]
    head_distribution: Dict[str, int]
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "sample_count": self.sample_count,
            "label_variance": self.label_variance,
            "head_distribution": self.head_distribution,
            "is_valid": self.is_valid,
            "issues": self.issues
        }


class SourceValidator:
    """
    来源验证器 - 诊断低价值来源
    
    诊断条件:
    1. 样本过少
    2. 标签方差过低
    3. 单一来源占比过高
    4. 来源内部标签口径不自洽
    """
    
    # 验证阈值
    MIN_SAMPLES = 5
    MIN_VARIANCE = 1.0
    MAX_SINGLE_HEAD_SHARE = 0.90
    
    @classmethod
    def diagnose_source(
        cls, 
        contracts: List[DataContract], 
        source_id: str
    ) -> SourceDiagnostics:
        """诊断单个来源"""
        source_contracts = [c for c in contracts if c.source_id == source_id]
        
        issues = []
        sample_count = len(source_contracts)
        
        # 检查样本数量
        if sample_count < cls.MIN_SAMPLES:
            issues.append(f"样本过少: {sample_count} < {cls.MIN_SAMPLES}")
        
        # 计算标签方差
        label_variance = {}
        for target in ['purity', 'yield_val']:
            values = [getattr(c, target) for c in source_contracts if getattr(c, target) is not None]
            if len(values) >= 2:
                variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
                label_variance[target] = variance
                
                if variance < cls.MIN_VARIANCE:
                    issues.append(f"{target}方差过低: {variance:.2f} < {cls.MIN_VARIANCE}")
        
        # 计算head分布
        head_counts = {}
        for c in source_contracts:
            head_id = c.head_id
            head_counts[head_id] = head_counts.get(head_id, 0) + 1
        
        # 检查单一head占比
        if head_counts:
            max_head_count = max(head_counts.values())
            max_head_share = max_head_count / sample_count
            if max_head_share > cls.MAX_SINGLE_HEAD_SHARE and sample_count > 5:
                issues.append(f"单一head占比过高: {max_head_share:.1%} > {cls.MAX_SINGLE_HEAD_SHARE:.1%}")
        
        # 检查标签口径自洽性
        unique_purity_stages = set(c.purity_stage for c in source_contracts)
        unique_yield_stages = set(c.yield_stage for c in source_contracts)
        
        if len(unique_purity_stages) > 3:
            issues.append(f"纯度阶段不自洽: {len(unique_purity_stages)} 种不同阶段")
        if len(unique_yield_stages) > 3:
            issues.append(f"收率阶段不自洽: {len(unique_yield_stages)} 种不同阶段")
        
        is_valid = len(issues) == 0
        
        return SourceDiagnostics(
            source_id=source_id,
            sample_count=sample_count,
            label_variance=label_variance,
            head_distribution=head_counts,
            is_valid=is_valid,
            issues=issues
        )
    
    @classmethod
    def validate_all_sources(
        cls, 
        contracts: List[DataContract]
    ) -> Tuple[Dict[str, SourceDiagnostics], List[str]]:
        """验证所有来源"""
        source_ids = set(c.source_id for c in contracts)
        
        diagnostics = {}
        invalid_sources = []
        
        for source_id in source_ids:
            diag = cls.diagnose_source(contracts, source_id)
            diagnostics[source_id] = diag
            if not diag.is_valid:
                invalid_sources.append(source_id)
        
        return diagnostics, invalid_sources


# ============================================================================
# Head训练门槛
# ============================================================================

@dataclass
class HeadEligibility:
    """Head训练资格评估"""
    head_id: str
    rows: int
    sources: int
    max_source_share: float
    target_semantics: Dict
    is_eligible: bool
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "head_id": self.head_id,
            "rows": self.rows,
            "sources": self.sources,
            "max_source_share": self.max_source_share,
            "target_semantics": self.target_semantics,
            "is_eligible": self.is_eligible,
            "issues": self.issues
        }


class HeadEligibilityChecker:
    """
    Head训练资格检查器
    
    默认训练门槛:
    - rows >= 20
    - sources >= 3
    - max_source_share <= 0.70
    """
    
    # 训练门槛
    MIN_ROWS = 20
    MIN_SOURCES = 3
    MAX_SOURCE_SHARE = 0.70
    
    @classmethod
    def check_eligibility(
        cls, 
        contracts: List[DataContract], 
        head_id: str
    ) -> HeadEligibility:
        """检查head训练资格"""
        head_contracts = [c for c in contracts if c.head_id == head_id]
        
        issues = []
        rows = len(head_contracts)
        
        # 统计来源
        source_counts = {}
        for c in head_contracts:
            source_counts[c.source_id] = source_counts.get(c.source_id, 0) + 1
        
        sources = len(source_counts)
        max_source_share = max(source_counts.values()) / rows if rows > 0 else 0
        
        # 检查门槛
        if rows < cls.MIN_ROWS:
            issues.append(f"样本不足: {rows} < {cls.MIN_ROWS}")
        
        if sources < cls.MIN_SOURCES:
            issues.append(f"来源不足: {sources} < {cls.MIN_SOURCES}")
        
        if max_source_share > cls.MAX_SOURCE_SHARE:
            issues.append(f"单一来源占比过高: {max_source_share:.1%} > {cls.MAX_SOURCE_SHARE:.1%}")
        
        # 获取目标语义
        if head_contracts:
            first = head_contracts[0]
            target_semantics = {
                "purity_stage": first.purity_stage.value,
                "yield_stage": first.yield_stage.value,
                "yield_basis_class": first.yield_basis_class.value
            }
        else:
            target_semantics = {}
        
        is_eligible = len(issues) == 0
        
        return HeadEligibility(
            head_id=head_id,
            rows=rows,
            sources=sources,
            max_source_share=max_source_share,
            target_semantics=target_semantics,
            is_eligible=is_eligible,
            issues=issues
        )
    
    @classmethod
    def check_all_heads(
        cls, 
        contracts: List[DataContract]
    ) -> Dict[str, HeadEligibility]:
        """检查所有head的训练资格"""
        head_ids = set(c.head_id for c in contracts)
        
        eligibilities = {}
        for head_id in head_ids:
            eligibilities[head_id] = cls.check_eligibility(contracts, head_id)
        
        return eligibilities


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 任务线
    'TaskLine',
    'TaskLineConfig',
    
    # 语义Head
    'PurityStage',
    'YieldStage',
    'YieldBasisClass',
    'SemanticHead',
    
    # 数据合同
    'DataContract',
    'CanonicalMapper',
    
    # 来源验证
    'SourceValidator',
    'SourceDiagnostics',
    
    # Head资格
    'HeadEligibilityChecker',
    'HeadEligibility',
]
