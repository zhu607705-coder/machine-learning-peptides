"""
多肽合成预测系统 - 架构包
==========================
可信度修复与二阶段架构升级

版本: 2.0.0
"""

from .core import (
    # 任务线
    TaskLine,
    TaskLineConfig,
    
    # 语义Head
    PurityStage,
    YieldStage,
    YieldBasisClass,
    SemanticHead,
    
    # 数据合同
    DataContract,
    CanonicalMapper,
    
    # 来源验证
    SourceValidator,
    SourceDiagnostics,
    
    # Head资格
    HeadEligibilityChecker,
    HeadEligibility,
)

from .feature_store import (
    FeatureExtractor,
    FeatureStore,
    FeatureBundle,
    SequenceGlobalFeatures,
    SequenceLocalRiskFeatures,
    ProcessContextFeatures,
    SemanticMetaFeatures,
    parse_sequence,
)

from .evaluation import (
    DataSplitter,
    HeadEvaluator,
    FoldResult,
    HeadEvaluationResult,
    EvaluationReportGenerator,
)

__all__ = [
    # 核心架构
    'TaskLine',
    'TaskLineConfig',
    'PurityStage',
    'YieldStage',
    'YieldBasisClass',
    'SemanticHead',
    'DataContract',
    'CanonicalMapper',
    'SourceValidator',
    'SourceDiagnostics',
    'HeadEligibilityChecker',
    'HeadEligibility',
    
    # 特征
    'FeatureExtractor',
    'FeatureStore',
    'FeatureBundle',
    'SequenceGlobalFeatures',
    'SequenceLocalRiskFeatures',
    'ProcessContextFeatures',
    'SemanticMetaFeatures',
    'parse_sequence',
    
    # 评估
    'DataSplitter',
    'HeadEvaluator',
    'FoldResult',
    'HeadEvaluationResult',
    'EvaluationReportGenerator',
]

__version__ = '2.0.0'
