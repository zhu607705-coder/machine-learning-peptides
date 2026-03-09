"""
测试与验收脚本
==============
验证建模可信度与工程边界

测试内容:
1. 数据合同与评估协议测试
2. 特征存储测试
3. 分割泄漏测试
4. Embedding模式测试

运行: pytest tests/test_architecture.py -v

版本: 2.0.0
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from architecture.core import (
    TaskLine, TaskLineConfig, DataContract, CanonicalMapper,
    PurityStage, YieldStage, YieldBasisClass, SemanticHead,
    SourceValidator, HeadEligibilityChecker
)
from architecture.feature_store import (
    FeatureExtractor, FeatureStore, parse_sequence
)
from architecture.evaluation import DataSplitter


# ============================================================================
# 数据合同测试
# ============================================================================

class TestDataContract:
    """数据合同测试"""
    
    def test_valid_contract(self):
        """测试有效合同创建"""
        contract = DataContract(
            source_id="test_source",
            purity_stage=PurityStage.FINAL_PRODUCT,
            yield_stage=YieldStage.ISOLATED,
            yield_basis_class=YieldBasisClass.ISOLATED_MASS,
            topology="linear",
            coupling_reagent="HATU",
            solvent="DMF",
            sequence="GAVL",
            purity=95.0,
            yield_val=80.0
        )
        
        assert contract.source_id == "test_source"
        assert contract.purity == 95.0
        assert contract.yield_val == 80.0
        assert contract.head_id == "final_product_isolated_isolated_mass"
    
    def test_invalid_purity_range(self):
        """测试无效纯度范围"""
        with pytest.raises(ValueError, match="纯度值.*超出有效范围"):
            DataContract(
                source_id="test",
                purity_stage=PurityStage.FINAL_PRODUCT,
                yield_stage=YieldStage.ISOLATED,
                yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                topology="linear",
                coupling_reagent="HATU",
                solvent="DMF",
                sequence="GAVL",
                purity=150.0,  # 无效值
                yield_val=80.0
            )
    
    def test_invalid_yield_range(self):
        """测试无效收率范围"""
        with pytest.raises(ValueError, match="收率值.*超出有效范围"):
            DataContract(
                source_id="test",
                purity_stage=PurityStage.FINAL_PRODUCT,
                yield_stage=YieldStage.ISOLATED,
                yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                topology="linear",
                coupling_reagent="HATU",
                solvent="DMF",
                sequence="GAVL",
                purity=95.0,
                yield_val=-10.0  # 无效值
            )
    
    def test_empty_sequence(self):
        """测试空序列"""
        with pytest.raises(ValueError, match="序列不能为空"):
            DataContract(
                source_id="test",
                purity_stage=PurityStage.FINAL_PRODUCT,
                yield_stage=YieldStage.ISOLATED,
                yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                topology="linear",
                coupling_reagent="HATU",
                solvent="DMF",
                sequence="",  # 空序列
                purity=95.0,
                yield_val=80.0
            )


# ============================================================================
# Canonical Mapping测试
# ============================================================================

class TestCanonicalMapper:
    """Canonical映射测试"""
    
    def test_field_mapping(self):
        """测试字段名映射"""
        assert CanonicalMapper.map_field_name("purity_pct") == "purity"
        assert CanonicalMapper.map_field_name("yield_pct") == "yield_val"
        assert CanonicalMapper.map_field_name("yield") == "yield_val"
    
    def test_stage_mapping(self):
        """测试阶段值映射"""
        # 纯度阶段
        assert CanonicalMapper.map_stage("final_product", "purity") == PurityStage.FINAL_PRODUCT
        assert CanonicalMapper.map_stage("crude", "purity") == PurityStage.CRUDE
        
        # 收率阶段
        assert CanonicalMapper.map_stage("isolated", "yield") == YieldStage.ISOLATED
        
        # 收率基准
        assert CanonicalMapper.map_stage("resin_substitution", "yield_basis") == YieldBasisClass.RESIN_SUBSTITUTION
    
    def test_create_contract_from_row(self):
        """测试从数据行创建合同"""
        row = {
            "sequence": "GAVL",
            "purity_pct": 95.0,
            "yield_pct": 80.0,
            "purity_stage": "final_product",
            "yield_stage": "isolated",
            "yield_basis_class": "isolated_mass",
            "topology": "linear",
            "coupling_reagent": "HATU",
            "solvent": "DMF"
        }
        
        contract = CanonicalMapper.create_contract(row, "test_source")
        
        assert contract.sequence == "GAVL"
        assert contract.purity == 95.0
        assert contract.yield_val == 80.0
        assert contract.purity_stage == PurityStage.FINAL_PRODUCT
        assert contract.yield_stage == YieldStage.ISOLATED
    
    def test_unique_head_mapping(self):
        """测试每条样本必须落入唯一head"""
        row = {
            "sequence": "GAVL",
            "purity_pct": 95.0,
            "yield_pct": 80.0,
            "purity_stage": "final_product",
            "yield_stage": "isolated",
            "yield_basis_class": "isolated_mass",
            "topology": "linear",
            "coupling_reagent": "HATU",
            "solvent": "DMF"
        }
        
        contract = CanonicalMapper.create_contract(row, "test_source")
        
        # head_id必须是确定性的
        expected_head = "final_product_isolated_isolated_mass"
        assert contract.head_id == expected_head


# ============================================================================
# 分割测试
# ============================================================================

class TestDataSplitter:
    """数据分割测试"""
    
    def test_no_source_leakage(self):
        """测试来源无泄漏"""
        train_sources = {"source_a", "source_b"}
        test_sources = {"source_c", "source_d"}
        
        # 应该不抛出异常
        DataSplitter.validate_no_leakage(train_sources, test_sources)
    
    def test_source_leakage_detected(self):
        """测试来源泄漏检测"""
        train_sources = {"source_a", "source_b"}
        test_sources = {"source_b", "source_c"}  # source_b泄漏
        
        with pytest.raises(ValueError, match="来源泄漏检测"):
            DataSplitter.validate_no_leakage(train_sources, test_sources)
    
    def test_groupkfold_no_overlap(self):
        """测试GroupKFold无重叠"""
        # 创建测试数据
        contracts = []
        for source in ["A", "B", "C", "D", "E"]:
            for i in range(10):
                contracts.append(DataContract(
                    source_id=source,
                    purity_stage=PurityStage.FINAL_PRODUCT,
                    yield_stage=YieldStage.ISOLATED,
                    yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                    topology="linear",
                    coupling_reagent="HATU",
                    solvent="DMF",
                    sequence=f"GAVL{i}",
                    purity=90.0 + i,
                    yield_val=80.0 + i
                ))
        
        # 执行分割
        for fold_id, train_idx, test_idx, train_sources, test_sources in \
            DataSplitter.groupkfold_split(contracts, n_splits=5):
            
            # 验证无重叠
            assert len(train_sources & test_sources) == 0
            
            # 验证索引无重叠
            assert len(set(train_idx) & set(test_idx)) == 0
    
    def test_loso_no_overlap(self):
        """测试LOSO无重叠"""
        # 创建测试数据
        contracts = []
        for source in ["A", "B", "C"]:
            for i in range(5):
                contracts.append(DataContract(
                    source_id=source,
                    purity_stage=PurityStage.FINAL_PRODUCT,
                    yield_stage=YieldStage.ISOLATED,
                    yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                    topology="linear",
                    coupling_reagent="HATU",
                    solvent="DMF",
                    sequence=f"GAVL{i}",
                    purity=90.0 + i,
                    yield_val=80.0 + i
                ))
        
        # 执行LOSO分割
        for fold_id, train_idx, test_idx, train_sources, test_sources in \
            DataSplitter.loso_split(contracts):
            
            # 验证测试集只有一个来源
            assert len(test_sources) == 1
            
            # 验证训练集不包含测试来源
            assert len(train_sources & test_sources) == 0


# ============================================================================
# 特征存储测试
# ============================================================================

class TestFeatureStore:
    """特征存储测试"""
    
    def test_sequence_hash_consistency(self):
        """测试同一序列多次生成特征结果一致"""
        hash1 = FeatureExtractor.compute_sequence_hash("GAVL")
        hash2 = FeatureExtractor.compute_sequence_hash("GAVL")
        
        assert hash1 == hash2
    
    def test_sequence_hash_different(self):
        """测试不同序列生成不同哈希"""
        hash1 = FeatureExtractor.compute_sequence_hash("GAVL")
        hash2 = FeatureExtractor.compute_sequence_hash("GAVLI")
        
        assert hash1 != hash2
    
    def test_feature_extraction(self):
        """测试特征提取"""
        contract = DataContract(
            source_id="test",
            purity_stage=PurityStage.FINAL_PRODUCT,
            yield_stage=YieldStage.ISOLATED,
            yield_basis_class=YieldBasisClass.ISOLATED_MASS,
            topology="linear",
            coupling_reagent="HATU",
            solvent="DMF",
            sequence="GAVL",
            purity=95.0,
            yield_val=80.0
        )
        
        features = FeatureExtractor.create_feature_bundle(contract)
        
        # 验证特征向量非空
        assert len(features.feature_vector) > 0
        
        # 验证全局特征
        assert features.sequence_global.length == 4
        assert features.sequence_global.molecular_weight > 0
        
        # 验证过程特征
        assert features.process_context.reagent_score == 1.0  # HATU
    
    def test_embedding_off_no_residual(self):
        """测试embedding关闭时不会残留旧embedding列"""
        contract = DataContract(
            source_id="test",
            purity_stage=PurityStage.FINAL_PRODUCT,
            yield_stage=YieldStage.ISOLATED,
            yield_basis_class=YieldBasisClass.ISOLATED_MASS,
            topology="linear",
            coupling_reagent="HATU",
            solvent="DMF",
            sequence="GAVL",
            purity=95.0,
            yield_val=80.0
        )
        
        # 不传入embedding
        features = FeatureExtractor.create_feature_bundle(contract, embedding=None)
        
        # 验证特征字典中没有embedding相关列
        feature_dict = features.to_dict()
        assert "embedding" not in feature_dict
        assert "esm_embedding" not in feature_dict


# ============================================================================
# Head资格测试
# ============================================================================

class TestHeadEligibility:
    """Head训练资格测试"""
    
    def test_eligible_head(self):
        """测试符合条件的head"""
        contracts = []
        for source in ["A", "B", "C", "D"]:
            for i in range(10):
                contracts.append(DataContract(
                    source_id=source,
                    purity_stage=PurityStage.FINAL_PRODUCT,
                    yield_stage=YieldStage.ISOLATED,
                    yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                    topology="linear",
                    coupling_reagent="HATU",
                    solvent="DMF",
                    sequence=f"GAVL{i}",
                    purity=90.0 + i,
                    yield_val=80.0 + i
                ))
        
        eligibility = HeadEligibilityChecker.check_eligibility(
            contracts, "final_product_isolated_isolated_mass"
        )
        
        assert eligibility.is_eligible
        assert eligibility.rows == 40
        assert eligibility.sources == 4
        assert eligibility.max_source_share == 0.25
    
    def test_ineligible_head_insufficient_rows(self):
        """测试样本不足的head"""
        contracts = []
        for i in range(10):  # 少于MIN_ROWS=20
            contracts.append(DataContract(
                source_id=f"source_{i}",
                purity_stage=PurityStage.FINAL_PRODUCT,
                yield_stage=YieldStage.ISOLATED,
                yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                topology="linear",
                coupling_reagent="HATU",
                solvent="DMF",
                sequence=f"GAVL{i}",
                purity=90.0 + i,
                yield_val=80.0 + i
            ))
        
        eligibility = HeadEligibilityChecker.check_eligibility(
            contracts, "final_product_isolated_isolated_mass"
        )
        
        assert not eligibility.is_eligible
        assert "样本不足" in str(eligibility.issues)
    
    def test_ineligible_head_insufficient_sources(self):
        """测试来源不足的head"""
        contracts = []
        for i in range(30):  # 样本足够
            contracts.append(DataContract(
                source_id="single_source",  # 只有一个来源
                purity_stage=PurityStage.FINAL_PRODUCT,
                yield_stage=YieldStage.ISOLATED,
                yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                topology="linear",
                coupling_reagent="HATU",
                solvent="DMF",
                sequence=f"GAVL{i}",
                purity=90.0 + (i % 10),  # 保持在有效范围内
                yield_val=80.0 + (i % 10)
            ))
        
        eligibility = HeadEligibilityChecker.check_eligibility(
            contracts, "final_product_isolated_isolated_mass"
        )
        
        assert not eligibility.is_eligible
        # 检查是否包含来源不足或单一来源占比过高的问题
        issues_str = str(eligibility.issues)
        assert "来源不足" in issues_str or "单一来源占比过高" in issues_str
    
    def test_ineligible_head_high_source_share(self):
        """测试单一来源占比过高的head"""
        contracts = []
        
        # 主导来源 (80%)
        for i in range(40):
            contracts.append(DataContract(
                source_id="dominant_source",
                purity_stage=PurityStage.FINAL_PRODUCT,
                yield_stage=YieldStage.ISOLATED,
                yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                topology="linear",
                coupling_reagent="HATU",
                solvent="DMF",
                sequence=f"GAVL{i}",
                purity=90.0 + (i % 10),  # 保持在有效范围内
                yield_val=80.0 + (i % 10)
            ))
        
        # 少量其他来源
        for source in ["A", "B"]:
            for i in range(5):
                contracts.append(DataContract(
                    source_id=source,
                    purity_stage=PurityStage.FINAL_PRODUCT,
                    yield_stage=YieldStage.ISOLATED,
                    yield_basis_class=YieldBasisClass.ISOLATED_MASS,
                    topology="linear",
                    coupling_reagent="HATU",
                    solvent="DMF",
                    sequence=f"GAVL{i}",
                    purity=90.0 + (i % 10),  # 保持在有效范围内
                    yield_val=80.0 + (i % 10)
                ))
        
        eligibility = HeadEligibilityChecker.check_eligibility(
            contracts, "final_product_isolated_isolated_mass"
        )
        
        assert not eligibility.is_eligible
        assert "单一来源占比过高" in str(eligibility.issues)


# ============================================================================
# 任务线配置测试
# ============================================================================

class TestTaskLineConfig:
    """任务线配置测试"""
    
    def test_step_proxy_config(self):
        """测试step_proxy任务线配置"""
        config = TaskLineConfig.get_config(TaskLine.STEP_PROXY)
        
        assert config["eval_protocol"] == "grouped_by_serial"
        assert "first_area" in config["target_variables"]
        assert config["deployable"] == "demo_only"
    
    def test_result_head_config(self):
        """测试result_head任务线配置"""
        config = TaskLineConfig.get_config(TaskLine.RESULT_HEAD)
        
        assert config["eval_protocol"] == "loso_or_groupkfold"
        assert "purity" in config["target_variables"]
        assert config["min_rows"] == 20
        assert config["min_sources"] == 3
        assert config["max_source_share"] == 0.70
    
    def test_valid_target(self):
        """测试有效目标检查"""
        assert TaskLineConfig.is_valid_target(TaskLine.STEP_PROXY, "first_area")
        assert TaskLineConfig.is_valid_target(TaskLine.RESULT_HEAD, "purity")
        assert not TaskLineConfig.is_valid_target(TaskLine.STEP_PROXY, "purity")


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
