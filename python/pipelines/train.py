"""
端到端训练脚本 - 可信度修复版
==============================
按照新的架构规范执行训练流程

使用方法:
    python -m pipelines.train_result_head --head=isolated_isolated
    python -m pipelines.train_result_head --task=result_head --eval=loso
    python -m pipelines.train_step_proxy --arch=gru_residual_small --loss=mse

版本: 2.0.0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入架构模块
from architecture.core import (
    TaskLine, TaskLineConfig, DataContract, CanonicalMapper,
    SourceValidator, HeadEligibilityChecker, HeadEligibility,
    PurityStage, YieldStage, YieldBasisClass
)
from architecture.feature_store import FeatureStore, FeatureExtractor
from architecture.evaluation import (
    HeadEvaluator, DataSplitter, EvaluationReportGenerator
)


# ============================================================================
# 数据加载器
# ============================================================================

class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_literature_data(filepath: str) -> pd.DataFrame:
        """加载文献数据"""
        df = pd.read_csv(filepath)
        print(f"✓ 加载文献数据: {len(df)} 条记录")
        return df
    
    @staticmethod
    def load_peptimizer_data(filepath: str) -> pd.DataFrame:
        """加载peptimizer步骤级数据"""
        df = pd.read_csv(filepath)
        print(f"✓ 加载peptimizer数据: {len(df)} 条记录")
        return df
    
    @staticmethod
    def create_contracts_from_literature(
        df: pd.DataFrame,
        source_prefix: str = "lit"
    ) -> List[DataContract]:
        """从文献数据创建数据合同"""
        contracts = []
        
        for idx, row in df.iterrows():
            try:
                # 提取source_id
                source_id = row.get('source_id', f"{source_prefix}_{idx}")
                
                # 使用CanonicalMapper创建合同
                contract = CanonicalMapper.create_contract(row.to_dict(), source_id)
                contracts.append(contract)
            except Exception as e:
                print(f"  ⚠ 跳过记录 {idx}: {e}")
                continue
        
        print(f"✓ 创建数据合同: {len(contracts)} 条")
        return contracts


# ============================================================================
# 结果级Head训练器
# ============================================================================

class ResultHeadTrainer:
    """
    结果级Head训练器
    
    严格遵循:
    - head-first建模
    - source-aware分割
    - 完整评估报告
    """
    
    def __init__(
        self,
        output_dir: str = "../models/result_head",
        reports_dir: str = "../reports"
    ):
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_store = FeatureStore()
        self.evaluator = HeadEvaluator()
    
    def prepare_data(
        self,
        contracts: List[DataContract],
        head_id: Optional[str] = None
    ) -> Tuple[List[DataContract], np.ndarray, np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 按head过滤
        if head_id:
            contracts = [c for c in contracts if c.head_id == head_id]
        
        # 检查训练资格
        if head_id:
            eligibility = HeadEligibilityChecker.check_eligibility(contracts, head_id)
            print(f"\nHead资格检查: {head_id}")
            print(f"  样本数: {eligibility.rows}")
            print(f"  来源数: {eligibility.sources}")
            print(f"  最大来源占比: {eligibility.max_source_share:.1%}")
            print(f"  可训练: {'是' if eligibility.is_eligible else '否'}")
            
            if not eligibility.is_eligible:
                print(f"  问题: {eligibility.issues}")
                return [], np.array([]), np.array([]), np.array([])
        
        # 提取特征
        print(f"\n提取特征...")
        features = self.feature_store.batch_create(contracts)
        
        # 构建特征矩阵
        feature_matrix = np.array([f.feature_vector for f in features])
        
        # 提取目标值
        purity_targets = np.array([c.purity for c in contracts])
        yield_targets = np.array([c.yield_val for c in contracts])
        
        return contracts, feature_matrix, purity_targets, yield_targets
    
    def train_head(
        self,
        contracts: List[DataContract],
        features: np.ndarray,
        targets: np.ndarray,
        target_name: str,
        eval_protocol: str = "loso",
        model_class=RandomForestRegressor,
        model_params: Optional[Dict] = None
    ):
        """训练单个head"""
        head_id = contracts[0].head_id if contracts else "unknown"
        
        print(f"\n{'='*60}")
        print(f"训练Head: {head_id}")
        print(f"目标: {target_name}")
        print(f"评估协议: {eval_protocol}")
        print(f"{'='*60}")
        
        # 过滤有效目标
        valid_mask = ~np.isnan(targets)
        valid_contracts = [c for c, v in zip(contracts, valid_mask) if v]
        valid_features = features[valid_mask]
        valid_targets = targets[valid_mask]
        
        if len(valid_targets) < 20:
            print(f"⚠ 有效样本不足: {len(valid_targets)} < 20")
            return None
        
        # 执行评估
        if eval_protocol == "loso":
            result = self.evaluator.evaluate_head_loso(
                valid_contracts, valid_features, valid_targets,
                target_name, model_class, model_params
            )
        else:
            result = self.evaluator.evaluate_head_groupkfold(
                valid_contracts, valid_features, valid_targets,
                target_name, model_class, n_splits=5, model_params=model_params
            )
        
        # 生成报告
        report = EvaluationReportGenerator.generate_head_report(result)
        print("\n" + report)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"head_{head_id}_{target_name}_{timestamp}.txt"
        EvaluationReportGenerator.save_report(result, str(report_path))
        
        # 保存JSON结果
        json_path = self.reports_dir / f"head_{head_id}_{target_name}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ JSON结果已保存: {json_path}")
        
        return result
    
    def run_full_evaluation(
        self,
        contracts: List[DataContract],
        eval_protocol: str = "loso"
    ):
        """运行完整评估（所有head）"""
        # 获取所有head
        head_ids = set(c.head_id for c in contracts)
        
        # 检查所有head的训练资格
        print("\n" + "="*60)
        print("Head训练资格检查")
        print("="*60)
        
        eligibilities = HeadEligibilityChecker.check_all_heads(contracts)
        
        trainable_heads = []
        for head_id, eligibility in sorted(eligibilities.items()):
            status = "✓ 可训练" if eligibility.is_eligible else "✗ 不可训练"
            print(f"  {head_id}: {status} (rows={eligibility.rows}, sources={eligibility.sources})")
            if eligibility.is_eligible:
                trainable_heads.append(head_id)
        
        print(f"\n可训练Head数: {len(trainable_heads)}/{len(head_ids)}")
        
        # 训练每个可训练的head
        results = {}
        for head_id in trainable_heads:
            head_contracts = [c for c in contracts if c.head_id == head_id]
            
            # 准备数据
            _, features, purity_targets, yield_targets = self.prepare_data(
                head_contracts, head_id
            )
            
            if len(features) == 0:
                continue
            
            # 训练纯度模型
            purity_result = self.train_head(
                head_contracts, features, purity_targets,
                "purity", eval_protocol
            )
            
            # 训练收率模型
            yield_result = self.train_head(
                head_contracts, features, yield_targets,
                "yield", eval_protocol
            )
            
            results[head_id] = {
                "purity": purity_result.to_dict() if purity_result else None,
                "yield": yield_result.to_dict() if yield_result else None
            }
        
        return results


# ============================================================================
# 步骤级代理训练器
# ============================================================================

class StepProxyTrainer:
    """
    步骤级代理训练器
    
    面向peptimizer的步骤级代理信号预测
    目标变量: first_area, first_height, first_width, first_diff
    
    注意: 这条线的价值是"过程风险代理建模"，不是终点指标预测
    """
    
    def __init__(
        self,
        output_dir: str = "../models/step_proxy",
        reports_dir: str = "../reports"
    ):
        self.output_dir = Path(output_dir)
        self.reports_dir = Path(reports_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def train(
        self,
        data_path: str,
        target: str = "first_area",
        loss: str = "mse"
    ):
        """训练步骤级代理模型"""
        print(f"\n{'='*60}")
        print(f"步骤级代理训练")
        print(f"{'='*60}")
        print(f"目标变量: {target}")
        print(f"损失函数: {loss}")
        print(f"注意: 这是过程风险代理建模，不是终点指标预测")
        print(f"{'='*60}")
        
        # 加载数据
        df = pd.read_csv(data_path)
        print(f"\n加载数据: {len(df)} 条记录")
        
        # 按serial分组分割
        train_df, val_df, test_df = DataSplitter.grouped_by_serial_split(df)
        print(f"训练集: {len(train_df)}")
        print(f"验证集: {len(val_df)}")
        print(f"测试集: {len(test_df)}")
        
        # TODO: 实现实际的模型训练
        # 这里只是框架，实际需要根据具体模型架构实现
        
        return {
            "target": target,
            "loss": loss,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "note": "步骤级代理模型，仅用于过程风险感知"
        }


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="多肽合成预测系统 - 可信度修复版")
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # result_head 子命令
    result_parser = subparsers.add_parser("result_head", help="结果级Head训练")
    result_parser.add_argument("--input", type=str, default="../data/real/final_purity_yield_literature.csv",
                              help="输入数据路径")
    result_parser.add_argument("--head", type=str, default=None,
                              help="指定head_id (如: final_product_isolated_isolated_mass)")
    result_parser.add_argument("--eval", type=str, default="loso", choices=["loso", "groupkfold"],
                              help="评估协议")
    
    # step_proxy 子命令
    step_parser = subparsers.add_parser("step_proxy", help="步骤级代理训练")
    step_parser.add_argument("--input", type=str, default="../data/real/synthesis_data.csv",
                            help="输入数据路径")
    step_parser.add_argument("--target", type=str, default="first_area",
                            help="目标变量")
    step_parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"],
                            help="损失函数")
    
    # diagnose 子命令
    diag_parser = subparsers.add_parser("diagnose", help="诊断数据")
    diag_parser.add_argument("--input", type=str, default="../data/real/final_purity_yield_literature.csv",
                            help="输入数据路径")
    
    args = parser.parse_args()
    
    if args.command == "result_head":
        # 加载数据
        df = DataLoader.load_literature_data(args.input)
        
        # 创建数据合同
        contracts = DataLoader.create_contracts_from_literature(df)
        
        # 来源诊断
        print("\n" + "="*60)
        print("来源诊断")
        print("="*60)
        diagnostics, invalid_sources = SourceValidator.validate_all_sources(contracts)
        print(f"有效来源: {len(diagnostics) - len(invalid_sources)}")
        print(f"无效来源: {len(invalid_sources)}")
        
        if invalid_sources:
            print(f"\n无效来源列表:")
            for source in invalid_sources[:5]:
                diag = diagnostics[source]
                print(f"  - {source}: {diag.issues}")
        
        # 训练
        trainer = ResultHeadTrainer()
        
        if args.head:
            # 训练指定head
            head_contracts = [c for c in contracts if c.head_id == args.head]
            if not head_contracts:
                print(f"⚠ 未找到head: {args.head}")
                return
            
            _, features, purity_targets, yield_targets = trainer.prepare_data(
                head_contracts, args.head
            )
            
            if len(features) > 0:
                trainer.train_head(
                    head_contracts, features, purity_targets,
                    "purity", args.eval
                )
                trainer.train_head(
                    head_contracts, features, yield_targets,
                    "yield", args.eval
                )
        else:
            # 运行完整评估
            trainer.run_full_evaluation(contracts, args.eval)
    
    elif args.command == "step_proxy":
        trainer = StepProxyTrainer()
        trainer.train(args.input, args.target, args.loss)
    
    elif args.command == "diagnose":
        # 加载数据
        df = DataLoader.load_literature_data(args.input)
        
        # 创建数据合同
        contracts = DataLoader.create_contracts_from_literature(df)
        
        # 来源诊断
        print("\n" + "="*60)
        print("来源诊断报告")
        print("="*60)
        diagnostics, invalid_sources = SourceValidator.validate_all_sources(contracts)
        
        for source_id, diag in sorted(diagnostics.items(), 
                                      key=lambda x: x[1].sample_count, reverse=True):
            status = "✓" if diag.is_valid else "✗"
            print(f"\n{status} {source_id}:")
            print(f"    样本数: {diag.sample_count}")
            print(f"    Head分布: {diag.head_distribution}")
            if diag.issues:
                print(f"    问题: {diag.issues}")
        
        # Head资格检查
        print("\n" + "="*60)
        print("Head训练资格检查")
        print("="*60)
        eligibilities = HeadEligibilityChecker.check_all_heads(contracts)
        
        for head_id, eligibility in sorted(eligibilities.items(),
                                          key=lambda x: x[1].rows, reverse=True):
            status = "✓ 可训练" if eligibility.is_eligible else "✗ 不可训练"
            print(f"\n{head_id}:")
            print(f"    状态: {status}")
            print(f"    样本数: {eligibility.rows}")
            print(f"    来源数: {eligibility.sources}")
            print(f"    最大来源占比: {eligibility.max_source_share:.1%}")
            if eligibility.issues:
                print(f"    问题: {eligibility.issues}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
