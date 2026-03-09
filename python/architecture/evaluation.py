"""
评估协议模块
============
实现严格的评估协议，禁止再用"看起来更好"的单次数值替代主结论

评估协议:
1. synthetic MLP: 固定 train/val/test (演示基线)
2. step_proxy: 固定 grouped-by-serial
3. result_head: 固定 GroupKFold 或 leave-one-source-out (LOSO)

版本: 2.0.0
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入核心架构
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.core import (
    DataContract, TaskLine, TaskLineConfig, SemanticHead,
    HeadEligibility, HeadEligibilityChecker
)


# ============================================================================
# 评估结果定义
# ============================================================================

@dataclass
class FoldResult:
    """单折评估结果"""
    fold_id: int
    train_sources: List[str]
    test_sources: List[str]
    train_size: int
    test_size: int
    rmse: float
    mae: float
    r2: float
    tolerance_5: float  # ±5%准确率
    tolerance_10: float  # ±10%准确率
    
    def to_dict(self) -> Dict:
        return {
            'fold_id': self.fold_id,
            'train_sources': self.train_sources,
            'test_sources': self.test_sources,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'tolerance_5': self.tolerance_5,
            'tolerance_10': self.tolerance_10,
        }


@dataclass
class HeadEvaluationResult:
    """Head评估结果"""
    head_id: str
    target: str  # 'purity' or 'yield'
    eval_protocol: str
    
    # 折级结果
    fold_results: List[FoldResult]
    
    # 汇总统计
    mean_rmse: float
    std_rmse: float
    mean_mae: float
    std_mae: float
    mean_r2: float
    std_r2: float
    mean_tolerance_5: float
    mean_tolerance_10: float
    
    # 来源级结果
    source_results: Dict[str, Dict]
    
    # 元数据
    total_samples: int
    total_sources: int
    is_deployable: bool
    
    def to_dict(self) -> Dict:
        return {
            'head_id': self.head_id,
            'target': self.target,
            'eval_protocol': self.eval_protocol,
            'fold_results': [f.to_dict() for f in self.fold_results],
            'mean_rmse': self.mean_rmse,
            'std_rmse': self.std_rmse,
            'mean_mae': self.mean_mae,
            'std_mae': self.std_mae,
            'mean_r2': self.mean_r2,
            'std_r2': self.std_r2,
            'mean_tolerance_5': self.mean_tolerance_5,
            'mean_tolerance_10': self.mean_tolerance_10,
            'source_results': self.source_results,
            'total_samples': self.total_samples,
            'total_sources': self.total_sources,
            'is_deployable': self.is_deployable,
        }


# ============================================================================
# 数据分割器
# ============================================================================

class DataSplitter:
    """
    数据分割器
    
    严格实现source-aware分割，禁止随机行切分
    """
    
    @staticmethod
    def validate_no_leakage(train_sources: set, test_sources: set):
        """验证无来源泄漏"""
        overlap = train_sources & test_sources
        if overlap:
            raise ValueError(f"来源泄漏检测: 训练集和测试集共享来源 {overlap}")
    
    @classmethod
    def groupkfold_split(
        cls,
        contracts: List[DataContract],
        n_splits: int = 5,
        random_state: int = 42
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """
        GroupKFold分割
        
        按source_id分组，确保同一来源的样本不会同时出现在训练集和测试集
        """
        df = pd.DataFrame([{
            'idx': i,
            'source_id': c.source_id
        } for i, c in enumerate(contracts)])
        
        gkf = GroupKFold(n_splits=n_splits)
        
        for fold_id, (train_idx, test_idx) in enumerate(
            gkf.split(df, groups=df['source_id'])
        ):
            # 验证无泄漏
            train_sources = set(df.iloc[train_idx]['source_id'])
            test_sources = set(df.iloc[test_idx]['source_id'])
            cls.validate_no_leakage(train_sources, test_sources)
            
            yield fold_id, train_idx.tolist(), test_idx.tolist(), train_sources, test_sources
    
    @classmethod
    def loso_split(
        cls,
        contracts: List[DataContract]
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Leave-One-Source-Out分割
        
        每次留出一个来源作为测试集
        """
        source_to_indices = defaultdict(list)
        for i, c in enumerate(contracts):
            source_to_indices[c.source_id].append(i)
        
        sources = list(source_to_indices.keys())
        
        for fold_id, test_source in enumerate(sources):
            test_indices = source_to_indices[test_source]
            train_indices = [i for i in range(len(contracts)) if i not in test_indices]
            
            train_sources = set(sources) - {test_source}
            test_sources = {test_source}
            
            yield fold_id, train_indices, test_indices, train_sources, test_sources
    
    @classmethod
    def grouped_by_serial_split(
        cls,
        df: pd.DataFrame,
        serial_col: str = 'serial',
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按serial分组的分割 (用于step_proxy任务)
        
        确保同一serial的所有步骤在同一个集合中
        """
        serials = df[serial_col].unique()
        np.random.seed(42)
        np.random.shuffle(serials)
        
        n_test = int(len(serials) * test_ratio)
        n_val = n_test
        
        test_serials = set(serials[:n_test])
        val_serials = set(serials[n_test:n_test+n_val])
        train_serials = set(serials[n_test+n_val:])
        
        train_df = df[df[serial_col].isin(train_serials)]
        val_df = df[df[serial_col].isin(val_serials)]
        test_df = df[df[serial_col].isin(test_serials)]
        
        return train_df, val_df, test_df


# ============================================================================
# 评估器
# ============================================================================

class HeadEvaluator:
    """
    Head评估器
    
    所有结果级模型统一以head为主单位训练和展示
    """
    
    def __init__(self):
        self.results: Dict[str, HeadEvaluationResult] = {}
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 容差准确率
        tolerance_5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100
        tolerance_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'tolerance_5': tolerance_5,
            'tolerance_10': tolerance_10,
        }
    
    def evaluate_head_loso(
        self,
        contracts: List[DataContract],
        features: np.ndarray,
        targets: np.ndarray,
        target_name: str,
        model_class,
        model_params: Dict = None
    ) -> HeadEvaluationResult:
        """
        LOSO评估
        
        Args:
            contracts: 数据合同列表
            features: 特征矩阵
            targets: 目标值数组
            target_name: 目标名称 ('purity' or 'yield')
            model_class: 模型类
            model_params: 模型参数
        """
        model_params = model_params or {}
        fold_results = []
        source_results = {}
        
        for fold_id, train_idx, test_idx, train_sources, test_sources in \
            DataSplitter.loso_split(contracts):
            
            # 分割数据
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = targets[train_idx], targets[test_idx]
            
            # 训练模型
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            metrics = self.compute_metrics(y_test, y_pred)
            
            # 记录折结果
            fold_results.append(FoldResult(
                fold_id=fold_id,
                train_sources=list(train_sources),
                test_sources=list(test_sources),
                train_size=len(train_idx),
                test_size=len(test_idx),
                **metrics
            ))
            
            # 记录来源结果
            for source in test_sources:
                if source not in source_results:
                    source_results[source] = {
                        'samples': 0,
                        'rmse_sum': 0,
                        'mae_sum': 0,
                    }
                source_results[source]['samples'] += len(test_idx)
                source_results[source]['rmse_sum'] += metrics['rmse'] * len(test_idx)
                source_results[source]['mae_sum'] += metrics['mae'] * len(test_idx)
        
        # 计算来源级平均
        for source in source_results:
            n = source_results[source]['samples']
            source_results[source]['avg_rmse'] = source_results[source]['rmse_sum'] / n
            source_results[source]['avg_mae'] = source_results[source]['mae_sum'] / n
        
        # 计算汇总统计
        rmses = [f.rmse for f in fold_results]
        maes = [f.mae for f in fold_results]
        r2s = [f.r2 for f in fold_results]
        tol5s = [f.tolerance_5 for f in fold_results]
        tol10s = [f.tolerance_10 for f in fold_results]
        
        head_id = contracts[0].head_id if contracts else "unknown"
        
        return HeadEvaluationResult(
            head_id=head_id,
            target=target_name,
            eval_protocol='loso',
            fold_results=fold_results,
            mean_rmse=np.mean(rmses),
            std_rmse=np.std(rmses),
            mean_mae=np.mean(maes),
            std_mae=np.std(maes),
            mean_r2=np.mean(r2s),
            std_r2=np.std(r2s),
            mean_tolerance_5=np.mean(tol5s),
            mean_tolerance_10=np.mean(tol10s),
            source_results=source_results,
            total_samples=len(contracts),
            total_sources=len(set(c.source_id for c in contracts)),
            is_deployable=len(fold_results) >= 3
        )
    
    def evaluate_head_groupkfold(
        self,
        contracts: List[DataContract],
        features: np.ndarray,
        targets: np.ndarray,
        target_name: str,
        model_class,
        n_splits: int = 5,
        model_params: Dict = None
    ) -> HeadEvaluationResult:
        """
        GroupKFold评估
        """
        model_params = model_params or {}
        fold_results = []
        source_results = defaultdict(lambda: {'samples': 0, 'rmse_sum': 0, 'mae_sum': 0})
        
        for fold_id, train_idx, test_idx, train_sources, test_sources in \
            DataSplitter.groupkfold_split(contracts, n_splits=n_splits):
            
            # 分割数据
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = targets[train_idx], targets[test_idx]
            
            # 训练模型
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            metrics = self.compute_metrics(y_test, y_pred)
            
            # 记录折结果
            fold_results.append(FoldResult(
                fold_id=fold_id,
                train_sources=list(train_sources),
                test_sources=list(test_sources),
                train_size=len(train_idx),
                test_size=len(test_idx),
                **metrics
            ))
            
            # 记录来源结果
            for i in test_idx:
                source = contracts[i].source_id
                source_results[source]['samples'] += 1
                source_results[source]['rmse_sum'] += (y_test[i] - y_pred[i]) ** 2
                source_results[source]['mae_sum'] += abs(y_test[i] - y_pred[i])
        
        # 计算来源级平均
        for source in source_results:
            n = source_results[source]['samples']
            source_results[source]['avg_rmse'] = np.sqrt(source_results[source]['rmse_sum'] / n)
            source_results[source]['avg_mae'] = source_results[source]['mae_sum'] / n
        
        # 计算汇总统计
        rmses = [f.rmse for f in fold_results]
        maes = [f.mae for f in fold_results]
        r2s = [f.r2 for f in fold_results]
        tol5s = [f.tolerance_5 for f in fold_results]
        tol10s = [f.tolerance_10 for f in fold_results]
        
        head_id = contracts[0].head_id if contracts else "unknown"
        
        return HeadEvaluationResult(
            head_id=head_id,
            target=target_name,
            eval_protocol='groupkfold',
            fold_results=fold_results,
            mean_rmse=np.mean(rmses),
            std_rmse=np.std(rmses),
            mean_mae=np.mean(maes),
            std_mae=np.std(maes),
            mean_r2=np.mean(r2s),
            std_r2=np.std(r2s),
            mean_tolerance_5=np.mean(tol5s),
            mean_tolerance_10=np.mean(tol10s),
            source_results=dict(source_results),
            total_samples=len(contracts),
            total_sources=len(set(c.source_id for c in contracts)),
            is_deployable=len(fold_results) >= 3
        )


# ============================================================================
# 评估报告生成器
# ============================================================================

class EvaluationReportGenerator:
    """评估报告生成器"""
    
    @staticmethod
    def generate_head_report(result: HeadEvaluationResult) -> str:
        """生成Head评估报告"""
        lines = []
        
        lines.append("="*70)
        lines.append(f"Head评估报告: {result.head_id}")
        lines.append("="*70)
        lines.append(f"目标变量: {result.target}")
        lines.append(f"评估协议: {result.eval_protocol}")
        lines.append(f"总样本数: {result.total_samples}")
        lines.append(f"总来源数: {result.total_sources}")
        lines.append(f"可部署: {'是' if result.is_deployable else '否'}")
        lines.append("")
        
        lines.append("-"*70)
        lines.append("汇总统计")
        lines.append("-"*70)
        lines.append(f"RMSE: {result.mean_rmse:.3f} ± {result.std_rmse:.3f}")
        lines.append(f"MAE: {result.mean_mae:.3f} ± {result.std_mae:.3f}")
        lines.append(f"R²: {result.mean_r2:.3f} ± {result.std_r2:.3f}")
        lines.append(f"准确率 (±5%): {result.mean_tolerance_5:.1f}%")
        lines.append(f"准确率 (±10%): {result.mean_tolerance_10:.1f}%")
        lines.append("")
        
        lines.append("-"*70)
        lines.append("折级结果")
        lines.append("-"*70)
        for fold in result.fold_results:
            lines.append(f"Fold {fold.fold_id}: RMSE={fold.rmse:.3f}, MAE={fold.mae:.3f}, "
                        f"R²={fold.r2:.3f}, train={fold.train_size}, test={fold.test_size}")
        lines.append("")
        
        lines.append("-"*70)
        lines.append("来源级结果 (Top 10)")
        lines.append("-"*70)
        sorted_sources = sorted(result.source_results.items(), 
                               key=lambda x: x[1]['avg_rmse'], reverse=True)[:10]
        for source, metrics in sorted_sources:
            lines.append(f"  {source}: RMSE={metrics['avg_rmse']:.3f}, "
                        f"MAE={metrics['avg_mae']:.3f}, n={metrics['samples']}")
        
        return "\n".join(lines)
    
    @staticmethod
    def save_report(result: HeadEvaluationResult, output_path: str):
        """保存评估报告"""
        report = EvaluationReportGenerator.generate_head_report(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 评估报告已保存: {output_path}")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'DataSplitter',
    'HeadEvaluator',
    'FoldResult',
    'HeadEvaluationResult',
    'EvaluationReportGenerator',
]
