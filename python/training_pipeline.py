"""
多肽合成预测系统 - 完整数据处理与模型训练流程
包含：数据获取、预处理、标注、训练、评估全流程

作者: Peptide Synthesis Predictor Team
日期: 2024
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 导入已有的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhance_dataset import (
    AMINO_ACID_PROPERTIES,
    COUPLING_DIFFICULTY,
    THREE_TO_ONE,
    parse_sequence,
    calculate_peptide_features
)
from data_preprocessor import DataPreprocessor, RawDataInput


# ============================================================================
# 第一阶段：数据获取
# ============================================================================

@dataclass
class DataSource:
    """数据源信息"""
    name: str
    url: str
    description: str
    data_type: str  # 'literature', 'database', 'experimental'
    quality_score: float  # 0-1
    citation: str


class DataAcquisition:
    """数据获取模块"""
    
    def __init__(self, output_dir: str = '../data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 注册可用数据源
        self.data_sources = self._register_data_sources()
        
    def _register_data_sources(self) -> Dict[str, DataSource]:
        """注册高质量数据源"""
        return {
            # 公开数据库
            'chembl': DataSource(
                name='ChEMBL',
                url='https://www.ebi.ac.uk/chembl/',
                description='生物活性分子数据库，包含肽类药物活性数据',
                data_type='database',
                quality_score=0.95,
                citation='Gaulton et al. (2017) Nucleic Acids Res. 45:D945-D954'
            ),
            'uniprot': DataSource(
                name='UniProt',
                url='https://www.uniprot.org/',
                description='蛋白质序列和功能数据库',
                data_type='database',
                quality_score=0.98,
                citation='UniProt Consortium (2023) Nucleic Acids Res. 51:D523-D531'
            ),
            'pdb': DataSource(
                name='PDB',
                url='https://www.rcsb.org/',
                description='蛋白质三维结构数据库',
                data_type='database',
                quality_score=0.95,
                citation='Berman et al. (2000) Nucleic Acids Res. 28:235-242'
            ),
            'bindingdb': DataSource(
                name='BindingDB',
                url='https://www.bindingdb.org/',
                description='药物靶标结合亲和力数据库',
                data_type='database',
                quality_score=0.90,
                citation='Gilson et al. (2016) Nucleic Acids Res. 44:D1045-D1053'
            ),
            # 文献数据源
            'literature_internal': DataSource(
                name='Internal Literature Data',
                url='local',
                description='项目内部收集的文献数据',
                data_type='literature',
                quality_score=0.85,
                citation='Multiple sources (see documentation)'
            ),
            # 实验数据
            'experimental': DataSource(
                name='Experimental Data',
                url='local',
                description='实验室合成仪实时数据',
                data_type='experimental',
                quality_score=0.92,
                citation='In-house experimental data'
            )
        }
    
    def get_data_source_info(self) -> pd.DataFrame:
        """获取所有数据源信息"""
        sources = []
        for key, source in self.data_sources.items():
            sources.append({
                'source_id': key,
                'name': source.name,
                'url': source.url,
                'type': source.data_type,
                'quality_score': source.quality_score,
                'citation': source.citation
            })
        return pd.DataFrame(sources)
    
    def load_existing_data(self) -> Tuple[pd.DataFrame, Dict]:
        """加载现有数据"""
        print("="*60)
        print("第一阶段：数据获取")
        print("="*60)
        
        loaded_data = []
        data_sources_used = []
        
        # 1. 加载文献数据
        literature_path = '../data/real/final_purity_yield_literature.csv'
        if os.path.exists(literature_path):
            df_lit = pd.read_csv(literature_path)
            # 标准化列名
            column_mapping = {
                'purity_pct': 'purity',
                'yield_pct': 'yield_val'
            }
            df_lit = df_lit.rename(columns=column_mapping)
            df_lit['data_source'] = 'literature'
            loaded_data.append(df_lit)
            data_sources_used.append('literature_internal')
            print(f"✓ 加载文献数据: {len(df_lit)} 条记录")
            print(f"  - 包含纯度数据: {df_lit['purity'].notna().sum()} 条")
            print(f"  - 包含收率数据: {df_lit['yield_val'].notna().sum()} 条")
        
        # 2. 加载实验数据
        exp_path = '../data/real/synthesis_data.csv'
        if os.path.exists(exp_path):
            df_exp = pd.read_csv(exp_path)
            # 转换实验数据格式
            df_exp_processed = self._process_experimental_data(df_exp)
            loaded_data.append(df_exp_processed)
            data_sources_used.append('experimental')
            print(f"✓ 加载实验数据: {len(df_exp_processed)} 条记录")
        
        # 3. 加载增强数据
        enhanced_path = '../data/enhanced_training_data.csv'
        if os.path.exists(enhanced_path):
            df_enhanced = pd.read_csv(enhanced_path)
            loaded_data.append(df_enhanced)
            print(f"✓ 加载增强数据: {len(df_enhanced)} 条记录")
        
        if not loaded_data:
            print("⚠ 未找到现有数据，将生成合成数据")
            return pd.DataFrame(), {}
        
        # 合并所有数据
        combined_df = pd.concat(loaded_data, ignore_index=True)
        
        # 记录数据来源
        source_info = {
            'total_records': len(combined_df),
            'sources': data_sources_used,
            'timestamp': datetime.now().isoformat(),
            'data_sources_detail': self.get_data_source_info().to_dict('records')
        }
        
        print(f"\n总计加载: {len(combined_df)} 条记录")
        print(f"数据来源: {', '.join(data_sources_used)}")
        
        return combined_df, source_info
    
    def _process_experimental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理实验数据格式"""
        processed = []
        
        for _, row in df.iterrows():
            try:
                pre_chain = str(row.get('pre-chain', ''))
                aa = str(row.get('amino_acid', ''))
                sequence = pre_chain + aa
                
                if len(sequence) > 0:
                    processed.append({
                        'sequence': sequence,
                        'coupling_reagent': row.get('coupling_agent', 'HBTU'),
                        'temperature': f"{row.get('temp_coupling', 25)}°C",
                        'purity': None,  # 实验数据可能没有纯度
                        'yield_val': None,
                        'data_source': 'experimental'
                    })
            except Exception as e:
                continue
        
        return pd.DataFrame(processed)


# ============================================================================
# 第二阶段：数据预处理
# ============================================================================

class DataCleaning:
    """数据清洗模块"""
    
    def __init__(self):
        self.cleaning_log = []
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """执行完整的数据清洗流程"""
        print("\n" + "="*60)
        print("第二阶段：数据预处理")
        print("="*60)
        
        initial_count = len(df)
        cleaning_stats = {
            'initial_records': initial_count,
            'steps': {}
        }
        
        # 1. 去除重复记录
        df = self._remove_duplicates(df)
        cleaning_stats['steps']['duplicates_removed'] = initial_count - len(df)
        
        # 2. 处理异常值
        df = self._handle_outliers(df)
        cleaning_stats['steps']['outliers_handled'] = 'completed'
        
        # 3. 处理缺失值
        df = self._handle_missing_values(df)
        cleaning_stats['steps']['missing_values_handled'] = 'completed'
        
        # 4. 标准化格式
        df = self._standardize_formats(df)
        cleaning_stats['steps']['formats_standardized'] = 'completed'
        
        # 5. 验证数据质量
        df = self._validate_data_quality(df)
        cleaning_stats['final_records'] = len(df)
        cleaning_stats['retention_rate'] = len(df) / initial_count if initial_count > 0 else 0
        
        print(f"\n清洗完成: {initial_count} → {len(df)} 条记录")
        print(f"保留率: {cleaning_stats['retention_rate']:.1%}")
        
        return df, cleaning_stats
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """去除重复记录"""
        before = len(df)
        
        # 基于关键字段去重
        key_columns = ['sequence']
        existing_cols = [col for col in key_columns if col in df.columns]
        
        if existing_cols:
            df = df.drop_duplicates(subset=existing_cols, keep='first')
        
        removed = before - len(df)
        if removed > 0:
            print(f"  ✓ 去除重复记录: {removed} 条")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        # 确保数值列为数值类型
        for col in ['purity', 'yield_val']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 纯度异常值
        if 'purity' in df.columns:
            invalid_purity = (df['purity'] < 0) | (df['purity'] > 100)
            df.loc[invalid_purity, 'purity'] = np.nan
            print(f"  ✓ 处理纯度异常值: {invalid_purity.sum()} 条")
        
        # 收率异常值
        if 'yield_val' in df.columns:
            invalid_yield = (df['yield_val'] < 0) | (df['yield_val'] > 100)
            df.loc[invalid_yield, 'yield_val'] = np.nan
            print(f"  ✓ 处理收率异常值: {invalid_yield.sum()} 条")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 序列必须存在
        if 'sequence' in df.columns:
            df = df.dropna(subset=['sequence'])
        
        # 数值型缺失值用中位数填充
        numeric_cols = ['purity', 'yield_val']
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                if not pd.isna(median_val):
                    missing_count = df[col].isna().sum()
                    df[col] = df[col].fillna(median_val)
                    if missing_count > 0:
                        print(f"  ✓ 填充 {col} 缺失值: {missing_count} 条 (中位数: {median_val:.2f})")
        
        # 类别型缺失值用众数填充
        categorical_cols = ['coupling_reagent', 'solvent', 'temperature']
        for col in categorical_cols:
            if col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    missing_count = df[col].isna().sum()
                    df[col] = df[col].fillna(mode_val[0])
                    if missing_count > 0:
                        print(f"  ✓ 填充 {col} 缺失值: {missing_count} 条 (众数: {mode_val[0]})")
        
        return df
    
    def _standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化格式"""
        # 标准化序列格式
        if 'sequence' in df.columns:
            df['sequence'] = df['sequence'].str.strip()
            df['sequence'] = df['sequence'].str.upper()
        
        # 标准化温度格式
        if 'temperature' in df.columns:
            df['temperature'] = df['temperature'].str.strip()
        
        print("  ✓ 标准化数据格式")
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证数据质量"""
        valid_mask = pd.Series([True] * len(df))
        
        # 验证序列有效性
        if 'sequence' in df.columns:
            for idx, row in df.iterrows():
                residues = parse_sequence(row['sequence'])
                if len(residues) == 0:
                    valid_mask[idx] = False
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"  ✓ 移除无效序列: {invalid_count} 条")
            df = df[valid_mask].reset_index(drop=True)
        
        return df


# ============================================================================
# 第三阶段：数据标注
# ============================================================================

class DataAnnotation:
    """数据标注模块"""
    
    def __init__(self):
        self.annotation_standards = self._create_annotation_standards()
        self.annotation_log = []
    
    def _create_annotation_standards(self) -> Dict:
        """创建标注标准"""
        return {
            'sequence': {
                'required': True,
                'format': '单字母或三字母代码',
                'example': 'GAVL 或 Gly-Ala-Val-Leu'
            },
            'purity': {
                'required': False,
                'range': [0, 100],
                'unit': '%',
                'measurement': 'HPLC纯度'
            },
            'yield_val': {
                'required': False,
                'range': [0, 100],
                'unit': '%',
                'measurement': '分离收率'
            },
            'coupling_reagent': {
                'required': False,
                'allowed_values': ['HATU', 'HBTU', 'PyBOP', 'DIC/Oxyma', 'Other'],
                'default': 'HBTU'
            },
            'solvent': {
                'required': False,
                'allowed_values': ['DMF', 'NMP', 'DMF/DCM', 'Other'],
                'default': 'DMF'
            },
            'temperature': {
                'required': False,
                'allowed_values': ['Room Temperature', 'Microwave 75°C', 'Microwave 90°C'],
                'default': 'Room Temperature'
            }
        }
    
    def annotate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """执行数据标注"""
        print("\n" + "="*60)
        print("第三阶段：数据标注")
        print("="*60)
        
        annotation_stats = {
            'total_records': len(df),
            'annotations_added': {}
        }
        
        # 使用预处理器提取特征
        preprocessor = DataPreprocessor()
        
        annotated_records = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                # 创建输入数据
                input_data = RawDataInput(
                    sequence=str(row.get('sequence', '')),
                    purity=row.get('purity'),
                    yield_val=row.get('yield_val') or row.get('yield'),
                    coupling_reagent=row.get('coupling_reagent'),
                    solvent=row.get('solvent'),
                    temperature=row.get('temperature'),
                    cleavage_time=row.get('cleavage_time'),
                    topology=row.get('topology'),
                    source=row.get('data_source', 'unknown')
                )
                
                # 处理记录
                processed = preprocessor.process_single_record(input_data)
                
                if processed:
                    record = {
                        'sequence': processed.sequence,
                        'length': processed.length,
                        'avg_hydrophobicity': processed.avg_hydrophobicity,
                        'total_charge': processed.total_charge,
                        'molecular_weight': processed.molecular_weight,
                        'avg_volume': processed.avg_volume,
                        'max_coupling_difficulty': processed.max_coupling_difficulty,
                        'bulky_ratio': processed.bulky_ratio,
                        'polar_ratio': processed.polar_ratio,
                        'charged_ratio': processed.charged_ratio,
                        'aromatic_ratio': processed.aromatic_ratio,
                        'sulfur_ratio': processed.sulfur_ratio,
                        'reagent_score': processed.reagent_score,
                        'solvent_score': processed.solvent_score,
                        'temperature_score': processed.temperature_score,
                        'cleavage_score': processed.cleavage_score,
                        'purity': processed.purity,
                        'yield_val': processed.yield_val,
                        'data_source': processed.source,
                        'annotation_quality': self._assess_quality(processed),
                        'annotated_at': datetime.now().isoformat()
                    }
                    annotated_records.append(record)
                else:
                    errors.append({'index': idx, 'error': 'Processing failed'})
                    
            except Exception as e:
                errors.append({'index': idx, 'error': str(e)})
        
        annotated_df = pd.DataFrame(annotated_records)
        
        # 双重校验
        annotated_df = self._double_check(annotated_df)
        
        annotation_stats['successful_annotations'] = len(annotated_df)
        annotation_stats['failed_annotations'] = len(errors)
        annotation_stats['accuracy_rate'] = len(annotated_df) / len(df) if len(df) > 0 else 0
        
        print(f"\n标注完成: {len(annotated_df)} 条记录")
        print(f"标注准确率: {annotation_stats['accuracy_rate']:.1%}")
        
        if errors:
            print(f"失败记录: {len(errors)} 条")
        
        return annotated_df, annotation_stats
    
    def _assess_quality(self, processed) -> float:
        """评估标注质量"""
        quality_score = 1.0
        
        # 检查必需字段
        if processed.purity is None:
            quality_score *= 0.8
        if processed.yield_val is None:
            quality_score *= 0.9
        if processed.length == 0:
            quality_score *= 0.5
        
        return quality_score
    
    def _double_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """双重校验机制"""
        print("\n执行双重校验...")
        
        # 校验1: 序列有效性
        valid_sequences = df['sequence'].apply(lambda x: len(parse_sequence(x)) > 0)
        df = df[valid_sequences]
        print(f"  ✓ 序列有效性校验通过: {valid_sequences.sum()} 条")
        
        # 校验2: 数值范围合理性
        if 'purity' in df.columns:
            valid_purity = (df['purity'].isna()) | ((df['purity'] >= 0) & (df['purity'] <= 100))
            df = df[valid_purity]
        
        if 'yield_val' in df.columns:
            valid_yield = (df['yield_val'].isna()) | ((df['yield_val'] >= 0) & (df['yield_val'] <= 100))
            df = df[valid_yield]
        
        print(f"  ✓ 数值范围校验通过")
        
        # 校验3: 特征完整性
        feature_cols = ['length', 'avg_hydrophobicity', 'molecular_weight']
        for col in feature_cols:
            if col in df.columns:
                df = df[~df[col].isna()]
        
        print(f"  ✓ 特征完整性校验通过")
        
        return df.reset_index(drop=True)
    
    def generate_annotation_report(self, df: pd.DataFrame) -> str:
        """生成标注报告"""
        report = []
        report.append("="*60)
        report.append("数据标注报告")
        report.append("="*60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n总记录数: {len(df)}")
        
        if 'data_source' in df.columns:
            report.append("\n数据来源分布:")
            for source, count in df['data_source'].value_counts().items():
                report.append(f"  - {source}: {count} 条 ({count/len(df):.1%})")
        
        if 'purity' in df.columns:
            purity_valid = df['purity'].notna().sum()
            report.append(f"\n纯度数据完整性: {purity_valid}/{len(df)} ({purity_valid/len(df):.1%})")
        
        if 'yield_val' in df.columns:
            yield_valid = df['yield_val'].notna().sum()
            report.append(f"收率数据完整性: {yield_valid}/{len(df)} ({yield_valid/len(df):.1%})")
        
        report.append("\n标注标准:")
        for field, standard in self.annotation_standards.items():
            report.append(f"  - {field}: {standard.get('format', standard.get('range', 'N/A'))}")
        
        return "\n".join(report)


# ============================================================================
# 第四阶段：模型训练
# ============================================================================

class ModelTraining:
    """模型训练模块"""
    
    def __init__(self, output_dir: str = '../models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.training_history = {}
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame,
        target_column: str = 'purity',
        test_size: float = 0.2,
        val_size: float = 0.125  # 0.125 * 0.8 = 0.1 of total
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """准备训练数据"""
        print("\n" + "="*60)
        print("第四阶段：模型训练")
        print("="*60)
        
        # 选择特征列
        feature_columns = [
            'length', 'avg_hydrophobicity', 'total_charge', 'molecular_weight',
            'avg_volume', 'max_coupling_difficulty', 'bulky_ratio', 'polar_ratio',
            'charged_ratio', 'aromatic_ratio', 'sulfur_ratio', 'reagent_score',
            'solvent_score', 'temperature_score', 'cleavage_score'
        ]
        
        # 过滤有效数据
        valid_df = df.dropna(subset=[target_column])
        print(f"有效数据: {len(valid_df)} 条 (目标变量: {target_column})")
        
        # 提取特征和目标
        X = valid_df[feature_columns]
        y = valid_df[target_column]
        
        # 分割数据集 (70% train, 20% val, 10% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-0.1), random_state=42
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(X_train)} 条 ({len(X_train)/len(valid_df):.1%})")
        print(f"  验证集: {len(X_val)} 条 ({len(X_val)/len(valid_df):.1%})")
        print(f"  测试集: {len(X_test)} 条 ({len(X_test)/len(valid_df):.1%})")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_column] = scaler
        
        # 保存数据集信息
        self.training_history[target_column] = {
            'feature_columns': feature_columns,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'target_mean': y.mean(),
            'target_std': y.std()
        }
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_models(
        self,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        target_column: str = 'purity'
    ) -> Dict:
        """训练多个模型"""
        print(f"\n训练目标: {target_column}")
        print("-"*40)
        
        results = {}
        
        # 模型1: 随机森林
        print("\n1. 训练随机森林模型...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_results = self._evaluate_model(rf_model, X_val, y_val, 'Random Forest')
        results['RandomForest'] = rf_results
        self.models[f'{target_column}_rf'] = rf_model
        
        # 模型2: 梯度提升
        print("\n2. 训练梯度提升模型...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_results = self._evaluate_model(gb_model, X_val, y_val, 'Gradient Boosting')
        results['GradientBoosting'] = gb_results
        self.models[f'{target_column}_gb'] = gb_model
        
        # 选择最佳模型
        best_model_name = min(results, key=lambda x: results[x]['rmse'])
        # 获取正确的模型键名
        model_key_mapping = {
            'RandomForest': 'rf',
            'GradientBoosting': 'gb'
        }
        model_key = model_key_mapping.get(best_model_name, best_model_name.lower())
        best_model = self.models[f'{target_column}_{model_key}']
        
        print(f"\n最佳模型: {best_model_name}")
        
        # 在测试集上评估
        test_results = self._evaluate_model(best_model, X_test, y_test, 'Best Model (Test)')
        results['BestModel_Test'] = test_results
        
        return results
    
    def _evaluate_model(self, model, X, y, model_name: str) -> Dict:
        """评估模型性能"""
        y_pred = model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # 计算准确率 (±5%范围内)
        accuracy_5 = np.mean(np.abs(y - y_pred) <= 5) * 100
        accuracy_10 = np.mean(np.abs(y - y_pred) <= 10) * 100
        
        print(f"  {model_name}:")
        print(f"    RMSE: {rmse:.3f}")
        print(f"    MAE: {mae:.3f}")
        print(f"    R²: {r2:.3f}")
        print(f"    准确率 (±5%): {accuracy_5:.1f}%")
        print(f"    准确率 (±10%): {accuracy_10:.1f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy_5': accuracy_5,
            'accuracy_10': accuracy_10,
            'predictions': y_pred,
            'actuals': y
        }
    
    def save_models(self, target_column: str):
        """保存训练好的模型"""
        for model_name, model in self.models.items():
            if target_column in model_name:
                model_path = os.path.join(self.output_dir, f'{model_name}.joblib')
                joblib.dump(model, model_path)
                print(f"✓ 保存模型: {model_path}")
        
        # 保存scaler
        if target_column in self.scalers:
            scaler_path = os.path.join(self.output_dir, f'{target_column}_scaler.joblib')
            joblib.dump(self.scalers[target_column], scaler_path)
            print(f"✓ 保存标准化器: {scaler_path}")


# ============================================================================
# 第五阶段：训练报告生成
# ============================================================================

class TrainingReportGenerator:
    """训练报告生成器"""
    
    def __init__(self, output_dir: str = '../reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        data_stats: Dict,
        cleaning_stats: Dict,
        annotation_stats: Dict,
        training_results: Dict,
        target_column: str
    ) -> str:
        """生成完整训练报告"""
        report = []
        
        report.append("="*70)
        report.append("多肽合成预测模型 - 训练报告")
        report.append("="*70)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"目标变量: {target_column}")
        report.append("")
        
        # 1. 数据概览
        report.append("-"*70)
        report.append("1. 数据概览")
        report.append("-"*70)
        report.append(f"原始数据量: {data_stats.get('total_records', 'N/A')}")
        report.append(f"数据来源: {', '.join(data_stats.get('sources', []))}")
        report.append(f"清洗后数据量: {cleaning_stats.get('final_records', 'N/A')}")
        report.append(f"数据保留率: {cleaning_stats.get('retention_rate', 0):.1%}")
        report.append("")
        
        # 2. 数据清洗
        report.append("-"*70)
        report.append("2. 数据清洗")
        report.append("-"*70)
        for step, result in cleaning_stats.get('steps', {}).items():
            report.append(f"  • {step}: {result}")
        report.append("")
        
        # 3. 数据标注
        report.append("-"*70)
        report.append("3. 数据标注")
        report.append("-"*70)
        report.append(f"成功标注: {annotation_stats.get('successful_annotations', 'N/A')}")
        report.append(f"标注准确率: {annotation_stats.get('accuracy_rate', 0):.1%}")
        report.append("")
        
        # 4. 模型性能
        report.append("-"*70)
        report.append("4. 模型性能")
        report.append("-"*70)
        for model_name, metrics in training_results.items():
            report.append(f"\n{model_name}:")
            report.append(f"  • RMSE: {metrics['rmse']:.3f}")
            report.append(f"  • MAE: {metrics['mae']:.3f}")
            report.append(f"  • R²: {metrics['r2']:.3f}")
            report.append(f"  • 准确率 (±5%): {metrics['accuracy_5']:.1f}%")
            report.append(f"  • 准确率 (±10%): {metrics['accuracy_10']:.1f}%")
        report.append("")
        
        # 5. 结论与建议
        report.append("-"*70)
        report.append("5. 结论与建议")
        report.append("-"*70)
        
        best_model = min(
            {k: v for k, v in training_results.items() if 'Test' not in k},
            key=lambda x: training_results[x]['rmse']
        )
        
        report.append(f"推荐模型: {best_model}")
        report.append(f"验证集RMSE: {training_results[best_model]['rmse']:.3f}")
        report.append(f"验证集R²: {training_results[best_model]['r2']:.3f}")
        report.append("")
        report.append("优化建议:")
        report.append("  1. 增加训练数据量以提升模型泛化能力")
        report.append("  2. 尝试深度学习模型（如Transformer）")
        report.append("  3. 进行超参数调优")
        report.append("  4. 增加更多特征工程")
        
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = os.path.join(
            self.output_dir, 
            f'training_report_{target_column}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ 训练报告已保存: {report_path}")
        
        return report_text


# ============================================================================
# 主流程
# ============================================================================

def main():
    """执行完整的数据处理与模型训练流程"""
    print("\n" + "="*70)
    print("多肽合成预测系统 - 完整训练流程")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # 初始化各模块
    acquisition = DataAcquisition()
    cleaning = DataCleaning()
    annotation = DataAnnotation()
    training = ModelTraining()
    reporter = TrainingReportGenerator()
    
    # 存储所有统计信息
    all_stats = {}
    
    try:
        # 第一阶段：数据获取
        df, data_stats = acquisition.load_existing_data()
        all_stats['data'] = data_stats
        
        if df.empty:
            print("\n⚠ 错误: 未找到可用数据")
            return
        
        # 第二阶段：数据清洗
        df_cleaned, cleaning_stats = cleaning.clean_data(df)
        all_stats['cleaning'] = cleaning_stats
        
        # 第三阶段：数据标注
        df_annotated, annotation_stats = annotation.annotate_data(df_cleaned)
        all_stats['annotation'] = annotation_stats
        
        # 生成标注报告
        annotation_report = annotation.generate_annotation_report(df_annotated)
        print("\n" + annotation_report)
        
        # 保存标注数据
        annotated_path = '../data/annotated_training_data.csv'
        df_annotated.to_csv(annotated_path, index=False)
        print(f"\n✓ 标注数据已保存: {annotated_path}")
        
        # 第四阶段：模型训练
        # 训练纯度预测模型
        if df_annotated['purity'].notna().sum() > 10:
            X_train, X_val, X_test, y_train, y_val, y_test = training.prepare_training_data(
                df_annotated, target_column='purity'
            )
            purity_results = training.train_models(
                X_train, X_val, X_test, y_train, y_val, y_test,
                target_column='purity'
            )
            training.save_models('purity')
            all_stats['training_purity'] = purity_results
            
            # 生成训练报告
            reporter.generate_report(
                data_stats, cleaning_stats, annotation_stats,
                purity_results, 'purity'
            )
        
        # 训练收率预测模型
        if df_annotated['yield_val'].notna().sum() > 10:
            X_train, X_val, X_test, y_train, y_val, y_test = training.prepare_training_data(
                df_annotated, target_column='yield_val'
            )
            yield_results = training.train_models(
                X_train, X_val, X_test, y_train, y_val, y_test,
                target_column='yield_val'
            )
            training.save_models('yield_val')
            all_stats['training_yield'] = yield_results
            
            # 生成训练报告
            reporter.generate_report(
                data_stats, cleaning_stats, annotation_stats,
                yield_results, 'yield_val'
            )
        
        # 保存完整统计信息
        stats_path = '../reports/training_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n✓ 统计信息已保存: {stats_path}")
        
        print("\n" + "="*70)
        print("训练流程完成！")
        print("="*70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
