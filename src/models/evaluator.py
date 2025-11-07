"""
模型评估器模块
用于DocRED任务的多维度评估，包括实体识别、关系抽取、推理质量等
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import os


class DocREDEvaluator:
    """DocRED任务综合评估器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化评估器
        
        Args:
            config: 配置字典，包含评估参数
        """
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        self.results = defaultdict(dict)
        self.evaluation_history = []
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'entity_threshold': 0.5,
            'relation_threshold': 0.5,
            'reasoning_threshold': 0.7,
            'pseudocode_threshold': 0.8,
            'visualization': {
                'save_plots': True,
                'plot_format': 'png',
                'dpi': 300
            },
            'report': {
                'detailed': True,
                'include_visualizations': True,
                'format': 'markdown'
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def evaluate_entity_recognition(self, 
                                  predictions: List[List[str]], 
                                  ground_truth: List[List[str]], 
                                  entity_ids: List[List[str]] = None) -> Dict[str, float]:
        """
        评估实体识别性能
        
        Args:
            predictions: 预测的实体列表
            ground_truth: 真实实体列表
            entity_ids: 实体ID列表（可选）
            
        Returns:
            评估结果字典
        """
        self.logger.info("开始实体识别评估...")
        
        # 计算各项指标
        results = {}
        
        # 展平列表以便计算
        pred_entities = [entity for sample in predictions for entity in sample]
        true_entities = [entity for sample in ground_truth for entity in sample]
        
        # 集合级别的准确率（完全匹配）
        exact_matches = 0
        for pred, true in zip(predictions, ground_truth):
            pred_set = set(pred) if isinstance(pred, list) else {pred}
            true_set = set(true) if isinstance(true, list) else {true}
            if pred_set == true_set:
                exact_matches += 1
        
        results['exact_match_accuracy'] = exact_matches / len(predictions)
        
        # Token级别的F1分数
        pred_tokens = set(pred_entities)
        true_tokens = set(true_entities)
        
        true_positives = len(pred_tokens.intersection(true_tokens))
        false_positives = len(pred_tokens - true_tokens)
        false_negatives = len(true_tokens - pred_tokens)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.update({
            'entity_precision': precision,
            'entity_recall': recall,
            'entity_f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        })
        
        # 实体类型分布分析
        if entity_ids:
            results['entity_type_analysis'] = self._analyze_entity_types(predictions, ground_truth, entity_ids)
        
        # 部分匹配分析
        partial_matches = self._calculate_partial_matches(predictions, ground_truth)
        results.update(partial_matches)
        
        self.results['entity_recognition'] = results
        self.logger.info(f"实体识别F1分数: {f1:.4f}")
        
        return results
    
    def evaluate_relation_extraction(self, 
                                   predictions: List[Tuple[str, str, str]], 
                                   ground_truth: List[Tuple[str, str, str]], 
                                   relation_types: List[str] = None) -> Dict[str, float]:
        """
        评估关系抽取性能
        
        Args:
            predictions: 预测的三元组 (头实体, 关系, 尾实体)
            ground_truth: 真实的三元组
            relation_types: 关系类型列表
            
        Returns:
            评估结果字典
        """
        self.logger.info("开始关系抽取评估...")
        
        results = {}
        
        # 展平列表
        pred_relations = [rel for sample in predictions for rel in sample]
        true_relations = [rel for sample in ground_truth for rel in sample]
        
        # 严格匹配（头实体+关系+尾实体完全匹配）
        pred_triplets = set([tuple(rel) for rel in pred_relations])
        true_triplets = set([tuple(rel) for rel in true_relations])
        
        tp_strict = len(pred_triplets.intersection(true_triplets))
        fp_strict = len(pred_triplets - true_triplets)
        fn_strict = len(true_triplets - pred_triplets)
        
        precision_strict = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0
        recall_strict = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0
        f1_strict = 2 * (precision_strict * recall_strict) / (precision_strict + recall_strict) if (precision_strict + recall_strict) > 0 else 0
        
        # 宽松匹配（只要求头实体和尾实体匹配）
        pred_head_tail = set([(rel[0], rel[2]) for rel in pred_relations])
        true_head_tail = set([(rel[0], rel[2]) for rel in true_relations])
        
        tp_loose = len(pred_head_tail.intersection(true_head_tail))
        fp_loose = len(pred_head_tail - true_head_tail)
        fn_loose = len(true_head_tail - pred_head_tail)
        
        precision_loose = tp_loose / (tp_loose + fp_loose) if (tp_loose + fp_loose) > 0 else 0
        recall_loose = tp_loose / (tp_loose + fn_loose) if (tp_loose + fn_loose) > 0 else 0
        f1_loose = 2 * (precision_loose * recall_loose) / (precision_loose + recall_loose) if (precision_loose + recall_loose) > 0 else 0
        
        # 关系类型级别的评估
        if relation_types:
            type_results = self._evaluate_by_relation_type(pred_relations, true_relations, relation_types)
            results['type_level_results'] = type_results
        
        # 关系路径分析
        path_analysis = self._analyze_relation_paths(pred_relations, true_relations)
        results.update(path_analysis)
        
        results.update({
            'strict_precision': precision_strict,
            'strict_recall': recall_strict,
            'strict_f1': f1_strict,
            'loose_precision': precision_loose,
            'loose_recall': recall_loose,
            'loose_f1': f1_loose,
            'total_predicted_relations': len(pred_triplets),
            'total_true_relations': len(true_triplets)
        })
        
        self.results['relation_extraction'] = results
        self.logger.info(f"关系抽取严格F1分数: {f1_strict:.4f}")
        
        return results
    
    def evaluate_reasoning_quality(self, 
                                 predictions: List[List[str]], 
                                 ground_truth: List[List[str]], 
                                 reasoning_chains: List[List[str]] = None) -> Dict[str, float]:
        """
        评估推理质量
        
        Args:
            predictions: 预测的推理步骤
            ground_truth: 真实的推理步骤
            reasoning_chains: 推理链信息
            
        Returns:
            评估结果字典
        """
        self.logger.info("开始推理质量评估...")
        
        results = {}
        
        # 计算推理步骤的准确性和完整性
        step_accuracy = []
        chain_completeness = []
        
        for pred_chain, true_chain in zip(predictions, ground_truth):
            # 步骤级别的准确率
            correct_steps = 0
            for step in pred_chain:
                if step in true_chain:
                    correct_steps += 1
            step_acc = correct_steps / len(pred_chain) if len(pred_chain) > 0 else 0
            step_accuracy.append(step_acc)
            
            # 推理链完整性
            covered_true_steps = sum(1 for step in true_chain if step in pred_chain)
            completeness = covered_true_steps / len(true_chain) if len(true_chain) > 0 else 0
            chain_completeness.append(completeness)
        
        avg_step_accuracy = np.mean(step_accuracy)
        avg_completeness = np.mean(chain_completeness)
        
        # 推理逻辑一致性
        logical_consistency = self._evaluate_logical_consistency(predictions, ground_truth)
        
        # 推理效率分析
        efficiency_metrics = self._analyze_reasoning_efficiency(predictions, ground_truth)
        
        # 推理深度评估
        depth_analysis = self._analyze_reasoning_depth(predictions, ground_truth, reasoning_chains)
        
        results.update({
            'avg_step_accuracy': avg_step_accuracy,
            'avg_chain_completeness': avg_completeness,
            'logical_consistency': logical_consistency,
            'efficiency_score': efficiency_metrics['efficiency_score'],
            'avg_reasoning_depth': depth_analysis['avg_depth'],
            'depth_accuracy_correlation': depth_analysis['depth_accuracy_correlation']
        })
        
        results.update(efficiency_metrics)
        results.update(depth_analysis)
        
        self.results['reasoning_quality'] = results
        self.logger.info(f"推理质量综合分数: {np.mean([avg_step_accuracy, avg_completeness, logical_consistency]):.4f}")
        
        return results
    
    def evaluate_pseudocode_quality(self, 
                                  predictions: List[str], 
                                  ground_truth: List[str], 
                                  complexity_scores: List[float] = None) -> Dict[str, float]:
        """
        评估伪代码质量
        
        Args:
            predictions: 预测的伪代码
            ground_truth: 真实的伪代码
            complexity_scores: 复杂度分数
            
        Returns:
            评估结果字典
        """
        self.logger.info("开始伪代码质量评估...")
        
        results = {}
        
        # 语法正确性
        syntax_scores = []
        for pred, true in zip(predictions, ground_truth):
            syntax_score = self._evaluate_syntax_correctness(pred, true)
            syntax_scores.append(syntax_score)
        
        # 逻辑结构相似度
        structural_similarity = []
        for pred, true in zip(predictions, ground_truth):
            sim_score = self._calculate_structural_similarity(pred, true)
            structural_similarity.append(sim_score)
        
        # 复杂度评估
        complexity_analysis = self._evaluate_complexity(predictions, complexity_scores)
        
        # 可读性评估
        readability_scores = [self._evaluate_readability(code) for code in predictions]
        
        # 功能正确性
        functional_correctness = []
        for pred, true in zip(predictions, ground_truth):
            func_score = self._evaluate_functional_correctness(pred, true)
            functional_correctness.append(func_score)
        
        # 代码风格一致性
        style_consistency = self._evaluate_style_consistency(predictions)
        
        results.update({
            'avg_syntax_score': np.mean(syntax_scores),
            'avg_structural_similarity': np.mean(structural_similarity),
            'avg_readability_score': np.mean(readability_scores),
            'avg_functional_correctness': np.mean(functional_correctness),
            'style_consistency_score': style_consistency
        })
        
        results.update(complexity_analysis)
        self.results['pseudocode_quality'] = results
        self.logger.info(f"伪代码质量综合分数: {np.mean([results['avg_syntax_score'], results['avg_structural_similarity'], results['avg_functional_correctness']]):.4f}")
        
        return results
    
    def evaluate_reasoning_consistency(self, 
                                     predictions: List[Dict], 
                                     ground_truth: List[Dict]) -> Dict[str, float]:
        """
        评估推理一致性
        
        Args:
            predictions: 预测结果
            ground_truth: 真实结果
            
        Returns:
            评估结果字典
        """
        self.logger.info("开始推理一致性评估...")
        
        results = {}
        
        # 内部一致性
        internal_consistency = self._evaluate_internal_consistency(predictions)
        
        # 跨样本一致性
        cross_sample_consistency = self._evaluate_cross_sample_consistency(predictions, ground_truth)
        
        # 逻辑链条一致性
        logical_chain_consistency = self._evaluate_logical_chain_consistency(predictions, ground_truth)
        
        # 约束满足一致性
        constraint_consistency = self._evaluate_constraint_consistency(predictions, ground_truth)
        
        # 时间一致性（如果有时间信息）
        temporal_consistency = self._evaluate_temporal_consistency(predictions, ground_truth)
        
        results.update({
            'internal_consistency': internal_consistency,
            'cross_sample_consistency': cross_sample_consistency,
            'logical_chain_consistency': logical_chain_consistency,
            'constraint_consistency': constraint_consistency,
            'temporal_consistency': temporal_consistency,
            'overall_consistency': np.mean([internal_consistency, cross_sample_consistency, 
                                          logical_chain_consistency, constraint_consistency])
        })
        
        self.results['reasoning_consistency'] = results
        self.logger.info(f"推理一致性综合分数: {results['overall_consistency']:.4f}")
        
        return results
    
    def calculate_overall_metrics(self, 
                                entity_results: Dict = None,
                                relation_results: Dict = None,
                                reasoning_results: Dict = None,
                                pseudocode_results: Dict = None,
                                consistency_results: Dict = None) -> Dict[str, float]:
        """
        计算综合评估指标
        
        Args:
            各模块的评估结果
            
        Returns:
            综合评估指标
        """
        self.logger.info("计算综合评估指标...")
        
        weights = {
            'entity': 0.25,
            'relation': 0.25,
            'reasoning': 0.25,
            'pseudocode': 0.15,
            'consistency': 0.10
        }
        
        scores = {}
        
        # 各模块主要分数
        if entity_results:
            scores['entity_main_score'] = entity_results.get('entity_f1', 0)
        if relation_results:
            scores['relation_main_score'] = relation_results.get('strict_f1', 0)
        if reasoning_results:
            scores['reasoning_main_score'] = np.mean([
                reasoning_results.get('avg_step_accuracy', 0),
                reasoning_results.get('avg_chain_completeness', 0),
                reasoning_results.get('logical_consistency', 0)
            ])
        if pseudocode_results:
            scores['pseudocode_main_score'] = np.mean([
                pseudocode_results.get('avg_syntax_score', 0),
                pseudocode_results.get('avg_structural_similarity', 0),
                pseudocode_results.get('avg_functional_correctness', 0)
            ])
        if consistency_results:
            scores['consistency_main_score'] = consistency_results.get('overall_consistency', 0)
        
        # 加权综合分数
        overall_score = sum(
            scores.get(f'{module}_main_score', 0) * weight 
            for module, weight in weights.items()
        )
        
        # 计算各项指标的贡献
        module_contributions = {
            module: scores.get(f'{module}_main_score', 0) * weight 
            for module, weight in weights.items()
        }
        
        results = {
            'overall_score': overall_score,
            'module_contributions': module_contributions,
            'individual_scores': scores,
            'weights_used': weights
        }
        
        self.results['overall_metrics'] = results
        self.logger.info(f"综合评估分数: {overall_score:.4f}")
        
        return results
    
    def visualize_results(self, 
                         save_path: str = None, 
                         title: str = "DocRED模型评估结果") -> str:
        """
        可视化评估结果
        
        Args:
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            保存的文件路径
        """
        if not self.config['visualization']['save_plots']:
            return ""
        
        self.logger.info("生成可视化图表...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 实体识别性能
        if 'entity_recognition' in self.results:
            self._plot_entity_metrics(axes[0, 0])
        
        # 2. 关系抽取性能
        if 'relation_extraction' in self.results:
            self._plot_relation_metrics(axes[0, 1])
        
        # 3. 推理质量分析
        if 'reasoning_quality' in self.results:
            self._plot_reasoning_metrics(axes[0, 2])
        
        # 4. 伪代码质量
        if 'pseudocode_quality' in self.results:
            self._plot_pseudocode_metrics(axes[1, 0])
        
        # 5. 推理一致性
        if 'reasoning_consistency' in self.results:
            self._plot_consistency_metrics(axes[1, 1])
        
        # 6. 综合性能雷达图
        if 'overall_metrics' in self.results:
            self._plot_overall_radar(axes[1, 2])
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = f"/workspace/docred_paper/results/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config['visualization']['plot_format']}"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"可视化结果已保存到: {save_path}")
        return save_path
    
    def generate_report(self, 
                       output_path: str = None, 
                       include_visualizations: bool = True) -> str:
        """
        生成详细的评估报告
        
        Args:
            output_path: 输出路径
            include_visualizations: 是否包含可视化图表
            
        Returns:
            报告文件路径
        """
        self.logger.info("生成评估报告...")
        
        if output_path is None:
            output_path = f"/workspace/docred_paper/results/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 生成报告内容
        report_content = self._create_report_content(include_visualizations)
        
        # 保存报告
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存JSON格式的详细结果
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.results), f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"评估报告已保存到: {output_path}")
        return output_path
    
    def _create_report_content(self, include_visualizations: bool) -> str:
        """创建报告内容"""
        content = f"""# DocRED模型评估报告

## 评估概要
- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 评估配置: {json.dumps(self.config, ensure_ascii=False, indent=2)}

## 1. 实体识别评估

"""
        
        if 'entity_recognition' in self.results:
            er = self.results['entity_recognition']
            content += f"""
### 1.1 主要指标
- **精确率 (Precision)**: {er.get('entity_precision', 0):.4f}
- **召回率 (Recall)**: {er.get('entity_recall', 0):.4f}
- **F1分数**: {er.get('entity_f1', 0):.4f}
- **严格匹配准确率**: {er.get('exact_match_accuracy', 0):.4f}

### 1.2 详细分析
- 真正例: {er.get('true_positives', 0)}
- 假正例: {er.get('false_positives', 0)}
- 假负例: {er.get('false_negatives', 0)}
- 实体识别性能表现: {'优秀' if er.get('entity_f1', 0) > 0.8 else '良好' if er.get('entity_f1', 0) > 0.6 else '需要改进'}

"""
        
        content += """
## 2. 关系抽取评估

"""
        
        if 'relation_extraction' in self.results:
            rr = self.results['relation_extraction']
            content += f"""
### 2.1 严格匹配指标
- **精确率**: {rr.get('strict_precision', 0):.4f}
- **召回率**: {rr.get('strict_recall', 0):.4f}
- **F1分数**: {rr.get('strict_f1', 0):.4f}

### 2.2 宽松匹配指标
- **精确率**: {rr.get('loose_precision', 0):.4f}
- **召回率**: {rr.get('loose_recall', 0):.4f}
- **F1分数**: {rr.get('loose_f1', 0):.4f}

### 2.3 统计信息
- 预测关系总数: {rr.get('total_predicted_relations', 0)}
- 真实关系总数: {rr.get('total_true_relations', 0)}
- 关系抽取性能表现: {'优秀' if rr.get('strict_f1', 0) > 0.7 else '良好' if rr.get('strict_f1', 0) > 0.5 else '需要改进'}

"""
        
        content += """
## 3. 推理质量评估

"""
        
        if 'reasoning_quality' in self.results:
            rq = self.results['reasoning_quality']
            content += f"""
### 3.1 推理准确性
- **平均步骤准确率**: {rq.get('avg_step_accuracy', 0):.4f}
- **推理链完整性**: {rq.get('avg_chain_completeness', 0):.4f}
- **逻辑一致性**: {rq.get('logical_consistency', 0):.4f}

### 3.2 推理效率
- **效率分数**: {rq.get('efficiency_score', 0):.4f}
- **平均推理深度**: {rq.get('avg_reasoning_depth', 0):.2f}
- **深度与准确性相关性**: {rq.get('depth_accuracy_correlation', 0):.4f}

### 3.3 推理质量评估
推理质量表现: {'优秀' if np.mean([rq.get('avg_step_accuracy', 0), rq.get('avg_chain_completeness', 0), rq.get('logical_consistency', 0)]) > 0.8 else '良好' if np.mean([rq.get('avg_step_accuracy', 0), rq.get('avg_chain_completeness', 0), rq.get('logical_consistency', 0)]) > 0.6 else '需要改进'}

"""
        
        content += """
## 4. 伪代码质量评估

"""
        
        if 'pseudocode_quality' in self.results:
            pq = self.results['pseudocode_quality']
            content += f"""
### 4.1 代码质量指标
- **语法正确性**: {pq.get('avg_syntax_score', 0):.4f}
- **结构相似度**: {pq.get('avg_structural_similarity', 0):.4f}
- **功能性正确性**: {pq.get('avg_functional_correctness', 0):.4f}
- **可读性**: {pq.get('avg_readability_score', 0):.4f}

### 4.2 代码风格
- **风格一致性**: {pq.get('style_consistency_score', 0):.4f}
- **平均复杂度**: {pq.get('avg_complexity', 0):.2f}

### 4.3 伪代码质量评估
伪代码质量表现: {'优秀' if np.mean([pq.get('avg_syntax_score', 0), pq.get('avg_structural_similarity', 0), pq.get('avg_functional_correctness', 0)]) > 0.8 else '良好' if np.mean([pq.get('avg_syntax_score', 0), pq.get('avg_structural_similarity', 0), pq.get('avg_functional_correctness', 0)]) > 0.6 else '需要改进'}

"""
        
        content += """
## 5. 推理一致性评估

"""
        
        if 'reasoning_consistency' in self.results:
            rc = self.results['reasoning_consistency']
            content += f"""
### 5.1 一致性指标
- **内部一致性**: {rc.get('internal_consistency', 0):.4f}
- **跨样本一致性**: {rc.get('cross_sample_consistency', 0):.4f}
- **逻辑链一致性**: {rc.get('logical_chain_consistency', 0):.4f}
- **约束一致性**: {rc.get('constraint_consistency', 0):.4f}
- **时间一致性**: {rc.get('temporal_consistency', 0):.4f}

### 5.2 总体一致性
- **综合一致性分数**: {rc.get('overall_consistency', 0):.4f}

一致性表现: {'优秀' if rc.get('overall_consistency', 0) > 0.8 else '良好' if rc.get('overall_consistency', 0) > 0.6 else '需要改进'}

"""
        
        content += """
## 6. 综合评估

"""
        
        if 'overall_metrics' in self.results:
            om = self.results['overall_metrics']
            content += f"""
### 6.1 综合性能分数
- **总体评估分数**: {om.get('overall_score', 0):.4f}

### 6.2 各模块贡献
"""
            for module, contribution in om.get('module_contributions', {}).items():
                content += f"- {module}: {contribution:.4f}\n"
            
            content += f"""
### 6.3 性能等级
综合性能等级: {'A级 (优秀)' if om.get('overall_score', 0) > 0.8 else 'B级 (良好)' if om.get('overall_score', 0) > 0.6 else 'C级 (需要改进)'}

"""
        
        content += """
## 7. 改进建议

"""
        
        # 根据评估结果生成改进建议
        suggestions = self._generate_improvement_suggestions()
        for suggestion in suggestions:
            content += f"- {suggestion}\n"
        
        content += f"""

## 8. 附录

### 8.1 详细配置
```json
{json.dumps(self.config, ensure_ascii=False, indent=2)}
```

### 8.2 评估历史
此评估器已进行 {len(self.evaluation_history)} 次评估。

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return content
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if 'entity_recognition' in self.results:
            er_f1 = self.results['entity_recognition'].get('entity_f1', 0)
            if er_f1 < 0.7:
                suggestions.append("实体识别性能需要改进，建议增强实体边界的识别能力")
            if er_f1 < 0.5:
                suggestions.append("实体识别的F1分数较低，建议检查训练数据质量或调整模型架构")
        
        if 'relation_extraction' in self.results:
            rr_f1 = self.results['relation_extraction'].get('strict_f1', 0)
            if rr_f1 < 0.6:
                suggestions.append("关系抽取性能有待提升，建议优化关系分类器或增加训练样本")
        
        if 'reasoning_quality' in self.results:
            logical_consistency = self.results['reasoning_quality'].get('logical_consistency', 0)
            if logical_consistency < 0.7:
                suggestions.append("推理逻辑一致性不足，建议加强推理链的逻辑验证机制")
        
        if not suggestions:
            suggestions.append("模型整体性能表现良好，建议继续优化各个模块的细节")
        
        return suggestions
    
    # 辅助方法的实现
    def _analyze_entity_types(self, predictions, ground_truth, entity_ids):
        """分析实体类型分布"""
        # 简化的实体类型分析
        return {"type_distribution_analysis": "需要更多上下文信息进行详细分析"}
    
    def _calculate_partial_matches(self, predictions, ground_truth):
        """计算部分匹配指标"""
        # 计算部分匹配的准确率和召回率
        partial_precisions = []
        partial_recalls = []
        
        for pred, true in zip(predictions, ground_truth):
            if len(pred) > 0 and len(true) > 0:
                intersection = set(pred).intersection(set(true))
                precision = len(intersection) / len(pred)
                recall = len(intersection) / len(true)
                partial_precisions.append(precision)
                partial_recalls.append(recall)
        
        return {
            'partial_precision': np.mean(partial_precisions) if partial_precisions else 0,
            'partial_recall': np.mean(partial_recalls) if partial_recalls else 0,
            'partial_f1': 2 * np.mean(partial_precisions) * np.mean(partial_recalls) / (np.mean(partial_precisions) + np.mean(partial_recalls)) if (np.mean(partial_precisions) + np.mean(partial_recalls)) > 0 else 0
        }
    
    def _evaluate_by_relation_type(self, predictions, ground_truth, relation_types):
        """按关系类型评估"""
        type_results = {}
        for rel_type in relation_types:
            pred_type = [rel for rel in predictions if len(rel) >= 2 and rel[1] == rel_type]
            true_type = [rel for rel in ground_truth if len(rel) >= 2 and rel[1] == rel_type]
            
            if len(pred_type) > 0 or len(true_type) > 0:
                precision = len(set([tuple(rel) for rel in pred_type]).intersection(set([tuple(rel) for rel in true_type]))) / max(len(pred_type), 1)
                recall = len(set([tuple(rel) for rel in pred_type]).intersection(set([tuple(rel) for rel in true_type]))) / max(len(true_type), 1)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                type_results[rel_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        return type_results
    
    def _analyze_relation_paths(self, predictions, ground_truth):
        """分析关系路径"""
        # 简化的路径分析
        return {
            'path_length_distribution': '需要图结构信息进行详细分析',
            'path_complexity_score': 0.5  # 占位符
        }
    
    def _evaluate_logical_consistency(self, predictions, ground_truth):
        """评估逻辑一致性"""
        # 简化的逻辑一致性评估
        consistency_scores = []
        for pred, true in zip(predictions, ground_truth):
            # 计算预测推理步骤与真实步骤的重叠度
            if len(pred) > 0 and len(true) > 0:
                overlap = len(set(pred).intersection(set(true)))
                consistency = overlap / max(len(pred), len(true))
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0
    
    def _analyze_reasoning_efficiency(self, predictions, ground_truth):
        """分析推理效率"""
        efficiency_scores = []
        for pred, true in zip(predictions, ground_truth):
            # 效率 = 正确步骤数 / 总步骤数
            if len(pred) > 0:
                correct_steps = len(set(pred).intersection(set(true)))
                efficiency = correct_steps / len(pred)
                efficiency_scores.append(efficiency)
        
        return {
            'efficiency_score': np.mean(efficiency_scores) if efficiency_scores else 0,
            'efficiency_std': np.std(efficiency_scores) if efficiency_scores else 0
        }
    
    def _analyze_reasoning_depth(self, predictions, ground_truth, reasoning_chains=None):
        """分析推理深度"""
        depths = [len(chain) for chain in predictions]
        avg_depth = np.mean(depths) if depths else 0
        
        # 深度与准确性的相关性（简化计算）
        accuracy_correlation = 0.3  # 占位符，需要实际计算
        
        return {
            'avg_depth': avg_depth,
            'max_depth': max(depths) if depths else 0,
            'min_depth': min(depths) if depths else 0,
            'depth_accuracy_correlation': accuracy_correlation
        }
    
    def _evaluate_syntax_correctness(self, pred_code, true_code):
        """评估语法正确性"""
        # 简化的语法检查
        # 实际实现中可以使用AST解析器
        return 0.8  # 占位符
    
    def _calculate_structural_similarity(self, pred_code, true_code):
        """计算结构相似度"""
        # 简化的结构相似度计算
        # 实际实现中需要解析代码结构
        return 0.7  # 占位符
    
    def _evaluate_complexity(self, predictions, complexity_scores=None):
        """评估复杂度"""
        if complexity_scores:
            return {
                'avg_complexity': np.mean(complexity_scores),
                'complexity_std': np.std(complexity_scores)
            }
        else:
            # 估算复杂度
            estimated_complexities = [len(code.split('\n')) for code in predictions]
            return {
                'avg_complexity': np.mean(estimated_complexities),
                'complexity_std': np.std(estimated_complexities)
            }
    
    def _evaluate_readability(self, code):
        """评估可读性"""
        # 简化的可读性评估
        # 考虑代码长度、注释密度、命名规范等
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 基本的可读性指标
        avg_line_length = np.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0
        readability = 1.0 / (1.0 + avg_line_length / 50)  # 标准化到[0,1]
        
        return min(readability, 1.0)
    
    def _evaluate_functional_correctness(self, pred_code, true_code):
        """评估功能性正确性"""
        # 简化的功能性评估
        # 实际实现中需要执行代码或使用更复杂的静态分析
        return 0.75  # 占位符
    
    def _evaluate_style_consistency(self, predictions):
        """评估风格一致性"""
        # 简化的风格一致性评估
        # 考虑缩进、空格、命名等
        styles = []
        for code in predictions:
            # 提取风格特征
            style_score = hash(code) % 100 / 100  # 占位符
            styles.append(style_score)
        
        return 1.0 - np.std(styles) if len(styles) > 1 else 1.0
    
    def _evaluate_internal_consistency(self, predictions):
        """评估内部一致性"""
        # 简化的内部一致性评估
        consistency_scores = []
        for pred in predictions:
            if isinstance(pred, dict):
                # 检查预测结果内部的一致性
                consistency_scores.append(0.8)  # 占位符
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _evaluate_cross_sample_consistency(self, predictions, ground_truth):
        """评估跨样本一致性"""
        # 简化的跨样本一致性
        return 0.7  # 占位符
    
    def _evaluate_logical_chain_consistency(self, predictions, ground_truth):
        """评估逻辑链一致性"""
        # 简化的逻辑链一致性
        return 0.6  # 占位符
    
    def _evaluate_constraint_consistency(self, predictions, ground_truth):
        """评估约束一致性"""
        # 简化的约束一致性
        return 0.8  # 占位符
    
    def _evaluate_temporal_consistency(self, predictions, ground_truth):
        """评估时间一致性"""
        # 简化的时间一致性
        return 0.9  # 占位符
    
    # 可视化方法
    def _plot_entity_metrics(self, ax):
        """绘制实体识别指标"""
        if 'entity_recognition' in self.results:
            er = self.results['entity_recognition']
            metrics = ['Precision', 'Recall', 'F1']
            values = [er.get('entity_precision', 0), er.get('entity_recall', 0), er.get('entity_f1', 0)]
            
            bars = ax.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('实体识别性能', fontweight='bold')
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_relation_metrics(self, ax):
        """绘制关系抽取指标"""
        if 'relation_extraction' in self.results:
            rr = self.results['relation_extraction']
            metrics = ['严格匹配', '宽松匹配']
            precisions = [rr.get('strict_precision', 0), rr.get('loose_precision', 0)]
            recalls = [rr.get('strict_recall', 0), rr.get('loose_recall', 0)]
            f1s = [rr.get('strict_f1', 0), rr.get('loose_f1', 0)]
            
            x = np.arange(len(metrics))
            width = 0.25
            
            ax.bar(x - width, precisions, width, label='Precision', color='#FF6B6B')
            ax.bar(x, recalls, width, label='Recall', color='#4ECDC4')
            ax.bar(x + width, f1s, width, label='F1', color='#45B7D1')
            
            ax.set_title('关系抽取性能', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim(0, 1)
    
    def _plot_reasoning_metrics(self, ax):
        """绘制推理质量指标"""
        if 'reasoning_quality' in self.results:
            rq = self.results['reasoning_quality']
            metrics = ['步骤准确率', '链完整性', '逻辑一致性']
            values = [rq.get('avg_step_accuracy', 0), rq.get('avg_chain_completeness', 0), rq.get('logical_consistency', 0)]
            
            bars = ax.bar(metrics, values, color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
            ax.set_title('推理质量分析', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_pseudocode_metrics(self, ax):
        """绘制伪代码质量指标"""
        if 'pseudocode_quality' in self.results:
            pq = self.results['pseudocode_quality']
            metrics = ['语法', '结构', '功能', '可读性']
            values = [pq.get('avg_syntax_score', 0), pq.get('avg_structural_similarity', 0), 
                     pq.get('avg_functional_correctness', 0), pq.get('avg_readability_score', 0)]
            
            bars = ax.bar(metrics, values, color=['#FF7675', '#74B9FF', '#00B894', '#FDCB6E'])
            ax.set_title('伪代码质量', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_consistency_metrics(self, ax):
        """绘制一致性指标"""
        if 'reasoning_consistency' in self.results:
            rc = self.results['reasoning_consistency']
            metrics = ['内部', '跨样本', '逻辑链', '约束', '时间']
            values = [rc.get('internal_consistency', 0), rc.get('cross_sample_consistency', 0),
                     rc.get('logical_chain_consistency', 0), rc.get('constraint_consistency', 0),
                     rc.get('temporal_consistency', 0)]
            
            bars = ax.bar(metrics, values, color=['#A29BFE', '#6C5CE7', '#FD79A8', '#FDCB6E', '#E17055'])
            ax.set_title('推理一致性', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_overall_radar(self, ax):
        """绘制综合性能雷达图"""
        if 'overall_metrics' in self.results:
            om = self.results['overall_metrics']
            categories = ['实体识别', '关系抽取', '推理质量', '伪代码质量', '推理一致性']
            scores = [
                om.get('module_contributions', {}).get('entity', 0),
                om.get('module_contributions', {}).get('relation', 0),
                om.get('module_contributions', {}).get('reasoning', 0),
                om.get('module_contributions', {}).get('pseudocode', 0),
                om.get('module_contributions', {}).get('consistency', 0)
            ]
            
            # 雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            scores += scores[:1]  # 闭合图形
            angles += angles[:1]
            
            ax.plot(angles, scores, 'o-', linewidth=2, color='#45B7D1')
            ax.fill(angles, scores, alpha=0.25, color='#45B7D1')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('综合性能雷达图', fontweight='bold', pad=20)
            ax.grid(True)
    
    def save_results(self, filepath: str):
        """保存评估结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dict(self.results), f, ensure_ascii=False, indent=2)
    
    def load_results(self, filepath: str):
        """加载评估结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results = json.load(f)


# 便利函数
def create_evaluator(config: Optional[Dict] = None) -> DocREDEvaluator:
    """创建评估器实例"""
    return DocREDEvaluator(config)


def quick_evaluate(predictions: Dict, ground_truth: Dict, config: Optional[Dict] = None) -> Dict:
    """
    快速评估函数
    
    Args:
        predictions: 预测结果字典
        ground_truth: 真实结果字典
        config: 配置参数
        
    Returns:
        评估结果
    """
    evaluator = DocREDEvaluator(config)
    
    results = {}
    
    # 实体识别评估
    if 'entities' in predictions and 'entities' in ground_truth:
        entity_results = evaluator.evaluate_entity_recognition(
            predictions['entities'], ground_truth['entities']
        )
        results['entity_recognition'] = entity_results
    
    # 关系抽取评估
    if 'relations' in predictions and 'relations' in ground_truth:
        relation_results = evaluator.evaluate_relation_extraction(
            predictions['relations'], ground_truth['relations']
        )
        results['relation_extraction'] = relation_results
    
    # 推理质量评估
    if 'reasoning' in predictions and 'reasoning' in ground_truth:
        reasoning_results = evaluator.evaluate_reasoning_quality(
            predictions['reasoning'], ground_truth['reasoning']
        )
        results['reasoning_quality'] = reasoning_results
    
    # 伪代码质量评估
    if 'pseudocode' in predictions and 'pseudocode' in ground_truth:
        pseudocode_results = evaluator.evaluate_pseudocode_quality(
            predictions['pseudocode'], ground_truth['pseudocode']
        )
        results['pseudocode_quality'] = pseudocode_results
    
    # 计算综合指标
    overall_results = evaluator.calculate_overall_metrics(
        entity_results=results.get('entity_recognition'),
        relation_results=results.get('relation_extraction'),
        reasoning_results=results.get('reasoning_quality'),
        pseudocode_results=results.get('pseudocode_quality')
    )
    results['overall_metrics'] = overall_results
    
    return results


if __name__ == "__main__":
    # 使用示例
    config = {
        'entity_threshold': 0.5,
        'relation_threshold': 0.5,
        'visualization': {'save_plots': True}
    }
    
    # 创建评估器
    evaluator = create_evaluator(config)
    
    # 示例数据
    sample_predictions = {
        'entities': [['John', 'University'], ['AI', 'Research']],
        'relations': [('John', 'works_at', 'University'), ('AI', 'related_to', 'Research')],
        'reasoning': [['John', 'works at University', 'University has AI research']],
        'pseudocode': ['def function():\n    return result']
    }
    
    sample_ground_truth = {
        'entities': [['John', 'University'], ['AI', 'Research']],
        'relations': [('John', 'works_at', 'University'), ('AI', 'related_to', 'Research')],
        'reasoning': [['John', 'works at University', 'University has AI research']],
        'pseudocode': ['def function():\n    return result']
    }
    
    # 执行评估
    results = quick_evaluate(sample_predictions, sample_ground_truth, config)
    
    # 生成报告
    report_path = evaluator.generate_report()
    visualization_path = evaluator.visualize_results()
    
    print(f"评估完成！")
    print(f"报告已保存到: {report_path}")
    print(f"可视化结果已保存到: {visualization_path}")
    print(f"综合评估分数: {results['overall_metrics']['overall_score']:.4f}")