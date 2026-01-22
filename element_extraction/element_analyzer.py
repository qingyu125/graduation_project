"""
要素分析模块 - 面向高效推理的要素深度分析

本模块提供对抽取要素的深度分析功能：
- 要素对比分析
- 效率-质量权衡分析
- 推理过程可视化
- 要素重要性评估

Copyright (c) MiniMax Agent
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AnalysisReport:
    """分析报告"""
    report_id: str
    analysis_type: str
    summary: Dict[str, Any]
    detailed_findings: List[Dict[str, Any]]
    recommendations: List[str]
    visualizations: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'analysis_type': self.analysis_type,
            'summary': self.summary,
            'detailed_findings': self.detailed_findings,
            'recommendations': self.recommendations,
            'visualizations': self.visualizations,
            'timestamp': self.timestamp
        }


class ElementAnalyzer:
    """
    要素分析器
    
    对抽取的要素进行深度分析，
    生成分析报告和可视化。
    """
    
    def __init__(self):
        self.analysis_history = []
    
    def analyze_confidence_patterns(self, 
                                   extraction_results: List[Any]) -> AnalysisReport:
        """
        分析置信度模式
        
        Args:
            extraction_results: 要素抽取结果列表
            
        Returns:
            分析报告
        """
        report_id = f"confidence_pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 收集数据
        all_confs = []
        trace_conf_patterns = defaultdict(list)
        critical_points_by_position = defaultdict(list)
        
        for result in extraction_results:
            for element in result.local_confidence_elements:
                all_confs.append(element.token_confidence)
                
                # 按trace分组
                trace_conf_patterns[element.trace_id].append({
                    'position': element.relative_position,
                    'confidence': element.token_confidence,
                    'is_critical': element.is_critical_point
                })
                
                # 按位置统计关键点
                if element.is_critical_point:
                    critical_points_by_position[int(element.relative_position * 10)].append(element.token_confidence)
        
        # 计算统计量
        conf_array = np.array(all_confs) if all_confs else np.array([0])
        
        summary = {
            'total_tokens_analyzed': len(all_confs),
            'mean_confidence': float(np.mean(conf_array)),
            'std_confidence': float(np.std(conf_array)),
            'median_confidence': float(np.median(conf_array)),
            'critical_point_ratio': float(np.sum(conf_array < np.percentile(conf_array, 10)) / max(len(conf_array), 1)),
            'confidence_range': [float(np.min(conf_array)), float(np.max(conf_array))]
        }
        
        # 详细发现
        detailed_findings = []
        
        # 发现1: 置信度分布特征
        if np.std(conf_array) > 0.2:
            detailed_findings.append({
                'finding_id': 'high_variance',
                'title': '高置信度方差',
                'description': f'置信度标准差为{np.std(conf_array):.3f}，表明推理过程存在明显的置信度波动',
                'severity': 'medium',
                'implication': '可能存在推理路径不稳定的问题'
            })
        
        # 发现2: 关键点分布
        if critical_points_by_position:
            critical_positions = sorted(critical_points_by_position.keys())
            early_critical = sum(1 for p in critical_positions if p < 3)
            late_critical = sum(1 for p in critical_positions if p >= 7)
            
            if early_critical > late_critical:
                detailed_findings.append({
                    'finding_id': 'early_critical',
                    'title': '早期关键点集中',
                    'description': f'前30%位置的关键点占比较高，推理初期存在较多不确定区域',
                    'severity': 'low',
                    'implication': '可能需要增强问题理解阶段的指导'
                })
        
        # 推荐
        recommendations = []
        if summary['mean_confidence'] > 0.5:
            recommendations.append('考虑增加推理深度以提高置信度')
        if np.std(conf_array) > 0.2:
            recommendations.append('建议分析低置信度区域的具体原因')
        recommendations.append('针对关键置信点进行针对性优化')
        
        return AnalysisReport(
            report_id=report_id,
            analysis_type='confidence_pattern',
            summary=summary,
            detailed_findings=detailed_findings,
            recommendations=recommendations,
            visualizations=['confidence_distribution', 'confidence_trend'],
            timestamp=datetime.now().isoformat()
        )
    
    def analyze_compression_effectiveness(self,
                                         extraction_results: List[Any]) -> AnalysisReport:
        """
        分析压缩效果
        
        Args:
            extraction_results: 要素抽取结果列表
            
        Returns:
            分析报告
        """
        report_id = f"compression_effect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 只分析RPC结果
        rpc_results = [r for r in extraction_results if r.rpc_enabled]
        
        if not rpc_results:
            return AnalysisReport(
                report_id=report_id,
                analysis_type='compression_effect',
                summary={'error': '没有RPC结果可分析'},
                detailed_findings=[],
                recommendations=['需要启用RPC才能分析压缩效果'],
                visualizations=[],
                timestamp=datetime.now().isoformat()
            )
        
        # 收集压缩数据
        all_ratios = []
        layer_efficiency = defaultdict(list)
        position_efficiency = defaultdict(list)
        
        for result in rpc_results:
            for element in result.compression_path_elements:
                all_ratios.append(element.compression_ratio)
                layer_efficiency[element.layer_id].append(element.compression_ratio)
                pos_bucket = element.position // 1000
                position_efficiency[pos_bucket].append(element.compression_ratio)
        
        # 计算统计量
        ratio_array = np.array(all_ratios) if all_ratios else np.array([1.0])
        
        summary = {
            'total_compressions': len(all_ratios),
            'mean_compression_ratio': float(np.mean(ratio_array)),
            'std_compression_ratio': float(np.std(ratio_array)),
            'max_compression_ratio': float(np.max(ratio_array)),
            'min_compression_ratio': float(np.min(ratio_array)),
            'layers_analyzed': len(layer_efficiency),
            'overall_efficiency': float(np.mean(ratio_array) * min(1.0, len(all_ratios) / 50))
        }
        
        # 详细发现
        detailed_findings = []
        
        # 发现1: 压缩效果评估
        if np.mean(ratio_array) < 1.1:
            detailed_findings.append({
                'finding_id': 'low_compression',
                'title': '压缩效果不佳',
                'description': f'平均压缩比为{np.mean(ratio_array):.2f}，低于预期',
                'severity': 'high',
                'implication': '需要调整RPC参数或问题类型不适合RPC'
            })
        elif np.mean(ratio_array) > 3.0:
            detailed_findings.append({
                'finding_id': 'high_compression',
                'title': '压缩效果显著',
                'description': f'平均压缩比为{np.mean(ratio_array):.2f}，实现了良好的压缩效果',
                'severity': 'info',
                'implication': '可以进一步优化以获得更好的效率'
            })
        
        # 发现2: 稳定性评估
        if np.std(ratio_array) > 0.5:
            detailed_findings.append({
                'finding_id': 'unstable_compression',
                'title': '压缩比不稳定',
                'description': f'压缩比标准差为{np.std(ratio_array):.2f}，不同位置的压缩效果差异较大',
                'severity': 'medium',
                'implication': '可能需要自适应调整压缩参数'
            })
        
        # 推荐
        recommendations = []
        if summary['mean_compression_ratio'] < 1.5:
            recommendations.append('建议增大P值或减小c值以提高压缩比')
        if np.std(ratio_array) > 0.5:
            recommendations.append('考虑实现自适应压缩策略')
        recommendations.append('针对不同问题类型进行参数调优')
        
        return AnalysisReport(
            report_id=report_id,
            analysis_type='compression_effect',
            summary=summary,
            detailed_findings=detailed_findings,
            recommendations=recommendations,
            visualizations=['compression_by_layer', 'compression_trend'],
            timestamp=datetime.now().isoformat()
        )
    
    def analyze_efficiency_quality_tradeoff(self,
                                            extraction_results: List[Any]) -> AnalysisReport:
        """
        分析效率-质量权衡
        
        Args:
            extraction_results: 要素抽取结果列表
            
        Returns:
            分析报告
        """
        report_id = f"tradeoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 收集评分数据
        efficiencies = []
        qualities = []
        
        for result in extraction_results:
            efficiencies.append(result.overall_efficiency_score)
            qualities.append(result.reasoning_quality_score)
        
        # 计算权衡指标
        eff_array = np.array(efficiencies)
        qual_array = np.array(qualities)
        
        # 计算帕累托前沿
        pareto_front = []
        for i, (e, q) in enumerate(zip(efficiencies, qualities)):
            is_dominated = False
            for j, (e2, q2) in enumerate(zip(efficiencies, qualities)):
                if i != j and e2 >= e and q2 >= q and (e2 > e or q2 > q):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append({
                    'index': i,
                    'efficiency': e,
                    'quality': q
                })
        
        summary = {
            'total_results': len(extraction_results),
            'mean_efficiency': float(np.mean(eff_array)) if len(eff_array) > 0 else 0,
            'mean_quality': float(np.mean(qual_array)) if len(qual_array) > 0 else 0,
            'pareto_optimal_count': len(pareto_front),
            'efficiency_quality_correlation': float(np.corrcoef(eff_array, qual_array)[0, 1]) if len(eff_array) > 1 else 0,
            'tradeoff_score': self._calculate_tradeoff_score(eff_array, qual_array)
        }
        
        # 详细发现
        detailed_findings = []
        
        # 发现1: 相关性分析
        if len(eff_array) > 1:
            corr = np.corrcoef(eff_array, qual_array)[0, 1]
            if abs(corr) > 0.5:
                detailed_findings.append({
                    'finding_id': 'strong_correlation',
                    'title': '效率与质量强相关',
                    'description': f'相关系数为{corr:.3f}，表明两者存在明显的线性关系',
                    'severity': 'info',
                    'implication': '可以通过优化一个指标来同时改善另一个'
                })
        
        # 发现2: 帕累托最优
        if len(pareto_front) < len(extraction_results):
            dominated_count = len(extraction_results) - len(pareto_front)
            detailed_findings.append({
                'finding_id': 'dominated_solutions',
                'title': '存在被支配解',
                'description': f'{dominated_count}个结果被其他结果支配，存在优化空间',
                'severity': 'medium',
                'implication': '可以参考帕累托最优解进行参数调整'
            })
        
        # 推荐
        recommendations = []
        if summary['efficiency_quality_correlation'] < -0.3:
            recommendations.append('效率和质量存在负相关，需要找到平衡点')
        if len(pareto_front) < len(extraction_results):
            recommendations.append('参考帕累托最优解进行参数优化')
        recommendations.append('根据具体应用场景选择合适的权衡策略')
        
        return AnalysisReport(
            report_id=report_id,
            analysis_type='efficiency_quality_tradeoff',
            summary=summary,
            detailed_findings=detailed_findings,
            recommendations=recommendations,
            visualizations=['pareto_front', 'tradeoff_heatmap'],
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_tradeoff_score(self, 
                                  efficiencies: np.ndarray, 
                                  qualities: np.ndarray) -> float:
        """计算权衡评分"""
        if len(efficiencies) == 0 or len(qualities) == 0:
            return 0.0
        
        # 归一化
        eff_norm = (efficiencies - np.min(efficiencies)) / (np.max(efficiencies) - np.min(efficiencies) + 1e-8)
        qual_norm = (qualities - np.min(qualities)) / (np.max(qualities) - np.min(qualities) + 1e-8)
        
        # 计算综合评分（几何平均）
        scores = np.sqrt(eff_norm * qual_norm)
        
        return float(np.mean(scores))
    
    def generate_comprehensive_report(self,
                                     extraction_results: List[Any]) -> Dict[str, AnalysisReport]:
        """
        生成综合分析报告
        
        Args:
            extraction_results: 要素抽取结果列表
            
        Returns:
            包含各类分析报告的字典
        """
        reports = {
            'confidence_pattern': self.analyze_confidence_patterns(extraction_results),
            'compression_effect': self.analyze_compression_effectiveness(extraction_results),
            'tradeoff': self.analyze_efficiency_quality_tradeoff(extraction_results)
        }
        
        self.analysis_history.extend(list(reports.values()))
        
        return reports
    
    def save_reports(self, 
                    reports: Dict[str, AnalysisReport], 
                    output_dir: str = "element_analysis"):
        """保存分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, report in reports.items():
            report_path = os.path.join(output_dir, f"{name}_report_{timestamp}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            
            print(f"报告已保存: {report_path}")


class EfficiencyQualityOptimizer:
    """
    效率-质量优化器
    
    基于要素分析结果，优化RPC参数以获得最佳的效率-质量权衡。
    """
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_params(self,
                       extraction_results: List[Any],
                       target_tradeoff: float = 0.5) -> Dict[str, Any]:
        """
        优化RPC参数
        
        Args:
            extraction_results: 要素抽取结果列表
            target_tradeoff: 目标权衡点（0=重质量，1=重效率）
            
        Returns:
            优化后的参数建议
        """
        # 分析当前结果
        analyzer = ElementAnalyzer()
        reports = analyzer.generate_comprehensive_report(extraction_results)
        
        tradeoff_report = reports['tradeoff']
        compression_report = reports['compression_effect']
        
        # 计算最优权衡点
        current_score = tradeoff_report.summary.get('tradeoff_score', 0)
        
        # 建议参数调整
        suggestions = {
            'current_parameters': {
                'P': 1024,
                'R': 32,
                'c': 4
            },
            'suggested_parameters': {},
            'rationale': []
        }
        
        # 根据分析结果给出建议
        if current_score < target_tradeoff:
            # 需要提高效率
            suggestions['suggested_parameters'] = {
                'P': 512,      # 更频繁压缩
                'R': 16,       # 减小保留窗口
                'c': 2         # 增大压缩比
            }
            suggestions['rationale'].append('当前效率较低，建议增加压缩频率和压缩比')
        else:
            # 需要提高质量
            suggestions['suggested_parameters'] = {
                'P': 2048,     # 减少压缩频率
                'R': 64,       # 增大保留窗口
                'c': 8         # 减小压缩比
            }
            suggestions['rationale'].append('当前质量可能受影响，建议降低压缩强度')
        
        # 添加通用建议
        if compression_report.summary.get('mean_compression_ratio', 1.0) < 1.5:
            suggestions['rationale'].append('压缩效果不显著，建议调整问题类型或使用保守参数')
        
        self.optimization_history.append(suggestions)
        
        return suggestions
    
    def compare_parameters(self,
                          results_a: List[Any],
                          results_b: List[Any],
                          labels: Tuple[str, str] = ("A", "B")) -> Dict[str, Any]:
        """
        比较两组参数的效率-质量权衡
        
        Args:
            results_a: 第一组结果
            results_b: 第二组结果
            labels: 组标签
            
        Returns:
            比较结果
        """
        analyzer = ElementAnalyzer()
        
        report_a = analyzer.analyze_efficiency_quality_tradeoff(results_a)
        report_b = analyzer.analyze_efficiency_quality_tradeoff(results_b)
        
        return {
            'label_a': labels[0],
            'label_b': labels[1],
            'a_efficiency': report_a.summary.get('mean_efficiency', 0),
            'b_efficiency': report_b.summary.get('mean_efficiency', 0),
            'a_quality': report_a.summary.get('mean_quality', 0),
            'b_quality': report_b.summary.get('mean_quality', 0),
            'a_tradeoff': report_a.summary.get('tradeoff_score', 0),
            'b_tradeoff': report_b.summary.get('tradeoff_score', 0),
            'winner': 'A' if report_a.summary.get('tradeoff_score', 0) > report_b.summary.get('tradeoff_score', 0) else 'B'
        }


def visualize_comparison(results_no_rpc: List[Any],
                        results_rpc: List[Any],
                        output_dir: str = "element_analysis"):
    """
    可视化对比RPC和非RPC的结果
    
    Args:
        results_no_rpc: 无RPC的结果
        results_rpc: 有RPC的结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 置信度对比
    ax1 = axes[0, 0]
    confs_no_rpc = []
    confs_rpc = []
    
    for result in results_no_rpc:
        confs_no_rpc.extend([e.token_confidence for e in result.local_confidence_elements])
    for result in results_rpc:
        confs_rpc.extend([e.token_confidence for e in result.local_confidence_elements])
    
    ax1.hist(confs_no_rpc, bins=30, alpha=0.6, label='No RPC', density=True)
    ax1.hist(confs_rpc, bins=30, alpha=0.6, label='With RPC', density=True)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Density')
    ax1.set_title('Confidence Distribution Comparison')
    ax1.legend()
    
    # 图2: 效率-质量对比
    ax2 = axes[0, 1]
    eff_no_rpc = [r.overall_efficiency_score for r in results_no_rpc]
    qual_no_rpc = [r.reasoning_quality_score for r in results_no_rpc]
    eff_rpc = [r.overall_efficiency_score for r in results_rpc]
    qual_rpc = [r.reasoning_quality_score for r in results_rpc]
    
    ax2.scatter(eff_no_rpc, qual_no_rpc, alpha=0.6, label='No RPC', s=100)
    ax2.scatter(eff_rpc, qual_rpc, alpha=0.6, label='With RPC', s=100)
    ax2.set_xlabel('Efficiency Score')
    ax2.set_ylabel('Quality Score')
    ax2.set_title('Efficiency vs Quality')
    ax2.legend()
    
    # 图3: 压缩比分布（仅RPC）
    ax3 = axes[1, 0]
    ratios_rpc = []
    for result in results_rpc:
        ratios_rpc.extend([e.compression_ratio for e in result.compression_path_elements])
    
    if ratios_rpc:
        ax3.hist(ratios_rpc, bins=20, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Compression Ratio')
        ax3.set_ylabel('Frequency')
        ax3.set_title('RPC Compression Ratio Distribution')
        ax3.axvline(np.mean(ratios_rpc), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(ratios_rpc):.2f}x')
        ax3.legend()
    
    # 图4: 综合评分对比
    ax4 = axes[1, 1]
    categories = ['Efficiency', 'Quality', 'Tradeoff']
    scores_no_rpc = [
        np.mean([r.overall_efficiency_score for r in results_no_rpc]),
        np.mean([r.reasoning_quality_score for r in results_no_rpc]),
        np.mean([r.overall_efficiency_score * r.reasoning_quality_score for r in results_no_rpc])
    ]
    scores_rpc = [
        np.mean([r.overall_efficiency_score for r in results_rpc]),
        np.mean([r.reasoning_quality_score for r in results_rpc]),
        np.mean([r.overall_efficiency_score * r.reasoning_quality_score for r in results_rpc])
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax4.bar(x - width/2, scores_no_rpc, width, label='No RPC')
    ax4.bar(x + width/2, scores_rpc, width, label='With RPC')
    ax4.set_ylabel('Score')
    ax4.set_title('Overall Score Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rpc_comparison_{timestamp}.png'), dpi=150)
    plt.close()
    
    print(f"对比可视化已保存到: {output_dir}/rpc_comparison_{timestamp}.png")


if __name__ == "__main__":
    # 示例用法
    print("要素分析模块示例")
    
    # 假设已经加载了抽取结果
    # results = [...]
    
    # 创建分析器
    analyzer = ElementAnalyzer()
    
    # 生成综合报告
    # reports = analyzer.generate_comprehensive_report(results)
    
    # 保存报告
    # analyzer.save_reports(reports)
    
    print("要素分析模块初始化完成")
