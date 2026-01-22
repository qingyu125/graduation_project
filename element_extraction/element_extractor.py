"""
要素抽取模块 - 面向高效推理的要素抽取与应用算法设计与实现

本模块实现两个维度的要素抽取：
1. 局部置信度要素抽取 - 从DeepConf推理过程中抽取
2. 路径压缩要素抽取 - 从RPC压缩过程中抽取

Copyright (c) MiniMax Agent
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


@dataclass
class LocalConfidenceElement:
    """局部置信度要素"""
    # 基础信息
    trace_id: int
    token_position: int
    
    # 置信度指标
    token_confidence: float           # 单个token的置信度
    sliding_window_confidence: float  # 滑动窗口平均置信度
    local_mean_confidence: float      # 局部均值置信度
    
    # 位置信息
    relative_position: float          # 在trace中的相对位置（0-1）
    is_critical_point: bool           # 是否为关键置信点
    
    # 上下文信息
    preceding_confidences: List[float] = field(default_factory=list)
    following_confidences: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': int(self.trace_id),
            'token_position': int(self.token_position),
            'token_confidence': float(self.token_confidence),
            'sliding_window_confidence': float(self.sliding_window_confidence),
            'local_mean_confidence': float(self.local_mean_confidence),
            'relative_position': float(self.relative_position),
            'is_critical_point': bool(self.is_critical_point),
            'preceding_confidences': [float(c) for c in self.preceding_confidences],
            'following_confidences': [float(c) for c in self.following_confidences]
        }


@dataclass
class CompressionPathElement:
    """路径压缩要素 - 记录RPC压缩过程中的关键信息"""
    # 压缩基本信息
    compression_id: int
    layer_id: int                      # 压缩发生的层
    position: int                      # 压缩发生的位置
    
    # 压缩参数
    P: int                             # 压缩间隔
    R: int                             # 保留窗口大小
    c: int                             # 压缩比
    
    # 压缩前状态
    original_token_count: int
    original_kv_size: float            # 原始KV缓存大小（估算）
    
    # 压缩后状态
    compressed_token_count: int
    compressed_kv_size: float          # 压缩后KV缓存大小（估算）
    
    # 压缩效果
    compression_ratio: float           # 实际压缩比
    tokens_compressed: int             # 被压缩的token数量
    tokens_kept: int                   # 保留的token数量
    
    # 重要性分布
    top_importance_scores: List[float] = field(default_factory=list)
    importance_distribution: Dict[str, float] = field(default_factory=dict)
    
    # 选择器信息
    selector_type: str = 'recent'
    aggregation_method: str = 'all'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'compression_id': int(self.compression_id),
            'layer_id': int(self.layer_id),
            'position': int(self.position),
            'P': int(self.P),
            'R': int(self.R),
            'c': int(self.c),
            'original_token_count': int(self.original_token_count),
            'original_kv_size': float(self.original_kv_size),
            'compressed_token_count': int(self.compressed_token_count),
            'compressed_kv_size': float(self.compressed_kv_size),
            'compression_ratio': float(self.compression_ratio),
            'tokens_compressed': int(self.tokens_compressed),
            'tokens_kept': int(self.tokens_kept),
            'top_importance_scores': [float(s) for s in self.top_importance_scores],
            'importance_distribution': {k: float(v) for k, v in self.importance_distribution.items()},
            'selector_type': str(self.selector_type),
            'aggregation_method': str(self.aggregation_method)
        }


@dataclass
class ElementExtractionResult:
    """要素抽取综合结果"""
    # 问题信息
    question_id: int
    question_text: str
    ground_truth: Optional[str]
    
    # 局部置信度要素
    local_confidence_elements: List[LocalConfidenceElement] = field(default_factory=list)
    confidence_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 路径压缩要素
    compression_path_elements: List[CompressionPathElement] = field(default_factory=list)
    compression_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 综合分析
    overall_efficiency_score: float = 0.0
    reasoning_quality_score: float = 0.0
    
    # 元数据
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    rpc_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question_id': int(self.question_id) if self.question_id is not None else None,
            'question_text': str(self.question_text) if self.question_text is not None else '',
            'ground_truth': str(self.ground_truth) if self.ground_truth is not None else None,
            'local_confidence_elements': [e.to_dict() for e in self.local_confidence_elements],
            'confidence_statistics': self._convert_stats_to_native(self.confidence_statistics),
            'compression_path_elements': [e.to_dict() for e in self.compression_path_elements],
            'compression_statistics': self._convert_stats_to_native(self.compression_statistics),
            'overall_efficiency_score': float(self.overall_efficiency_score),
            'reasoning_quality_score': float(self.reasoning_quality_score),
            'extraction_timestamp': str(self.extraction_timestamp),
            'rpc_enabled': bool(self.rpc_enabled)
        }
    
    def _convert_stats_to_native(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """将统计字典中的numpy类型转换为Python原生类型"""
        result = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                result[key] = self._convert_stats_to_native(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [self._convert_item(v) for v in value]
            else:
                result[key] = self._convert_item(value)
        return result
    
    def _convert_item(self, item: Any) -> Any:
        """将单个item转换为Python原生类型"""
        if isinstance(item, np.integer):
            return int(item)
        elif isinstance(item, np.floating):
            return float(item)
        elif isinstance(item, np.bool_):
            return bool(item)
        elif isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: self._convert_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._convert_item(v) for v in item]
        else:
            return item


class LocalConfidenceExtractor:
    """
    局部置信度要素抽取器
    
    从DeepConf推理过程中抽取局部置信度要素，
    包括单token置信度、滑动窗口置信度、关键置信点等。
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 critical_threshold: float = 0.3,
                 importance_percentile: float = 0.1):
        """
        初始化局部置信度要素抽取器
        
        Args:
            window_size: 滑动窗口大小
            critical_threshold: 关键置信点阈值（低于此值认为是关键点）
            importance_percentile: 重要置信点的百分位
        """
        self.window_size = window_size
        self.critical_threshold = critical_threshold
        self.importance_percentile = importance_percentile
    
    def extract_from_traces(self, 
                           traces: List[Dict[str, Any]], 
                           question_text: str = "",
                           ground_truth: Optional[str] = None) -> Tuple[List[LocalConfidenceElement], Dict[str, Any]]:
        """
        从多个trace中抽取局部置信度要素
        
        Args:
            traces: DeepConf推理结果列表
            question_text: 问题文本
            ground_truth: 正确答案
            
        Returns:
            要素列表和统计信息
        """
        all_elements = []
        all_confs = []
        trace_confs_dict = {}
        
        for trace_idx, trace in enumerate(traces):
            confs = trace.get('confs', [])
            if not confs:
                continue
            
            # 为每个token创建要素
            trace_elements = self._extract_from_single_trace(
                confs, trace_idx, len(traces)
            )
            all_elements.extend(trace_elements)
            all_confs.extend(confs)
            trace_confs_dict[trace_idx] = confs
        
        # 计算统计信息
        statistics = self._compute_statistics(all_confs, trace_confs_dict)
        
        return all_elements, statistics
    
    def _extract_from_single_trace(self, 
                                   confs: List[float], 
                                   trace_id: int,
                                   total_traces: int) -> List[LocalConfidenceElement]:
        """从单个trace中抽取要素"""
        elements = []
        num_tokens = len(confs)
        
        if num_tokens == 0:
            return elements
        
        # 计算滑动窗口置信度
        sliding_confs = self._compute_sliding_window(confs, self.window_size)
        
        # 确定关键置信点（低置信度区域）
        threshold = np.percentile(confs, self.importance_percentile * 100)
        
        for pos in range(num_tokens):
            token_conf = confs[pos]
            window_conf = sliding_confs[min(pos, len(sliding_confs) - 1)] if sliding_confs else token_conf
            
            # 计算局部均值
            local_start = max(0, pos - self.window_size // 2)
            local_end = min(num_tokens, pos + self.window_size // 2 + 1)
            local_mean = np.mean(confs[local_start:local_end]) if local_end > local_start else token_conf
            
            element = LocalConfidenceElement(
                trace_id=trace_id,
                token_position=pos,
                token_confidence=token_conf,
                sliding_window_confidence=window_conf,
                local_mean_confidence=local_mean,
                relative_position=pos / num_tokens if num_tokens > 0 else 0,
                is_critical_point=token_conf < threshold,
                preceding_confidences=confs[max(0, pos-5):pos],
                following_confidences=confs[pos+1:min(pos+6, num_tokens)]
            )
            elements.append(element)
        
        return elements
    
    def _compute_sliding_window(self, confs: List[float], window_size: int) -> List[float]:
        """计算滑动窗口平均置信度"""
        if len(confs) < window_size:
            return [np.mean(confs)] if confs else []
        
        sliding_confs = []
        current_sum = sum(confs[:window_size])
        sliding_confs.append(current_sum / window_size)
        
        for i in range(1, len(confs) - window_size + 1):
            current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
            sliding_confs.append(current_sum / window_size)
        
        return sliding_confs
    
    def _compute_statistics(self, 
                           all_confs: List[float],
                           trace_confs_dict: Dict[int, List[float]]) -> Dict[str, Any]:
        """计算置信度统计信息"""
        if not all_confs:
            return {}
        
        statistics = {
            'global_mean': float(np.mean(all_confs)),
            'global_std': float(np.std(all_confs)),
            'global_min': float(np.min(all_confs)),
            'global_max': float(np.max(all_confs)),
            'total_tokens': len(all_confs),
            'trace_count': len(trace_confs_dict),
            'per_trace_statistics': {},
            'confidence_distribution': self._compute_distribution(all_confs),
            'critical_points_count': sum(1 for c in all_confs if c < np.percentile(all_confs, 10))
        }
        
        # 每个trace的统计
        for trace_id, confs in trace_confs_dict.items():
            if confs:
                statistics['per_trace_statistics'][trace_id] = {
                    'mean': float(np.mean(confs)),
                    'std': float(np.std(confs)),
                    'min': float(np.min(confs)),
                    'max': float(np.max(confs)),
                    'token_count': len(confs)
                }
        
        return statistics
    
    def _compute_distribution(self, confs: List[float], num_bins: int = 10) -> Dict[str, float]:
        """计算置信度分布"""
        if not confs:
            return {}
        
        hist, bin_edges = np.histogram(confs, bins=num_bins)
        total = len(confs)
        
        distribution = {}
        for i in range(num_bins):
            key = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            distribution[key] = hist[i] / total if total > 0 else 0
        
        return distribution


class CompressionPathExtractor:
    """
    路径压缩要素抽取器
    
    从RPC压缩过程中抽取路径压缩要素，
    记录压缩位置、压缩比例、重要性分布等信息。
    """
    
    def __init__(self, 
                 P: int = 1024,
                 R: int = 32,
                 c: int = 4,
                 kv_per_token_mb: float = 0.001):
        """
        初始化路径压缩要素抽取器
        
        Args:
            P: 压缩间隔
            R: 保留窗口大小
            c: 压缩比
            kv_per_token_mb: 每个token的KV缓存估算大小（MB）
        """
        self.P = P
        self.R = R
        self.c = c
        self.kv_per_token_mb = kv_per_token_mb
        self.compression_events = []
    
    def record_compression_event(self,
                                layer_id: int,
                                position: int,
                                original_count: int,
                                compressed_count: int,
                                importance_scores: Optional[List[float]] = None):
        """
        记录一次压缩事件
        
        Args:
            layer_id: 压缩发生的层ID
            position: 压缩发生的位置
            original_count: 压缩前token数量
            compressed_count: 压缩后token数量
            importance_scores: 重要性分数列表（如果有）
        """
        event = {
            'layer_id': layer_id,
            'position': position,
            'P': self.P,
            'R': self.R,
            'c': self.c,
            'original_token_count': original_count,
            'compressed_token_count': compressed_count,
            'compression_ratio': original_count / compressed_count if compressed_count > 0 else 1.0,
            'tokens_compressed': original_count - compressed_count,
            'tokens_kept': compressed_count,
            'original_kv_size': original_count * self.kv_per_token_mb,
            'compressed_kv_size': compressed_count * self.kv_per_token_mb,
            'timestamp': datetime.now().isoformat()
        }
        
        if importance_scores:
            event['top_importance_scores'] = sorted(importance_scores, reverse=True)[:10]
            event['importance_distribution'] = self._compute_importance_distribution(importance_scores)
        
        self.compression_events.append(event)
    
    def _compute_importance_distribution(self, scores: List[float]) -> Dict[str, float]:
        """计算重要性分数分布"""
        if not scores:
            return {}
        
        # 将分数分为高、中、低三个等级
        high_threshold = np.percentile(scores, 66)
        low_threshold = np.percentile(scores, 33)
        
        distribution = {
            'high': sum(1 for s in scores if s >= high_threshold) / len(scores),
            'medium': sum(1 for s in scores if low_threshold <= s < high_threshold) / len(scores),
            'low': sum(1 for s in scores if s < low_threshold) / len(scores)
        }
        
        return distribution
    
    def extract_from_result(self, 
                           result: Dict[str, Any]) -> Tuple[List[CompressionPathElement], Dict[str, Any]]:
        """
        从RPC结果中抽取路径压缩要素

        Args:
            result: RPC推理结果
            
        Returns:
            要素列表和统计信息
        """
        elements = []

        # 如果有记录的压缩事件，使用这些事件
        if self.compression_events:
            for idx, event in enumerate(self.compression_events):
                element = CompressionPathElement(
                    compression_id=idx,
                    layer_id=event.get('layer_id', 0),
                    position=event.get('position', 0),
                    P=event.get('P', self.P),
                    R=event.get('R', self.R),
                    c=event.get('c', self.c),
                    original_token_count=event.get('original_token_count', 0),
                    original_kv_size=event.get('original_kv_size', 0),
                    compressed_token_count=event.get('compressed_token_count', 0),
                    compressed_kv_size=event.get('compressed_kv_size', 0),
                    compression_ratio=event.get('compression_ratio', 1.0),
                    tokens_compressed=event.get('tokens_compressed', 0),
                    tokens_kept=event.get('tokens_kept', 0),
                    top_importance_scores=event.get('top_importance_scores', []),
                    importance_distribution=event.get('importance_distribution', {}),
                    selector_type='recent',
                    aggregation_method='all'
                )
                elements.append(element)
        elif result.get('rpc_enabled', False):
            # 如果没有显式记录的压缩事件，但RPC已启用
            # 基于RPC参数和Token数估算压缩事件
            estimated_elements = self._estimate_compression_events_from_result(result)
            elements.extend(estimated_elements)

        # 计算统计信息（传递elements以计算估算事件的统计）
        statistics = self._compute_statistics(elements)

        return elements, statistics

    def _estimate_compression_events_from_result(self, result: Dict[str, Any]) -> List[CompressionPathElement]:
        """
        从RPC结果中估算压缩事件
        
        当压缩事件没有被显式记录时，基于以下信息估算：
        1. RPC参数 (P, R, c)
        2. 推理Token数
        3. 置信度分布（用于估计重要性）
        """
        elements = []
        
        # 提取traces中的置信度信息
        traces = result.get('all_traces', [])
        if not traces:
            traces = result.get('final_traces', [])
        
        if not traces:
            return elements
        
        # 收集所有置信度数据用于重要性估计
        all_confs = []
        for trace in traces:
            confs = trace.get('confs', [])
            all_confs.extend(confs)
        
        if not all_confs:
            return elements
        
        # RPC参数
        P = self.P  # 压缩间隔
        R = self.R  # 保留窗口
        c = self.c  # 压缩比
        T = int(P / c)  # 每次保留的token数
        
        # 估算发生的压缩次数
        # 假设在位置P, 2P, 3P...处发生压缩
        total_tokens = len(all_confs)
        compression_positions = list(range(P, total_tokens, P))
        
        # 计算理论压缩比
        theoretical_ratio = P / T  # = c
        
        # 估算每次压缩的信息
        num_layers = 28  # 假设Qwen2-8B有28层
        
        for idx, pos in enumerate(compression_positions):
            # 计算该位置应该保留的token数
            tokens_to_keep = T + R  # T个重要token + R个最近token
            tokens_compressed = P - tokens_to_keep
            
            # 基于置信度分布估算重要性
            # 假设置信度较低的token更可能被压缩
            start_idx = max(0, pos - P)
            end_idx = min(pos, total_tokens)
            segment_confs = all_confs[start_idx:end_idx]
            
            if segment_confs:
                # 计算该段的重要性分布
                importance_scores = self._estimate_importance_from_confidences(segment_confs)
            else:
                importance_scores = []
            
            # 为每层创建一个压缩事件
            for layer_id in range(min(num_layers, 4)):  # 最多记录4层
                element = CompressionPathElement(
                    compression_id=len(elements),
                    layer_id=layer_id,
                    position=pos,
                    P=P,
                    R=R,
                    c=c,
                    original_token_count=P,
                    original_kv_size=P * self.kv_per_token_mb,
                    compressed_token_count=tokens_to_keep,
                    compressed_kv_size=tokens_to_keep * self.kv_per_token_mb,
                    compression_ratio=theoretical_ratio,
                    tokens_compressed=tokens_compressed,
                    tokens_kept=tokens_to_keep,
                    top_importance_scores=sorted(importance_scores, reverse=True)[:10] if importance_scores else [],
                    importance_distribution=self._compute_importance_distribution(importance_scores) if importance_scores else {},
                    selector_type='recent',
                    aggregation_method='all'
                )
                elements.append(element)
        
        return elements
    
    def _estimate_importance_from_confidences(self, confs: List[float]) -> List[float]:
        """
        从置信度数据估算token重要性
        
        假设：置信度越高的token越重要（越不可能被压缩）
        重要性 = 归一化的置信度
        """
        if not confs:
            return []
        
        # 归一化置信度到0-1范围作为重要性分数
        min_conf = min(confs)
        max_conf = max(confs)
        
        if max_conf - min_conf < 1e-8:
            return [0.5] * len(confs)
        
        importance = [(c - min_conf) / (max_conf - min_conf) for c in confs]
        
        return importance
    
    def _compute_statistics(self, elements: List[CompressionPathElement] = None) -> Dict[str, Any]:
        """计算压缩统计信息"""
        if elements is None:
            elements = []
        
        if not self.compression_events and not elements:
            return {
                'total_compressions': 0,
                'avg_compression_ratio': 0,
                'total_tokens_compressed': 0,
                'total_kv_saved_mb': 0,
                'note': '无显式压缩事件记录'
            }
        
        # 合并显式事件和估算元素
        all_ratios = []
        tokens_compressed = 0
        kv_saved = 0
        
        # 处理显式事件
        for event in self.compression_events:
            all_ratios.append(event['compression_ratio'])
            tokens_compressed += event['tokens_compressed']
            kv_saved += (event['original_kv_size'] - event['compressed_kv_size'])
        
        # 处理估算元素
        for elem in elements:
            all_ratios.append(elem.compression_ratio)
            tokens_compressed += elem.tokens_compressed
            kv_saved += (elem.original_kv_size - elem.compressed_kv_size)
        
        is_estimated = len(elements) > 0 and not self.compression_events
        
        return {
            'total_compressions': len(self.compression_events) + len(elements),
            'avg_compression_ratio': float(np.mean(all_ratios)) if all_ratios else 0,
            'compression_ratio_std': float(np.std(all_ratios)) if all_ratios else 0,
            'min_compression_ratio': float(np.min(all_ratios)) if all_ratios else 0,
            'max_compression_ratio': float(np.max(all_ratios)) if all_ratios else 0,
            'total_tokens_compressed': tokens_compressed,
            'total_kv_saved_mb': kv_saved,
            'avg_tokens_per_compression': tokens_compressed / (len(self.compression_events) + len(elements)) if (len(self.compression_events) + len(elements)) > 0 else 0,
            'compression_efficiency': float(np.mean(all_ratios)) * min(1.0, tokens_compressed / 10000) if all_ratios else 0,
            'is_estimated': is_estimated
        }


class ElementExtractionPipeline:
    """
    要素抽取流程控制器
    
    整合局部置信度和路径压缩两个维度的要素抽取，
    生成综合分析结果。
    """
    
    def __init__(self, 
                 confidence_window_size: int = 50,
                 critical_percentile: float = 0.1,
                 rpc_P: int = 1024,
                 rpc_R: int = 32,
                 rpc_c: int = 4):
        """
        初始化要素抽取流程
        
        Args:
            confidence_window_size: 置信度滑动窗口大小
            critical_percentile: 关键置信点百分位
            rpc_P: RPC压缩间隔
            rpc_R: RPC保留窗口
            rpc_c: RPC压缩比
        """
        self.confidence_extractor = LocalConfidenceExtractor(
            window_size=confidence_window_size,
            critical_threshold=0.3,
            importance_percentile=critical_percentile
        )
        
        self.compression_extractor = CompressionPathExtractor(
            P=rpc_P,
            R=rpc_R,
            c=rpc_c
        )
        
        self.results = []
    
    def extract_from_result(self, 
                           result: Dict[str, Any],
                           question_text: str = "",
                           ground_truth: Optional[str] = None) -> ElementExtractionResult:
        """
        从推理结果中抽取所有要素
        
        Args:
            result: DeepConf/RPC推理结果
            question_text: 问题文本
            ground_truth: 正确答案
            
        Returns:
            综合要素抽取结果
        """
        question_id = result.get('question_id', 0)
        rpc_enabled = result.get('rpc_enabled', False)
        
        # 提取traces
        traces = result.get('all_traces', [])
        if not traces:
            traces = result.get('final_traces', [])
        if not traces:
            traces = result.get('warmup_traces', [])
        
        # 抽取局部置信度要素
        confidence_elements, confidence_stats = self.confidence_extractor.extract_from_traces(
            traces, question_text, ground_truth
        )
        
        # 抽取路径压缩要素
        compression_elements, compression_stats = self.compression_extractor.extract_from_result(result)
        
        # 计算综合评分
        efficiency_score = self._compute_efficiency_score(compression_stats)
        quality_score = self._compute_quality_score(confidence_stats)
        
        # 创建结果对象
        extraction_result = ElementExtractionResult(
            question_id=question_id,
            question_text=question_text,
            ground_truth=ground_truth,
            local_confidence_elements=confidence_elements,
            confidence_statistics=confidence_stats,
            compression_path_elements=compression_elements,
            compression_statistics=compression_stats,
            overall_efficiency_score=efficiency_score,
            reasoning_quality_score=quality_score,
            rpc_enabled=rpc_enabled
        )
        
        self.results.append(extraction_result)
        
        return extraction_result
    
    def _compute_efficiency_score(self, compression_stats: Dict[str, Any]) -> float:
        """计算效率评分（基于压缩效果）"""
        if not compression_stats or compression_stats.get('total_compressions', 0) == 0:
            return 0.0
        
        avg_ratio = compression_stats.get('avg_compression_ratio', 1.0)
        total_compressions = compression_stats.get('total_compressions', 1)
        
        # 效率评分 = 压缩比 * 压缩频率（归一化）
        efficiency = min(avg_ratio, 5.0) * min(total_compressions / 10, 1.0)
        
        return round(efficiency, 4)
    
    def _compute_quality_score(self, confidence_stats: Dict[str, Any]) -> float:
        """计算推理质量评分（基于置信度）"""
        if not confidence_stats:
            return 0.0
        
        mean_conf = confidence_stats.get('global_mean', 0)
        std_conf = confidence_stats.get('global_std', 0)
        critical_ratio = confidence_stats.get('critical_points_count', 0) / max(confidence_stats.get('total_tokens', 1), 1)
        
        # 质量评分 = 高置信度 + 低波动 + 少关键点
        quality = (1.0 - min(mean_conf, 1.0)) * 0.5 + (1.0 - min(std_conf, 0.5)) * 0.3 + (1.0 - min(critical_ratio, 0.2)) * 0.2
        
        return round(quality, 4)
    
    def save_results(self, output_dir: str = "element_extraction_results"):
        """保存所有抽取结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存为JSON
        json_path = os.path.join(output_dir, f"extraction_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
        
        # 保存为Pickle
        pkl_path = os.path.join(output_dir, f"extraction_results_{timestamp}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"要素抽取结果已保存到:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pkl_path}")
        
        return json_path, pkl_path
    
    def generate_visualization(self, output_dir: str = "element_extraction_results"):
        """生成可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            print("没有可可视化的结果")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 图1: 置信度分布
        self._plot_confidence_distribution(output_dir, timestamp)
        
        # 图2: 压缩效果
        self._plot_compression_effect(output_dir, timestamp)
        
        # 图3: 综合评分雷达图
        self._plot_comprehensive_scores(output_dir, timestamp)
        
        print(f"可视化图表已保存到: {output_dir}")
    
    def _plot_confidence_distribution(self, output_dir: str, timestamp: str):
        """绘制置信度分布图"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 收集所有置信度数据
        all_confs = []
        for result in self.results:
            for element in result.local_confidence_elements:
                all_confs.append(element.token_confidence)
        
        if not all_confs:
            return
        
        # 子图1: 置信度直方图
        ax1 = axes[0, 0]
        ax1.hist(all_confs, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Token Confidence Distribution')
        ax1.axvline(np.mean(all_confs), color='r', linestyle='--', label=f'Mean: {np.mean(all_confs):.3f}')
        ax1.legend()
        
        # 子图2: 滑动窗口置信度趋势
        ax2 = axes[0, 1]
        for i, result in enumerate(self.results[:3]):  # 最多显示3个trace
            window_confs = [e.sliding_window_confidence for e in result.local_confidence_elements[:100]]
            if window_confs:
                ax2.plot(range(len(window_confs)), window_confs, label=f'Trace {i}', alpha=0.7)
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Sliding Window Confidence')
        ax2.set_title('Confidence Trend Over Tokens')
        ax2.legend()
        
        # 子图3: 置信度箱线图（按trace）
        ax3 = axes[1, 0]
        trace_confs = []
        trace_labels = []
        for i, result in enumerate(self.results[:5]):
            confs = [e.token_confidence for e in result.local_confidence_elements]
            if confs:
                trace_confs.append(confs)
                trace_labels.append(f'Trace {i}')
        
        if trace_confs:
            ax3.boxplot(trace_confs, labels=trace_labels)
            ax3.set_xlabel('Trace')
            ax3.set_ylabel('Confidence')
            ax3.set_title('Confidence Distribution by Trace')
        
        # 子图4: 关键置信点位置分布
        ax4 = axes[1, 1]
        critical_positions = []
        for result in self.results:
            for element in result.local_confidence_elements:
                if element.is_critical_point:
                    critical_positions.append(element.relative_position)
        
        if critical_positions:
            ax4.hist(critical_positions, bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax4.set_xlabel('Relative Position (0-1)')
            ax4.set_ylabel('Critical Points Count')
            ax4.set_title('Critical Confidence Points Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confidence_distribution_{timestamp}.png'), dpi=150)
        plt.close()
    
    def _plot_compression_effect(self, output_dir: str, timestamp: str):
        """绘制压缩效果图"""
        if not any(r.rpc_enabled for r in self.results):
            print("没有RPC结果可可视化")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 收集压缩数据
        all_ratios = []
        all_tokens_saved = []
        
        for result in self.results:
            if result.rpc_enabled:
                for element in result.compression_path_elements:
                    all_ratios.append(element.compression_ratio)
                    all_tokens_saved.append(element.tokens_compressed)
        
        # 子图1: 压缩比分布
        ax1 = axes[0]
        if all_ratios:
            ax1.hist(all_ratios, bins=20, edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Compression Ratio')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Compression Ratio Distribution')
            ax1.axvline(np.mean(all_ratios), color='r', linestyle='--', label=f'Mean: {np.mean(all_ratios):.2f}x')
            ax1.legend()
        
        # 子图2: 每层压缩比
        ax2 = axes[1]
        layer_ratios = {}
        for result in self.results:
            if result.rpc_enabled:
                for element in result.compression_path_elements:
                    layer_id = element.layer_id
                    if layer_id not in layer_ratios:
                        layer_ratios[layer_id] = []
                    layer_ratios[layer_id].append(element.compression_ratio)
        
        if layer_ratios:
            layers = sorted(layer_ratios.keys())
            means = [np.mean(layer_ratios[l]) for l in layers]
            stds = [np.std(layer_ratios[l]) for l in layers]
            ax2.bar(range(len(layers)), means, yerr=stds, capsize=5)
            ax2.set_xlabel('Layer ID')
            ax2.set_ylabel('Mean Compression Ratio')
            ax2.set_title('Compression Ratio by Layer')
            ax2.set_xticks(range(len(layers)))
            ax2.set_xticklabels([str(l) for l in layers])
        
        # 子图3: 压缩效率随位置变化
        ax3 = axes[2]
        position_ratios = {}
        for result in self.results:
            if result.rpc_enabled:
                for element in result.compression_path_elements:
                    pos_bucket = element.position // 1000  # 每1000 token一个桶
                    if pos_bucket not in position_ratios:
                        position_ratios[pos_bucket] = []
                    position_ratios[pos_bucket].append(element.compression_ratio)
        
        if position_ratios:
            positions = sorted(position_ratios.keys())
            means = [np.mean(position_ratios[p]) for p in positions]
            ax3.plot([p * 1000 for p in positions], means, marker='o')
            ax3.set_xlabel('Token Position')
            ax3.set_ylabel('Mean Compression Ratio')
            ax3.set_title('Compression Efficiency Over Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'compression_effect_{timestamp}.png'), dpi=150)
        plt.close()
    
    def _plot_comprehensive_scores(self, output_dir: str, timestamp: str):
        """绘制综合评分图"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 子图1: 效率vs质量评分散点图
        ax1 = axes[0]
        efficiencies = [r.overall_efficiency_score for r in self.results]
        qualities = [r.reasoning_quality_score for r in self.results]
        
        ax1.scatter(efficiencies, qualities, alpha=0.6, s=100)
        ax1.set_xlabel('Efficiency Score')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Efficiency vs Quality Trade-off')
        
        # 添加标签
        for i, r in enumerate(self.results):
            ax1.annotate(f'Q{r.question_id}', (r.overall_efficiency_score, r.reasoning_quality_score))
        
        # 子图2: 评分对比柱状图
        ax2 = axes[1]
        x = range(len(self.results))
        eff_scores = [r.overall_efficiency_score for r in self.results]
        qual_scores = [r.reasoning_quality_score for r in self.results]
        
        width = 0.35
        ax2.bar([i - width/2 for i in x], eff_scores, width, label='Efficiency', alpha=0.7)
        ax2.bar([i + width/2 for i in x], qual_scores, width, label='Quality', alpha=0.7)
        ax2.set_xlabel('Question ID')
        ax2.set_ylabel('Score')
        ax2.set_title('Comprehensive Scores Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Q{r.question_id}' for r in self.results])
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comprehensive_scores_{timestamp}.png'), dpi=150)
        plt.close()


def load_and_extract(result_path: str,
                     question_text: str = "",
                     ground_truth: Optional[str] = None,
                     **extraction_kwargs) -> ElementExtractionResult:
    """
    便利函数：从文件加载结果并进行要素抽取
    
    Args:
        result_path: 结果文件路径（.pkl或.json）
        question_text: 问题文本
        ground_truth: 正确答案
        **extraction_kwargs: 其他抽取参数
        
    Returns:
        要素抽取结果
    """
    # 加载结果
    if result_path.endswith('.pkl'):
        with open(result_path, 'rb') as f:
            result = pickle.load(f)
    elif result_path.endswith('.json'):
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {result_path}")
    
    # 创建抽取管道
    pipeline = ElementExtractionPipeline(
        confidence_window_size=extraction_kwargs.get('confidence_window_size', 50),
        critical_percentile=extraction_kwargs.get('critical_percentile', 0.1),
        rpc_P=extraction_kwargs.get('rpc_P', 1024),
        rpc_R=extraction_kwargs.get('rpc_R', 32),
        rpc_c=extraction_kwargs.get('rpc_c', 4)
    )
    
    # 执行抽取
    return pipeline.extract_from_result(result, question_text, ground_truth)


if __name__ == "__main__":
    # 示例用法
    print("要素抽取模块示例")
    
    # 从文件加载结果并抽取要素
    result = load_and_extract(
        result_path="online_outputs/deepthink_online_rpc_qid0_ridonline_run_rpc_20251202_094650.pkl",
        question_text="Find the sum of all integer bases...",
        ground_truth="70"
    )
    
    print(f"抽取完成:")
    print(f"  问题ID: {result.question_id}")
    print(f"  局部置信度要素数量: {len(result.local_confidence_elements)}")
    print(f"  路径压缩要素数量: {len(result.compression_path_elements)}")
    print(f"  效率评分: {result.overall_efficiency_score}")
    print(f"  质量评分: {result.reasoning_quality_score}")
