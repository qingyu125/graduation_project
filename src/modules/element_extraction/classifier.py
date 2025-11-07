"""
要素分类器模块
对提取的要素进行分类、验证和质量评估
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import json
import numpy as np
from collections import Counter, defaultdict
import re

from extractor import ExtractedElement, ExtractedEntity, ExtractedRelation, ElementType
from ast_parser import ASTNode

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """质量等级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRELIABLE = "unreliable"


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    quality_level: QualityLevel
    confidence_score: float
    issues: List[str]
    suggestions: List[str]
    evidence_strength: float


@dataclass
class ClassificationResult:
    """分类结果"""
    category: str
    subcategory: str
    confidence: float
    reasoning: str
    evidence: List[str]


class ElementClassifier:
    """要素分类器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 质量阈值
        self.quality_thresholds = {
            QualityLevel.HIGH: 0.8,
            QualityLevel.MEDIUM: 0.6,
            QualityLevel.LOW: 0.4,
            QualityLevel.UNRELIABLE: 0.0
        }
        
        # 实体分类规则
        self.entity_subcategories = {
            'PERSON': {
                'NAME': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                'TITLE': r'\b(?:Mr|Mrs|Ms|Dr|Prof|CEO|President|Director)\b',
                'ROLE': r'\b(?:CEO|president|director|manager|chairman|professor|doctor)\b'
            },
            'ORG': {
                'COMPANY': r'\b[A-Z][A-Z][A-Za-z]*(?:\s+[A-Z][A-Z][A-Za-z]*)*\s+(?:Inc|Corp|Ltd|LLC)\b',
                'INSTITUTION': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|Institute|School)\b',
                'GOVERNMENT': r'\b(?:Government|Administration|Ministry|Department)\b'
            },
            'LOC': {
                'CITY': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b',
                'COUNTRY': r'\b(?:United States|USA|China|Japan|Germany|UK|France|Italy|Spain|Canada)\b',
                'BUILDING': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Building|Tower|Center|Plaza)\b'
            }
        }
        
        # 关系分类规则
        self.relation_subcategories = {
            'PHYSICAL': {
                'LOCATED_IN': r'(?:located|situated)\s+in',
                'PART_OF': r'part of',
                'NEAR': r'(?:near|close to|adjacent to)'
            },
            'AFFILIATION': {
                'WORKS_FOR': r'(?:works?|works?)\s+for',
                'STUDIES_AT': r'(?:studied|study)\s+at',
                'MEMBER_OF': r'(?:member|employee)\s+of'
            },
            'PERSONAL': {
                'MARRIED_TO': r'(?:married|coupled)\s+(?:to|with)',
                'FAMILY': r'(?:father|mother|son|daughter|brother|sister)\s+of',
                'SIBLING': r'(?:brother|sister)\s+of'
            }
        }
        
        # 质量评估特征
        self.quality_features = {
            'length_ratio': 0.2,  # 要素长度与上下文长度比
            'keyword_density': 0.3,  # 关键词密度
            'context_relevance': 0.25,  # 上下文相关性
            'pattern_match': 0.25  # 模式匹配得分
        }
    
    def classify_and_validate_elements(self, elements: Dict[str, Any]) -> Dict[str, Any]:
        """
        对提取的要素进行分类和验证
        
        Args:
            elements: 包含提取要素的字典
            
        Returns:
            分类和验证结果
        """
        try:
            logger.info("开始要素分类和验证")
            
            # 分类实体
            classified_entities = self.classify_entities(
                elements.get('extracted_entities', [])
            )
            
            # 分类关系
            classified_relations = self.classify_relations(
                elements.get('extracted_relations', [])
            )
            
            # 验证要素
            validation_results = self.validate_elements(
                classified_entities, classified_relations,
                elements.get('extracted_events', []),
                elements.get('extracted_attributes', [])
            )
            
            # 构建结果
            result = {
                'classified_entities': classified_entities,
                'classified_relations': classified_relations,
                'validation_results': validation_results,
                'quality_metrics': self._calculate_quality_metrics(validation_results),
                'recommendations': self._generate_recommendations(validation_results)
            }
            
            logger.info(f"分类验证完成 - 实体: {len(classified_entities)}, 关系: {len(classified_relations)}")
            return result
            
        except Exception as e:
            logger.error(f"要素分类验证失败: {str(e)}")
            raise
    
    def classify_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对实体进行分类"""
        classified = []
        
        for entity in entities:
            try:
                # 基本分类
                category = entity.get('category', 'MISC')
                subcategory = self._classify_entity_subcategory(entity, category)
                
                # 质量评估
                quality_result = self._assess_entity_quality(entity)
                
                # 构建结果
                classified_entity = {
                    **entity,
                    'subcategory': subcategory,
                    'quality_assessment': {
                        'level': quality_result.quality_level.value,
                        'confidence': quality_result.confidence_score,
                        'issues': quality_result.issues,
                        'suggestions': quality_result.suggestions
                    },
                    'classification_result': {
                        'category': category,
                        'subcategory': subcategory,
                        'confidence': quality_result.confidence_score,
                        'reasoning': self._generate_classification_reasoning(entity, category, subcategory)
                    }
                }
                
                classified.append(classified_entity)
                
            except Exception as e:
                logger.warning(f"实体分类失败: {entity}, 错误: {str(e)}")
                # 添加错误标记
                classified.append({
                    **entity,
                    'subcategory': 'UNKNOWN',
                    'quality_assessment': {
                        'level': 'unreliable',
                        'confidence': 0.1,
                        'issues': [f'分类错误: {str(e)}'],
                        'suggestions': ['重新提取要素']
                    }
                })
        
        return classified
    
    def classify_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对关系进行分类"""
        classified = []
        
        for relation in relations:
            try:
                # 基本分类
                category = relation.get('category', 'unknown')
                subcategory = self._classify_relation_subcategory(relation, category)
                
                # 质量评估
                quality_result = self._assess_relation_quality(relation)
                
                # 构建结果
                classified_relation = {
                    **relation,
                    'subcategory': subcategory,
                    'quality_assessment': {
                        'level': quality_result.quality_level.value,
                        'confidence': quality_result.confidence_score,
                        'issues': quality_result.issues,
                        'suggestions': quality_result.suggestions
                    },
                    'classification_result': {
                        'category': category,
                        'subcategory': subcategory,
                        'confidence': quality_result.confidence_score,
                        'reasoning': self._generate_classification_reasoning(relation, category, subcategory)
                    }
                }
                
                classified.append(classified_relation)
                
            except Exception as e:
                logger.warning(f"关系分类失败: {relation}, 错误: {str(e)}")
                # 添加错误标记
                classified.append({
                    **relation,
                    'subcategory': 'UNKNOWN',
                    'quality_assessment': {
                        'level': 'unreliable',
                        'confidence': 0.1,
                        'issues': [f'分类错误: {str(e)}'],
                        'suggestions': ['重新提取要素']
                    }
                })
        
        return classified
    
    def validate_elements(self, entities: List[Dict], relations: List[Dict], 
                         events: List[Dict], attributes: List[Dict]) -> Dict[str, Any]:
        """验证要素质量"""
        validation_results = {
            'entity_validation': [],
            'relation_validation': [],
            'event_validation': [],
            'attribute_validation': [],
            'overall_quality': self._assess_overall_quality(entities, relations, events, attributes)
        }
        
        # 验证实体
        for entity in entities:
            entity_validation = self._validate_entity(entity)
            validation_results['entity_validation'].append(entity_validation)
        
        # 验证关系
        for relation in relations:
            relation_validation = self._validate_relation(relation, entities)
            validation_results['relation_validation'].append(relation_validation)
        
        # 验证事件和属性（简化处理）
        for event in events:
            validation_results['event_validation'].append({
                'id': event.get('id'),
                'is_valid': True,
                'confidence': event.get('confidence', 0.5),
                'issues': [],
                'suggestions': []
            })
        
        for attr in attributes:
            validation_results['attribute_validation'].append({
                'id': attr.get('id'),
                'is_valid': True,
                'confidence': attr.get('confidence', 0.5),
                'issues': [],
                'suggestions': []
            })
        
        return validation_results
    
    def _classify_entity_subcategory(self, entity: Dict[str, Any], category: str) -> str:
        """分类实体子类别"""
        text = entity.get('text', '')
        
        if category in self.entity_subcategories:
            for subcategory, patterns in self.entity_subcategories[category].items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        return subcategory
        
        return 'UNKNOWN'
    
    def _classify_relation_subcategory(self, relation: Dict[str, Any], category: str) -> str:
        """分类关系子类别"""
        relation_type = relation.get('relation_type', '')
        context = relation.get('context', '')
        text_to_analyze = f"{relation_type} {context}".lower()
        
        if category in self.relation_subcategories:
            for subcategory, patterns in self.relation_subcategories[category].items():
                for pattern in patterns:
                    if re.search(pattern, text_to_analyze):
                        return subcategory
        
        return 'UNKNOWN'
    
    def _assess_entity_quality(self, entity: Dict[str, Any]) -> ValidationResult:
        """评估实体质量"""
        issues = []
        suggestions = []
        confidence_factors = []
        
        # 检查基本属性
        text = entity.get('text', '')
        if len(text) == 0:
            issues.append("实体文本为空")
            suggestions.append("检查文本提取过程")
            confidence_factors.append(0.1)
        elif len(text) < 2:
            issues.append("实体文本过短")
            suggestions.append("考虑合并或扩展实体")
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.8)
        
        # 检查类别信息
        category = entity.get('category', 'MISC')
        if category == 'MISC':
            issues.append("实体类别未确定")
            suggestions.append("使用更精确的分类规则")
            confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.7)
        
        # 检查置信度
        original_confidence = entity.get('confidence', 0.0)
        confidence_factors.append(original_confidence)
        
        # 计算综合置信度
        final_confidence = np.mean(confidence_factors)
        
        # 确定质量等级
        quality_level = self._get_quality_level(final_confidence)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            quality_level=quality_level,
            confidence_score=final_confidence,
            issues=issues,
            suggestions=suggestions,
            evidence_strength=final_confidence
        )
    
    def _assess_relation_quality(self, relation: Dict[str, Any]) -> ValidationResult:
        """评估关系质量"""
        issues = []
        suggestions = []
        confidence_factors = []
        
        # 检查基本属性
        head_id = relation.get('head_entity_id', '')
        tail_id = relation.get('tail_entity_id', '')
        
        if not head_id or not tail_id:
            issues.append("关系参与者ID缺失")
            suggestions.append("检查实体提取和关系构建过程")
            confidence_factors.append(0.2)
        elif head_id == tail_id:
            issues.append("关系参与者相同（自反关系）")
            suggestions.append("验证实体识别的准确性")
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.7)
        
        # 检查关系类型
        relation_type = relation.get('relation_type', '')
        if not relation_type or relation_type == 'related_to':
            issues.append("关系类型过于泛化")
            suggestions.append("使用更具体的关系词汇")
            confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.6)
        
        # 检查置信度
        original_confidence = relation.get('confidence', 0.0)
        confidence_factors.append(original_confidence)
        
        # 计算综合置信度
        final_confidence = np.mean(confidence_factors)
        
        # 确定质量等级
        quality_level = self._get_quality_level(final_confidence)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            quality_level=quality_level,
            confidence_score=final_confidence,
            issues=issues,
            suggestions=suggestions,
            evidence_strength=final_confidence
        )
    
    def _validate_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """验证单个实体"""
        return {
            'id': entity.get('id'),
            'is_valid': True,  # 简化的验证逻辑
            'confidence': entity.get('confidence', 0.5),
            'issues': [],
            'suggestions': []
        }
    
    def _validate_relation(self, relation: Dict[str, Any], entities: List[Dict]) -> Dict[str, Any]:
        """验证单个关系"""
        head_id = relation.get('head_entity_id', '')
        tail_id = relation.get('tail_entity_id', '')
        
        # 检查参与者是否存在
        head_exists = any(e.get('id') == head_id for e in entities)
        tail_exists = any(e.get('id') == tail_id for e in entities)
        
        issues = []
        if not head_exists:
            issues.append("头实体不存在")
        if not tail_exists:
            issues.append("尾实体不存在")
        
        return {
            'id': relation.get('id'),
            'is_valid': len(issues) == 0,
            'confidence': relation.get('confidence', 0.5),
            'issues': issues,
            'suggestions': []
        }
    
    def _assess_overall_quality(self, entities: List[Dict], relations: List[Dict], 
                               events: List[Dict], attributes: List[Dict]) -> Dict[str, Any]:
        """评估整体质量"""
        all_elements = entities + relations + events + attributes
        
        if not all_elements:
            return {
                'overall_score': 0.0,
                'quality_level': 'unreliable',
                'coverage': 0.0,
                'consistency': 0.0
            }
        
        # 计算平均置信度
        confidences = [elem.get('confidence', 0.0) for elem in all_elements]
        avg_confidence = np.mean(confidences)
        
        # 计算覆盖率（基于类型分布）
        type_counts = Counter(elem.get('category', 'unknown') for elem in all_elements)
        total_types = len(type_counts)
        coverage = min(1.0, total_types / 5)  # 假设期望至少5种类型
        
        # 计算一致性（基于质量评估分布）
        high_quality = sum(1 for elem in all_elements 
                          if elem.get('quality_assessment', {}).get('level') == 'high')
        consistency = high_quality / len(all_elements) if all_elements else 0.0
        
        # 综合得分
        overall_score = (avg_confidence + coverage + consistency) / 3
        
        return {
            'overall_score': overall_score,
            'quality_level': self._get_quality_level(overall_score).value,
            'coverage': coverage,
            'consistency': consistency,
            'element_counts': {
                'entities': len(entities),
                'relations': len(relations),
                'events': len(events),
                'attributes': len(attributes)
            }
        }
    
    def _calculate_quality_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算质量指标"""
        metrics = {}
        
        for element_type, results in validation_results.items():
            if element_type == 'overall_quality':
                continue
                
            valid_count = sum(1 for result in results if result.get('is_valid', False))
            total_count = len(results)
            
            metrics[element_type] = {
                'total': total_count,
                'valid': valid_count,
                'validity_rate': valid_count / total_count if total_count > 0 else 0.0,
                'avg_confidence': np.mean([result.get('confidence', 0.0) for result in results]) if results else 0.0
            }
        
        return metrics
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于验证结果生成建议
        overall_quality = validation_results.get('overall_quality', {})
        quality_level = overall_quality.get('quality_level', 'unreliable')
        
        if quality_level in ['unreliable', 'low']:
            recommendations.append("建议重新检查AST解析和要素提取过程")
            recommendations.append("考虑优化伪代码生成质量")
        
        entity_validation = validation_results.get('entity_validation', [])
        if entity_validation:
            avg_confidence = np.mean([r.get('confidence', 0.0) for r in entity_validation])
            if avg_confidence < 0.6:
                recommendations.append("实体识别置信度较低，建议优化实体识别规则")
        
        relation_validation = validation_results.get('relation_validation', [])
        if relation_validation:
            invalid_count = sum(1 for r in relation_validation if not r.get('is_valid', True))
            if invalid_count > len(relation_validation) * 0.3:
                recommendations.append("关系验证失败率较高，建议检查实体匹配和关系构建")
        
        return recommendations
    
    def _generate_classification_reasoning(self, element: Dict, category: str, subcategory: str) -> str:
        """生成分类推理说明"""
        text = element.get('text', '')
        context = element.get('context', '')
        
        reasoning_parts = []
        
        if category != 'MISC':
            reasoning_parts.append(f"基于文本'{text}'的特征分类为{category}")
        
        if subcategory != 'UNKNOWN':
            reasoning_parts.append(f"进一步细分为{subcategory}子类别")
        
        if context:
            reasoning_parts.append(f"上下文信息: {context}")
        
        return "；".join(reasoning_parts) if reasoning_parts else "基于默认规则分类"
    
    def _get_quality_level(self, confidence: float) -> QualityLevel:
        """根据置信度获取质量等级"""
        if confidence >= self.quality_thresholds[QualityLevel.HIGH]:
            return QualityLevel.HIGH
        elif confidence >= self.quality_thresholds[QualityLevel.MEDIUM]:
            return QualityLevel.MEDIUM
        elif confidence >= self.quality_thresholds[QualityLevel.LOW]:
            return QualityLevel.LOW
        else:
            return QualityLevel.UNRELIABLE