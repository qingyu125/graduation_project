"""
验证器模块
验证提取要素的一致性、完整性和质量
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)


class ValidationError(Enum):
    """验证错误类型"""
    INCONSISTENT_ENTITIES = "inconsistent_entities"
    INVALID_RELATIONS = "invalid_relations"
    MISSING_EVIDENCE = "missing_evidence"
    DUPLICATE_ELEMENTS = "duplicate_elements"
    LOW_QUALITY = "low_quality"
    FORMAT_ERROR = "format_error"


@dataclass
class ValidationIssue:
    """验证问题"""
    error_type: ValidationError
    element_id: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str
    evidence: List[str]


@dataclass
class ValidationReport:
    """验证报告"""
    is_valid: bool
    overall_score: float
    issues: List[ValidationIssue]
    recommendations: List[str]
    statistics: Dict[str, Any]
    quality_indicators: Dict[str, float]


class ElementValidator:
    """要素验证器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 验证阈值
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.max_duplicate_threshold = self.config.get('max_duplicate_threshold', 0.8)
        self.min_evidence_threshold = self.config.get('min_evidence_threshold', 1)
        
        # 实体一致性规则
        self.entity_consistency_rules = {
            'PERSON': {
                'name_patterns': [r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'],
                'invalid_chars': [r'[0-9@#$%^&*()]'],
                'min_length': 2,
                'max_length': 50
            },
            'ORG': {
                'name_patterns': [r'^[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*$'],
                'invalid_chars': [r'[@#$%^&*()]'],
                'min_length': 2,
                'max_length': 100
            },
            'LOC': {
                'name_patterns': [r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|Street|Road))?$'],
                'invalid_chars': [r'[@#$%^&*()]'],
                'min_length': 2,
                'max_length': 100
            }
        }
        
        # 关系有效性规则
        self.relation_validity_rules = {
            'symmetric_relations': ['married_to', 'sibling_of', 'neighbor_of'],
            'transitive_relations': ['ancestor_of', 'part_of', 'subset_of'],
            'valid_patterns': {
                'PERSON': ['works_for', 'married_to', 'parent_of', 'friend_of'],
                'ORG': ['part_of', 'subsidiary_of', 'acquired_by'],
                'LOC': ['located_in', 'near', 'border']
            }
        }
    
    def validate_extraction_result(self, extraction_result: Dict[str, Any]) -> ValidationReport:
        """
        验证提取结果
        
        Args:
            extraction_result: 包含提取要素的字典
            
        Returns:
            ValidationReport: 验证报告
        """
        try:
            logger.info("开始验证提取结果")
            
            issues = []
            recommendations = []
            statistics = {}
            
            # 验证实体
            entity_issues = self._validate_entities(
                extraction_result.get('extracted_entities', [])
            )
            issues.extend(entity_issues)
            
            # 验证关系
            relation_issues = self._validate_relations(
                extraction_result.get('extracted_relations', []),
                extraction_result.get('extracted_entities', [])
            )
            issues.extend(relation_issues)
            
            # 验证事件和属性
            event_issues = self._validate_events(
                extraction_result.get('extracted_events', [])
            )
            issues.extend(event_issues)
            
            attribute_issues = self._validate_attributes(
                extraction_result.get('extracted_attributes', [])
            )
            issues.extend(attribute_issues)
            
            # 检查一致性
            consistency_issues = self._check_consistency(extraction_result)
            issues.extend(consistency_issues)
            
            # 生成统计信息
            statistics = self._generate_validation_statistics(issues, extraction_result)
            
            # 生成建议
            recommendations = self._generate_validation_recommendations(issues, statistics)
            
            # 计算总体评分
            overall_score = self._calculate_overall_score(issues, statistics)
            is_valid = overall_score >= self.min_confidence
            
            # 构建报告
            report = ValidationReport(
                is_valid=is_valid,
                overall_score=overall_score,
                issues=issues,
                recommendations=recommendations,
                statistics=statistics,
                quality_indicators=self._calculate_quality_indicators(extraction_result)
            )
            
            logger.info(f"验证完成，评分: {overall_score:.2f}, 发现 {len(issues)} 个问题")
            return report
            
        except Exception as e:
            logger.error(f"验证过程失败: {str(e)}")
            raise
    
    def _validate_entities(self, entities: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """验证实体"""
        issues = []
        
        for entity in entities:
            entity_issues = self._validate_single_entity(entity)
            issues.extend(entity_issues)
        
        # 检查重复实体
        duplicate_issues = self._check_duplicate_entities(entities)
        issues.extend(duplicate_issues)
        
        return issues
    
    def _validate_single_entity(self, entity: Dict[str, Any]) -> List[ValidationIssue]:
        """验证单个实体"""
        issues = []
        
        text = entity.get('text', '')
        category = entity.get('category', 'MISC')
        confidence = entity.get('confidence', 0.0)
        
        # 检查基本属性
        if not text or len(text.strip()) == 0:
            issues.append(ValidationIssue(
                error_type=ValidationError.FORMAT_ERROR,
                element_id=entity.get('id', ''),
                severity='error',
                message='实体文本为空',
                suggestion='检查文本提取过程',
                evidence=[]
            ))
        
        # 检查类别一致性
        if category in self.entity_consistency_rules:
            category_issues = self._check_entity_category_consistency(entity, category)
            issues.extend(category_issues)
        
        # 检查置信度
        if confidence < self.min_confidence:
            issues.append(ValidationIssue(
                error_type=ValidationError.LOW_QUALITY,
                element_id=entity.get('id', ''),
                severity='warning',
                message=f'实体置信度过低: {confidence:.2f}',
                suggestion='考虑重新提取或提高阈值',
                evidence=[f'置信度: {confidence}']
            ))
        
        return issues
    
    def _check_entity_category_consistency(self, entity: Dict[str, Any], category: str) -> List[ValidationIssue]:
        """检查实体类别一致性"""
        issues = []
        text = entity.get('text', '')
        rules = self.entity_consistency_rules[category]
        
        # 检查模式匹配
        valid_pattern = False
        for pattern in rules['name_patterns']:
            if re.match(pattern, text, re.IGNORECASE):
                valid_pattern = True
                break
        
        if not valid_pattern:
            issues.append(ValidationIssue(
                error_type=ValidationError.INCONSISTENT_ENTITIES,
                element_id=entity.get('id', ''),
                severity='warning',
                message=f'实体文本不符合{category}的命名模式',
                suggestion=f'检查文本格式: {text}',
                evidence=[f'模式: {rules["name_patterns"]}']
            ))
        
        # 检查无效字符
        for invalid_pattern in rules['invalid_chars']:
            if re.search(invalid_pattern, text):
                issues.append(ValidationIssue(
                    error_type=ValidationError.INCONSISTENT_ENTITIES,
                    element_id=entity.get('id', ''),
                    severity='warning',
                    message=f'实体文本包含无效字符: {text}',
                    suggestion='移除特殊字符或重新分类',
                    evidence=[f'匹配模式: {invalid_pattern}']
                ))
        
        # 检查长度
        if len(text) < rules['min_length']:
            issues.append(ValidationIssue(
                error_type=ValidationError.FORMAT_ERROR,
                element_id=entity.get('id', ''),
                severity='warning',
                message=f'实体文本过短: {len(text)} 字符',
                suggestion='检查文本提取是否完整',
                evidence=[f'长度: {len(text)}, 最小要求: {rules["min_length"]}']
            ))
        
        if len(text) > rules['max_length']:
            issues.append(ValidationIssue(
                error_type=ValidationError.FORMAT_ERROR,
                element_id=entity.get('id', ''),
                severity='info',
                message=f'实体文本过长: {len(text)} 字符',
                suggestion='考虑文本截断或重新分类',
                evidence=[f'长度: {len(text)}, 最大建议: {rules["max_length"]}']
            ))
        
        return issues
    
    def _check_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """检查重复实体"""
        issues = []
        text_counter = Counter()
        id_counter = Counter()
        
        # 统计文本和ID出现次数
        for entity in entities:
            text = entity.get('text', '').lower()
            entity_id = entity.get('id', '')
            
            text_counter[text] += 1
            id_counter[entity_id] += 1
        
        # 检查重复文本
        for text, count in text_counter.items():
            if count > 1:
                duplicate_entities = [e for e in entities if e.get('text', '').lower() == text]
                entity_ids = [e.get('id', '') for e in duplicate_entities]
                
                issues.append(ValidationIssue(
                    error_type=ValidationError.DUPLICATE_ELEMENTS,
                    element_id=entity_ids[0] if entity_ids else '',
                    severity='warning',
                    message=f'发现重复实体文本: "{text}" (出现{count}次)',
                    suggestion='合并重复实体或验证提取逻辑',
                    evidence=[f'实体ID: {entity_ids}']
                ))
        
        # 检查重复ID
        for entity_id, count in id_counter.items():
            if count > 1 and entity_id:
                issues.append(ValidationIssue(
                    error_type=ValidationError.DUPLICATE_ELEMENTS,
                    element_id=entity_id,
                    severity='error',
                    message=f'发现重复实体ID: {entity_id} (出现{count}次)',
                    suggestion='检查ID生成逻辑',
                    evidence=[]
                ))
        
        return issues
    
    def _validate_relations(self, relations: List[Dict[str, Any]], 
                           entities: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """验证关系"""
        issues = []
        
        # 构建实体映射
        entity_map = {entity.get('id'): entity for entity in entities}
        
        for relation in relations:
            relation_issues = self._validate_single_relation(relation, entity_map)
            issues.extend(relation_issues)
        
        # 检查关系一致性
        consistency_issues = self._check_relation_consistency(relations, entity_map)
        issues.extend(consistency_issues)
        
        return issues
    
    def _validate_single_relation(self, relation: Dict[str, Any], 
                                 entity_map: Dict[str, Dict]) -> List[ValidationIssue]:
        """验证单个关系"""
        issues = []
        
        head_id = relation.get('head_entity_id', '')
        tail_id = relation.get('tail_entity_id', '')
        relation_type = relation.get('relation_type', '')
        confidence = relation.get('confidence', 0.0)
        
        # 检查参与者存在性
        if head_id not in entity_map:
            issues.append(ValidationIssue(
                error_type=ValidationError.INVALID_RELATIONS,
                element_id=relation.get('id', ''),
                severity='error',
                message=f'头实体不存在: {head_id}',
                suggestion='检查实体ID映射',
                evidence=[]
            ))
        
        if tail_id not in entity_map:
            issues.append(ValidationIssue(
                error_type=ValidationError.INVALID_RELATIONS,
                element_id=relation.get('id', ''),
                severity='error',
                message=f'尾实体不存在: {tail_id}',
                suggestion='检查实体ID映射',
                evidence=[]
            ))
        
        # 检查自反关系
        if head_id == tail_id and head_id:
            issues.append(ValidationIssue(
                error_type=ValidationError.INVALID_RELATIONS,
                element_id=relation.get('id', ''),
                severity='warning',
                message='关系参与者相同（自反关系）',
                suggestion='验证关系是否真的为自反关系',
                evidence=[f'实体ID: {head_id}']
            ))
        
        # 检查关系类型有效性
        if not relation_type or relation_type == 'related_to':
            issues.append(ValidationIssue(
                error_type=ValidationError.INVALID_RELATIONS,
                element_id=relation.get('id', ''),
                severity='warning',
                message='关系类型过于泛化',
                suggestion='使用更具体的关系词汇',
                evidence=[f'关系类型: {relation_type}']
            ))
        
        # 检查置信度
        if confidence < self.min_confidence:
            issues.append(ValidationIssue(
                error_type=ValidationError.LOW_QUALITY,
                element_id=relation.get('id', ''),
                severity='warning',
                message=f'关系置信度过低: {confidence:.2f}',
                suggestion='考虑重新提取或提高阈值',
                evidence=[f'置信度: {confidence}']
            ))
        
        return issues
    
    def _check_relation_consistency(self, relations: List[Dict[str, Any]], 
                                   entity_map: Dict[str, Dict]) -> List[ValidationIssue]:
        """检查关系一致性"""
        issues = []
        
        # 检查对称关系
        for relation in relations:
            relation_type = relation.get('relation_type', '')
            head_id = relation.get('head_entity_id', '')
            tail_id = relation.get('tail_entity_id', '')
            
            if relation_type in self.relation_validity_rules['symmetric_relations']:
                # 查找对应的对称关系
                symmetric_relation = None
                for other_rel in relations:
                    if (other_rel.get('relation_type') == relation_type and
                        other_rel.get('head_entity_id') == tail_id and
                        other_rel.get('tail_entity_id') == head_id):
                        symmetric_relation = other_rel
                        break
                
                if not symmetric_relation:
                    issues.append(ValidationIssue(
                        error_type=ValidationError.INCONSISTENT_ENTITIES,
                        element_id=relation.get('id', ''),
                        severity='info',
                        message=f'缺少对称关系: {relation_type}({head_id}, {tail_id})',
                        suggestion=f'确认是否应该有对称关系 {relation_type}({tail_id}, {head_id})',
                        evidence=[]
                    ))
        
        return issues
    
    def _validate_events(self, events: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """验证事件"""
        issues = []
        
        for event in events:
            if not event.get('text', ''):
                issues.append(ValidationIssue(
                    error_type=ValidationError.FORMAT_ERROR,
                    element_id=event.get('id', ''),
                    severity='warning',
                    message='事件文本为空',
                    suggestion='检查事件提取过程',
                    evidence=[]
                ))
        
        return issues
    
    def _validate_attributes(self, attributes: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """验证属性"""
        issues = []
        
        for attribute in attributes:
            if not attribute.get('attribute', ''):
                issues.append(ValidationIssue(
                    error_type=ValidationError.FORMAT_ERROR,
                    element_id=attribute.get('id', ''),
                    severity='info',
                    message='属性值为空',
                    suggestion='检查属性提取过程',
                    evidence=[]
                ))
        
        return issues
    
    def _check_consistency(self, extraction_result: Dict[str, Any]) -> List[ValidationIssue]:
        """检查整体一致性"""
        issues = []
        
        entities = extraction_result.get('extracted_entities', [])
        relations = extraction_result.get('extracted_relations', [])
        
        # 构建实体映射
        entity_map = {entity.get('id'): entity for entity in entities}
        
        # 检查关系引用的完整性
        for relation in relations:
            head_id = relation.get('head_entity_id', '')
            tail_id = relation.get('tail_entity_id', '')
            
            if head_id not in entity_map:
                issues.append(ValidationIssue(
                    error_type=ValidationError.INCONSISTENT_ENTITIES,
                    element_id=relation.get('id', ''),
                    severity='error',
                    message=f'关系引用不存在的头实体: {head_id}',
                    suggestion='修复实体ID映射或移除无效关系',
                    evidence=[]
                ))
            
            if tail_id not in entity_map:
                issues.append(ValidationIssue(
                    error_type=ValidationError.INCONSISTENT_ENTITIES,
                    element_id=relation.get('id', ''),
                    severity='error',
                    message=f'关系引用不存在的尾实体: {tail_id}',
                    suggestion='修复实体ID映射或移除无效关系',
                    evidence=[]
                ))
        
        return issues
    
    def _generate_validation_statistics(self, issues: List[ValidationIssue], 
                                       extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证统计信息"""
        statistics = {
            'total_elements': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'error_types': defaultdict(int),
            'severity_distribution': defaultdict(int)
        }
        
        # 统计问题
        for issue in issues:
            statistics['severity_distribution'][issue.severity] += 1
            statistics['error_types'][issue.error_type.value] += 1
            
            if issue.severity == 'error':
                statistics['error_count'] += 1
            elif issue.severity == 'warning':
                statistics['warning_count'] += 1
            elif issue.severity == 'info':
                statistics['info_count'] += 1
        
        # 统计要素总数
        elements = (
            extraction_result.get('extracted_entities', []) +
            extraction_result.get('extracted_relations', []) +
            extraction_result.get('extracted_events', []) +
            extraction_result.get('extracted_attributes', [])
        )
        statistics['total_elements'] = len(elements)
        
        return dict(statistics)
    
    def _generate_validation_recommendations(self, issues: List[ValidationIssue], 
                                           statistics: Dict[str, Any]) -> List[str]:
        """生成验证建议"""
        recommendations = []
        
        error_count = statistics.get('error_count', 0)
        total_elements = statistics.get('total_elements', 0)
        
        if error_count > total_elements * 0.1:  # 超过10%的错误率
            recommendations.append("错误率较高，建议重新检查提取算法和参数设置")
        
        duplicate_count = sum(1 for issue in issues 
                             if issue.error_type == ValidationError.DUPLICATE_ELEMENTS)
        if duplicate_count > 0:
            recommendations.append("发现重复要素，建议添加去重逻辑")
        
        invalid_relation_count = sum(1 for issue in issues 
                                   if issue.error_type == ValidationError.INVALID_RELATIONS)
        if invalid_relation_count > 0:
            recommendations.append("发现无效关系，建议检查实体ID映射和关系构建逻辑")
        
        if not recommendations:
            recommendations.append("验证结果良好，当前提取质量可接受")
        
        return recommendations
    
    def _calculate_overall_score(self, issues: List[ValidationIssue], 
                                statistics: Dict[str, Any]) -> float:
        """计算总体评分"""
        total_elements = statistics.get('total_elements', 1)
        error_count = statistics.get('error_count', 0)
        warning_count = statistics.get('warning_count', 0)
        
        # 基础分数
        base_score = 1.0
        
        # 错误扣分（权重0.6）
        error_penalty = (error_count / total_elements) * 0.6
        
        # 警告扣分（权重0.3）
        warning_penalty = (warning_count / total_elements) * 0.3
        
        # 重复要素扣分（权重0.1）
        duplicate_issues = [issue for issue in issues 
                          if issue.error_type == ValidationError.DUPLICATE_ELEMENTS]
        duplicate_penalty = (len(duplicate_issues) / total_elements) * 0.1
        
        final_score = max(0.0, base_score - error_penalty - warning_penalty - duplicate_penalty)
        return final_score
    
    def _calculate_quality_indicators(self, extraction_result: Dict[str, Any]) -> Dict[str, float]:
        """计算质量指标"""
        indicators = {}
        
        entities = extraction_result.get('extracted_entities', [])
        relations = extraction_result.get('extracted_relations', [])
        
        # 实体质量指标
        if entities:
            entity_confidences = [entity.get('confidence', 0.0) for entity in entities]
            indicators['entity_avg_confidence'] = sum(entity_confidences) / len(entity_confidences)
            indicators['entity_high_confidence_ratio'] = sum(1 for c in entity_confidences if c >= 0.8) / len(entity_confidences)
        else:
            indicators['entity_avg_confidence'] = 0.0
            indicators['entity_high_confidence_ratio'] = 0.0
        
        # 关系质量指标
        if relations:
            relation_confidences = [rel.get('confidence', 0.0) for rel in relations]
            indicators['relation_avg_confidence'] = sum(relation_confidences) / len(relation_confidences)
            indicators['relation_high_confidence_ratio'] = sum(1 for c in relation_confidences if c >= 0.8) / len(relation_confidences)
        else:
            indicators['relation_avg_confidence'] = 0.0
            indicators['relation_high_confidence_ratio'] = 0.0
        
        return indicators