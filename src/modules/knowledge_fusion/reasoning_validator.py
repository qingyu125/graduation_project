"""
推理结果验证和一致性检查模块
实现逻辑一致性验证、事实正确性检查和推理质量评估功能
支持多种验证策略和质量度量方法
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import Counter, defaultdict
import re
from datetime import datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """验证级别枚举"""
    BASIC = "basic"        # 基础验证
    STRICT = "strict"      # 严格验证
    COMPREHENSIVE = "comprehensive"  # 全面验证
    RIGOROUS = "rigorous"  # 严谨验证

class IssueSeverity(Enum):
    """问题严重性等级"""
    CRITICAL = "critical"  # 严重问题
    ERROR = "error"        # 错误
    WARNING = "warning"    # 警告
    INFO = "info"          # 信息

class ConsistencyType(Enum):
    """一致性类型"""
    LOGICAL = "logical"        # 逻辑一致性
    FACTUAL = "factual"        # 事实一致性
    TEMPORAL = "temporal"      # 时间一致性
    SEMANTIC = "semantic"      # 语义一致性
    STRUCTURAL = "structural"  # 结构一致性

@dataclass
class ValidationRule:
    """验证规则数据结构"""
    rule_id: str
    rule_name: str
    description: str
    validation_type: ConsistencyType
    severity: IssueSeverity
    is_active: bool = True
    parameters: Dict[str, Any] = None

@dataclass
class ValidationIssue:
    """验证问题数据结构"""
    issue_id: str
    rule_id: str
    severity: IssueSeverity
    issue_type: ConsistencyType
    description: str
    details: Dict[str, Any]
    affected_elements: List[str]
    suggested_resolution: str
    confidence: float
    timestamp: str

@dataclass
class ValidationResult:
    """验证结果数据结构"""
    validation_id: str
    overall_score: float
    is_valid: bool
    validation_level: ValidationLevel
    issues: List[ValidationIssue]
    metrics: Dict[str, Any]
    recommendations: List[str]
    validation_details: Dict[str, Any]

class LogicalConsistencyChecker:
    """逻辑一致性检查器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化逻辑一致性检查器"""
        self.config = config or {}
        self.rules = self._initialize_logical_rules()
        
        # 预设的逻辑冲突模式
        self.conflict_patterns = {
            'entity_contradiction': r'同一人不能既.*?又.*?',
            'relation_cycle': r'关系循环',
            'type_mismatch': r'类型不匹配',
            'conflicting_evidence': r'相互矛盾的证据'
        }
        
        logger.info("逻辑一致性检查器初始化完成")
    
    def check_logical_consistency(self, entities: List[Dict], relations: List[Dict]) -> List[ValidationIssue]:
        """检查逻辑一致性"""
        try:
            logger.info("开始逻辑一致性检查")
            issues = []
            
            # 1. 检查实体类型一致性
            entity_type_issues = self._check_entity_type_consistency(entities)
            issues.extend(entity_type_issues)
            
            # 2. 检查关系逻辑一致性
            relation_issues = self._check_relation_consistency(relations, entities)
            issues.extend(relation_issues)
            
            # 3. 检查循环依赖
            cycle_issues = self._check_circular_dependencies(relations)
            issues.extend(cycle_issues)
            
            # 4. 检查逻辑冲突
            conflict_issues = self._check_logical_conflicts(entities, relations)
            issues.extend(conflict_issues)
            
            logger.info(f"逻辑一致性检查完成: 发现 {len(issues)} 个问题")
            return issues
            
        except Exception as e:
            logger.error(f"逻辑一致性检查失败: {e}")
            return []
    
    def _check_entity_type_consistency(self, entities: List[Dict]) -> List[ValidationIssue]:
        """检查实体类型一致性"""
        issues = []
        
        # 统计每个名称对应的类型
        name_types = defaultdict(set)
        for entity in entities:
            name = entity.get('name', '')
            entity_type = entity.get('type', '')
            if name and entity_type:
                name_types[name].add(entity_type)
        
        # 检查类型冲突
        for name, types in name_types.items():
            if len(types) > 1:
                issue = ValidationIssue(
                    issue_id=f"type_conflict_{hashlib.md5(name.encode()).hexdigest()[:8]}",
                    rule_id="entity_type_consistency",
                    severity=IssueSeverity.ERROR,
                    issue_type=ConsistencyType.LOGICAL,
                    description=f"实体 '{name}' 有冲突的类型: {list(types)}",
                    details={
                        'entity_name': name,
                        'conflicting_types': list(types),
                        'entity_count': len(types)
                    },
                    affected_elements=[name],
                    suggested_resolution="统一实体类型或创建不同的实体实例",
                    confidence=0.9,
                    timestamp=datetime.now().isoformat()
                )
                issues.append(issue)
        
        return issues
    
    def _check_relation_consistency(self, relations: List[Dict], entities: List[Dict]) -> List[ValidationIssue]:
        """检查关系逻辑一致性"""
        issues = []
        
        # 构建实体索引
        entity_index = {entity.get('name', ''): entity.get('type', '') for entity in entities}
        
        # 检查关系是否符合实体类型约束
        for relation in relations:
            source_name = relation.get('source_name', '')
            target_name = relation.get('target_name', '')
            relation_type = relation.get('relation_type', '')
            
            source_type = entity_index.get(source_name, '')
            target_type = entity_index.get(target_name, '')
            
            # 检查常见的关系类型约束
            if relation_type in ['spouse_of', 'parent_of', 'child_of'] and source_type == target_type:
                issue = ValidationIssue(
                    issue_id=f"relation_type_mismatch_{hashlib.md5(f'{source_name}_{target_name}'.encode()).hexdigest()[:8]}",
                    rule_id="relation_type_constraint",
                    severity=IssueSeverity.WARNING,
                    issue_type=ConsistencyType.LOGICAL,
                    description=f"关系 '{source_name} -> {target_name}' 类型不匹配: {relation_type} 需要不同类型的实体",
                    details={
                        'source_name': source_name,
                        'target_name': target_name,
                        'relation_type': relation_type,
                        'source_type': source_type,
                        'target_type': target_type
                    },
                    affected_elements=[source_name, target_name],
                    suggested_resolution="检查关系类型是否正确或实体类型是否需要调整",
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                )
                issues.append(issue)
        
        return issues
    
    def _check_circular_dependencies(self, relations: List[Dict]) -> List[ValidationIssue]:
        """检查循环依赖"""
        issues = []
        
        # 构建关系图
        graph = defaultdict(list)
        for relation in relations:
            source = relation.get('source_name', '')
            target = relation.get('target_name', '')
            if source and target:
                graph[source].append(target)
        
        # 检测简单循环
        def has_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack, path):
                        return True
                elif neighbor in rec_stack:
                    # 发现循环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    return cycle
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        visited = set()
        rec_stack = set()
        path = []
        
        for node in graph:
            if node not in visited:
                cycle = has_cycle(node, visited, rec_stack, path)
                if cycle:
                    issue = ValidationIssue(
                        issue_id=f"circular_dependency_{hashlib.md5(''.join(cycle).encode()).hexdigest()[:8]}",
                        rule_id="circular_dependency",
                        severity=IssueSeverity.WARNING,
                        issue_type=ConsistencyType.LOGICAL,
                        description=f"发现关系循环: {' -> '.join(cycle)}",
                        details={
                            'cycle_elements': cycle,
                            'cycle_length': len(cycle)
                        },
                        affected_elements=cycle,
                        suggested_resolution="检查循环中的关系是否正确，可能需要删除或调整某些关系",
                        confidence=0.9,
                        timestamp=datetime.now().isoformat()
                    )
                    issues.append(issue)
        
        return issues
    
    def _check_logical_conflicts(self, entities: List[Dict], relations: List[Dict]) -> List[ValidationIssue]:
        """检查逻辑冲突"""
        issues = []
        
        # 检查相互排斥的关系
        conflicting_pairs = [
            ('located_in', 'not_located_in'),
            ('is_part_of', 'not_part_of'),
            ('same_as', 'different_from')
        ]
        
        # 统计每种关系的存在情况
        relation_pairs = defaultdict(set)
        for relation in relations:
            source = relation.get('source_name', '')
            target = relation.get('target_name', '')
            relation_type = relation.get('relation_type', '')
            if source and target and relation_type:
                key = (source, target)
                relation_pairs[key].add(relation_type)
        
        # 检查冲突关系
        for (source, target), types in relation_pairs.items():
            for rel1, rel2 in conflicting_pairs:
                if rel1 in types and rel2 in types:
                    issue = ValidationIssue(
                        issue_id=f"conflicting_relations_{hashlib.md5(f'{source}_{target}'.encode()).hexdigest()[:8]}",
                        rule_id="conflicting_relations",
                        severity=IssueSeverity.CRITICAL,
                        issue_type=ConsistencyType.LOGICAL,
                        description=f"实体 '{source}' 和 '{target}' 之间存在相互冲突的关系: {rel1} 和 {rel2}",
                        details={
                            'source': source,
                            'target': target,
                            'conflicting_relations': [rel1, rel2],
                            'all_relations': list(types)
                        },
                        affected_elements=[source, target],
                        suggested_resolution="检查两个关系中哪一个是正确的，删除冲突的关系",
                        confidence=0.95,
                        timestamp=datetime.now().isoformat()
                    )
                    issues.append(issue)
        
        return issues
    
    def _initialize_logical_rules(self) -> List[ValidationRule]:
        """初始化逻辑验证规则"""
        return [
            ValidationRule(
                rule_id="entity_type_consistency",
                rule_name="实体类型一致性",
                description="检查同名实体是否具有一致的类型标签",
                validation_type=ConsistencyType.LOGICAL,
                severity=IssueSeverity.ERROR
            ),
            ValidationRule(
                rule_id="relation_type_constraint",
                rule_name="关系类型约束",
                description="检查关系是否满足实体类型约束",
                validation_type=ConsistencyType.LOGICAL,
                severity=IssueSeverity.WARNING
            ),
            ValidationRule(
                rule_id="circular_dependency",
                rule_name="循环依赖检查",
                description="检测关系图中的循环依赖",
                validation_type=ConsistencyType.LOGICAL,
                severity=IssueSeverity.WARNING
            ),
            ValidationRule(
                rule_id="conflicting_relations",
                rule_name="冲突关系检查",
                description="检测相互矛盾的关系",
                validation_type=ConsistencyType.LOGICAL,
                severity=IssueSeverity.CRITICAL
            )
        ]

class FactualConsistencyChecker:
    """事实一致性检查器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化事实一致性检查器"""
        self.config = config or {}
        self.fact_rules = self._initialize_factual_rules()
        
        # 常见的事实冲突模式
        self.fact_conflicts = {
            'temporal_conflict': r'时间信息冲突',
            'numeric_conflict': r'数值信息冲突',
            'attribute_conflict': r'属性值冲突'
        }
        
        logger.info("事实一致性检查器初始化完成")
    
    def check_factual_consistency(self, entities: List[Dict], relations: List[Dict], 
                                knowledge_graph: Any = None) -> List[ValidationIssue]:
        """检查事实一致性"""
        try:
            logger.info("开始事实一致性检查")
            issues = []
            
            # 1. 检查数值一致性
            numeric_issues = self._check_numeric_consistency(entities)
            issues.extend(numeric_issues)
            
            # 2. 检查时间一致性
            temporal_issues = self._check_temporal_consistency(entities, relations)
            issues.extend(temporal_issues)
            
            # 3. 检查属性一致性
            attribute_issues = self._check_attribute_consistency(entities)
            issues.extend(attribute_issues)
            
            # 4. 使用知识图谱验证（如果有）
            if knowledge_graph:
                kg_issues = self._check_against_knowledge_graph(entities, relations, knowledge_graph)
                issues.extend(kg_issues)
            
            logger.info(f"事实一致性检查完成: 发现 {len(issues)} 个问题")
            return issues
            
        except Exception as e:
            logger.error(f"事实一致性检查失败: {e}")
            return []
    
    def _check_numeric_consistency(self, entities: List[Dict]) -> List[ValidationIssue]:
        """检查数值一致性"""
        issues = []
        
        # 收集数值属性
        numeric_attributes = defaultdict(list)
        for entity in entities:
            for key, value in entity.get('properties', {}).items():
                if isinstance(value, (int, float)) and key in ['age', 'count', 'number', 'value']:
                    numeric_attributes[key].append((entity.get('name', ''), value))
        
        # 检查数值冲突
        for attr_name, values in numeric_attributes.items():
            if len(values) > 1:
                names = [v[0] for v in values]
                nums = [v[1] for v in values]
                
                # 检查是否表示相同实体但数值不同
                if len(set(names)) == 1:  # 同一个实体
                    if max(nums) - min(nums) > 0.1:  # 数值差异超过阈值
                        issue = ValidationIssue(
                            issue_id=f"numeric_conflict_{hashlib.md5(attr_name.encode()).hexdigest()[:8]}",
                            rule_id="numeric_consistency",
                            severity=IssueSeverity.ERROR,
                            issue_type=ConsistencyType.FACTUAL,
                            description=f"实体 '{names[0]}' 的属性 '{attr_name}' 有冲突的数值: {nums}",
                            details={
                                'entity_name': names[0],
                                'attribute': attr_name,
                                'conflicting_values': nums,
                                'max_difference': max(nums) - min(nums)
                            },
                            affected_elements=names,
                            suggested_resolution="统一数值或确认哪个是正确的值",
                            confidence=0.8,
                            timestamp=datetime.now().isoformat()
                        )
                        issues.append(issue)
        
        return issues
    
    def _check_temporal_consistency(self, entities: List[Dict], relations: List[Dict]) -> List[ValidationIssue]:
        """检查时间一致性"""
        issues = []
        
        # 提取时间信息
        temporal_info = self._extract_temporal_info(entities, relations)
        
        # 检查时间冲突
        for source, targets in temporal_info.items():
            for target, time_data in targets.items():
                if 'birth_date' in time_data and 'death_date' in time_data:
                    birth = time_data['birth_date']
                    death = time_data['death_date']
                    
                    if self._is_date_after(birth, death):
                        issue = ValidationIssue(
                            issue_id=f"temporal_conflict_{hashlib.md5(f'{source}_{target}'.encode()).hexdigest()[:8]}",
                            rule_id="temporal_consistency",
                            severity=IssueSeverity.ERROR,
                            issue_type=ConsistencyType.TEMPORAL,
                            description=f"时间冲突: {source} 的出生日期 {birth} 晚于死亡日期 {death}",
                            details={
                                'entity': source,
                                'birth_date': birth,
                                'death_date': death
                            },
                            affected_elements=[source],
                            suggested_resolution="检查日期是否正确或是否表示不同的事件",
                            confidence=0.9,
                            timestamp=datetime.now().isoformat()
                        )
                        issues.append(issue)
        
        return issues
    
    def _extract_temporal_info(self, entities: List[Dict], relations: List[Dict]) -> Dict:
        """提取时间信息"""
        temporal_info = defaultdict(lambda: defaultdict(dict))
        
        # 从实体中提取时间信息
        for entity in entities:
            name = entity.get('name', '')
            for key, value in entity.get('properties', {}).items():
                if 'date' in key.lower() or 'time' in key.lower():
                    temporal_info[name][key] = value
        
        # 从关系中提取时间信息
        for relation in relations:
            source = relation.get('source_name', '')
            target = relation.get('target_name', '')
            if 'date' in relation.get('context', {}):
                temporal_info[source][f"relation_date_{relation.get('relation_type', '')}"] = relation['context']['date']
                temporal_info[target][f"relation_date_{relation.get('relation_type', '')}"] = relation['context']['date']
        
        return temporal_info
    
    def _is_date_after(self, date1: str, date2: str) -> bool:
        """检查date1是否在date2之后（简化版）"""
        try:
            # 简化的日期比较，实际应用中应该使用更好的日期处理
            return date1 > date2
        except:
            return False
    
    def _check_attribute_consistency(self, entities: List[Dict]) -> List[ValidationIssue]:
        """检查属性一致性"""
        issues = []
        
        # 收集属性值
        attribute_values = defaultdict(dict)
        for entity in entities:
            entity_name = entity.get('name', '')
            for key, value in entity.get('properties', {}).items():
                if key not in attribute_values[entity_name]:
                    attribute_values[entity_name][key] = []
                attribute_values[entity_name][key].append(value)
        
        # 检查属性冲突
        for entity_name, attributes in attribute_values.items():
            for attr_name, values in attributes.items():
                if len(set(str(v) for v in values)) > 1:  # 有不同的值
                    # 如果是枚举类型，检查是否是合理的变化
                    if not self._is_reasonable_attribute_change(attr_name, values):
                        issue = ValidationIssue(
                            issue_id=f"attribute_conflict_{hashlib.md5(f'{entity_name}_{attr_name}'.encode()).hexdigest()[:8]}",
                            rule_id="attribute_consistency",
                            severity=IssueSeverity.WARNING,
                            issue_type=ConsistencyType.FACTUAL,
                            description=f"实体 '{entity_name}' 的属性 '{attr_name}' 有冲突的值: {values}",
                            details={
                                'entity_name': entity_name,
                                'attribute': attr_name,
                                'conflicting_values': values,
                                'value_count': len(values)
                            },
                            affected_elements=[entity_name],
                            suggested_resolution="确认属性值是否表示不同时间点的状态",
                            confidence=0.7,
                            timestamp=datetime.now().isoformat()
                        )
                        issues.append(issue)
        
        return issues
    
    def _is_reasonable_attribute_change(self, attr_name: str, values: List) -> bool:
        """检查属性值的变化是否合理"""
        # 一些属性的值可以随时间变化
        mutable_attributes = ['status', 'position', 'location', 'state']
        
        if any(mutable in attr_name.lower() for mutable in mutable_attributes):
            return True
        
        # 其他属性值通常应该是稳定的
        return False
    
    def _check_against_knowledge_graph(self, entities: List[Dict], relations: List[Dict], 
                                     knowledge_graph: Any) -> List[ValidationIssue]:
        """使用知识图谱验证事实"""
        issues = []
        
        # 验证实体
        for entity in entities:
            entity_name = entity.get('name', '')
            entity_type = entity.get('type', '')
            
            # 查询知识图谱中的匹配实体
            matches = knowledge_graph.query_entities(entity_name, entity_type)
            
            if matches:
                # 检查类型一致性
                for match in matches:
                    if match.entity_type != entity_type:
                        issue = ValidationIssue(
                            issue_id=f"kg_type_mismatch_{hashlib.md5(entity_name.encode()).hexdigest()[:8]}",
                            rule_id="knowledge_graph_consistency",
                            severity=IssueSeverity.WARNING,
                            issue_type=ConsistencyType.SEMANTIC,
                            description=f"实体 '{entity_name}' 在知识图谱中的类型 ({match.entity_type}) 与输入类型 ({entity_type}) 不一致",
                            details={
                                'entity_name': entity_name,
                                'input_type': entity_type,
                                'kg_type': match.entity_type,
                                'kg_entity': asdict(match)
                            },
                            affected_elements=[entity_name],
                            suggested_resolution="确认类型标签是否正确或知识图谱信息是否需要更新",
                            confidence=0.6,
                            timestamp=datetime.now().isoformat()
                        )
                        issues.append(issue)
        
        return issues
    
    def _initialize_factual_rules(self) -> List[ValidationRule]:
        """初始化事实验证规则"""
        return [
            ValidationRule(
                rule_id="numeric_consistency",
                rule_name="数值一致性",
                description="检查数值属性的冲突",
                validation_type=ConsistencyType.FACTUAL,
                severity=IssueSeverity.ERROR
            ),
            ValidationRule(
                rule_id="temporal_consistency",
                rule_name="时间一致性",
                description="检查时间信息的逻辑冲突",
                validation_type=ConsistencyType.TEMPORAL,
                severity=IssueSeverity.ERROR
            ),
            ValidationRule(
                rule_id="attribute_consistency",
                rule_name="属性一致性",
                description="检查属性值的冲突",
                validation_type=ConsistencyType.FACTUAL,
                severity=IssueSeverity.WARNING
            ),
            ValidationRule(
                rule_id="knowledge_graph_consistency",
                rule_name="知识图谱一致性",
                description="与外部知识图谱的一致性检查",
                validation_type=ConsistencyType.SEMANTIC,
                severity=IssueSeverity.WARNING
            )
        ]

class ReasoningQualityEvaluator:
    """推理质量评估器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化推理质量评估器"""
        self.config = config or {}
        self.quality_metrics = self._initialize_quality_metrics()
        
        logger.info("推理质量评估器初始化完成")
    
    def evaluate_reasoning_quality(self, fusion_result: Dict, 
                                 validation_issues: List[ValidationIssue]) -> Dict[str, Any]:
        """评估推理质量"""
        try:
            logger.info("开始推理质量评估")
            
            # 计算各个质量指标
            quality_scores = {}
            
            # 1. 准确性得分
            accuracy_score = self._calculate_accuracy_score(validation_issues)
            quality_scores['accuracy'] = accuracy_score
            
            # 2. 一致性得分
            consistency_score = self._calculate_consistency_score(validation_issues)
            quality_scores['consistency'] = consistency_score
            
            # 3. 完整性得分
            completeness_score = self._calculate_completeness_score(fusion_result)
            quality_scores['completeness'] = completeness_score
            
            # 4. 置信度得分
            confidence_score = self._calculate_confidence_score(fusion_result)
            quality_scores['confidence'] = confidence_score
            
            # 5. 综合质量得分
            overall_score = self._calculate_overall_score(quality_scores)
            quality_scores['overall'] = overall_score
            
            # 生成质量报告
            quality_report = {
                'quality_scores': quality_scores,
                'quality_level': self._determine_quality_level(overall_score),
                'strengths': self._identify_strengths(quality_scores),
                'weaknesses': self._identify_weaknesses(quality_scores, validation_issues),
                'recommendations': self._generate_recommendations(quality_scores, validation_issues),
                'validation_issues_summary': self._summarize_issues(validation_issues)
            }
            
            logger.info(f"推理质量评估完成: 总体得分 = {overall_score:.3f}")
            return quality_report
            
        except Exception as e:
            logger.error(f"推理质量评估失败: {e}")
            return {'error': str(e)}
    
    def _calculate_accuracy_score(self, validation_issues: List[ValidationIssue]) -> float:
        """计算准确性得分"""
        if not validation_issues:
            return 1.0
        
        # 根据问题严重性计算准确性得分
        severity_weights = {
            IssueSeverity.CRITICAL: 0.3,
            IssueSeverity.ERROR: 0.2,
            IssueSeverity.WARNING: 0.1,
            IssueSeverity.INFO: 0.05
        }
        
        total_penalty = 0.0
        for issue in validation_issues:
            weight = severity_weights.get(issue.severity, 0.1)
            total_penalty += weight
        
        return max(0.0, 1.0 - total_penalty)
    
    def _calculate_consistency_score(self, validation_issues: List[ValidationIssue]) -> float:
        """计算一致性得分"""
        logical_issues = [issue for issue in validation_issues if issue.issue_type == ConsistencyType.LOGICAL]
        factual_issues = [issue for issue in validation_issues if issue.issue_type == ConsistencyType.FACTUAL]
        
        if not logical_issues and not factual_issues:
            return 1.0
        
        # 逻辑问题权重更高
        logical_penalty = len(logical_issues) * 0.2
        factual_penalty = len(factual_issues) * 0.1
        
        return max(0.0, 1.0 - logical_penalty - factual_penalty)
    
    def _calculate_completeness_score(self, fusion_result: Dict) -> float:
        """计算完整性得分"""
        entity_fusion = fusion_result.get('entity_fusion', {})
        relation_fusion = fusion_result.get('relation_fusion', {})
        
        # 基础数据完整度
        entity_stats = entity_fusion.get('fusion_statistics', {})
        relation_stats = relation_fusion.get('fusion_statistics', {})
        
        original_entities = entity_stats.get('original_count', 1)
        fused_entities = entity_stats.get('fused_count', 1)
        original_relations = relation_stats.get('original_count', 1)
        fused_relations = relation_stats.get('fused_count', 1)
        
        # 计算完整度
        entity_coverage = fused_entities / max(original_entities, 1)
        relation_coverage = fused_relations / max(original_relations, 1)
        
        return (entity_coverage + relation_coverage) / 2
    
    def _calculate_confidence_score(self, fusion_result: Dict) -> float:
        """计算置信度得分"""
        quality = fusion_result.get('overall_quality', {})
        return quality.get('overall_quality', 0.0)
    
    def _calculate_overall_score(self, quality_scores: Dict[str, float]) -> float:
        """计算综合质量得分"""
        # 加权平均
        weights = {
            'accuracy': 0.3,
            'consistency': 0.25,
            'completeness': 0.25,
            'confidence': 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_scores:
                weighted_sum += quality_scores[metric] * weight
                total_weight += weight
        
        return weighted_sum / max(total_weight, 1.0)
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """确定质量等级"""
        if overall_score >= 0.9:
            return "优秀"
        elif overall_score >= 0.8:
            return "良好"
        elif overall_score >= 0.7:
            return "中等"
        elif overall_score >= 0.6:
            return "一般"
        else:
            return "较差"
    
    def _identify_strengths(self, quality_scores: Dict[str, float]) -> List[str]:
        """识别优势"""
        strengths = []
        
        for metric, score in quality_scores.items():
            if score >= 0.8:
                if metric == 'accuracy':
                    strengths.append("推理准确性高")
                elif metric == 'consistency':
                    strengths.append("逻辑一致性良好")
                elif metric == 'completeness':
                    strengths.append("信息完整性高")
                elif metric == 'confidence':
                    strengths.append("置信度评估可靠")
        
        return strengths
    
    def _identify_weaknesses(self, quality_scores: Dict[str, float], 
                           validation_issues: List[ValidationIssue]) -> List[str]:
        """识别弱点"""
        weaknesses = []
        
        # 质量分数低于0.7的指标
        for metric, score in quality_scores.items():
            if score < 0.7:
                if metric == 'accuracy':
                    weaknesses.append("推理准确性需要改进")
                elif metric == 'consistency':
                    weaknesses.append("逻辑一致性存在问题")
                elif metric == 'completeness':
                    weaknesses.append("信息完整性不足")
                elif metric == 'confidence':
                    weaknesses.append("置信度评估不可靠")
        
        # 根据验证问题识别的弱点
        issue_count = len(validation_issues)
        if issue_count > 10:
            weaknesses.append("验证问题数量过多")
        elif issue_count > 5:
            weaknesses.append("存在较多验证问题")
        
        critical_issues = [issue for issue in validation_issues if issue.severity == IssueSeverity.CRITICAL]
        if critical_issues:
            weaknesses.append("存在严重错误需要修复")
        
        return weaknesses
    
    def _generate_recommendations(self, quality_scores: Dict[str, float], 
                                validation_issues: List[ValidationIssue]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 根据质量分数提供建议
        for metric, score in quality_scores.items():
            if score < 0.6:
                if metric == 'accuracy':
                    recommendations.append("提高推理算法的准确性，减少错误推断")
                elif metric == 'consistency':
                    recommendations.append("加强逻辑约束检查，确保推理结果的一致性")
                elif metric == 'completeness':
                    recommendations.append("增加信息提取的完整性，避免遗漏重要事实")
                elif metric == 'confidence':
                    recommendations.append("改进置信度评估方法，提供更可靠的不确定性度量")
        
        # 根据验证问题提供建议
        issue_types = Counter(issue.issue_type for issue in validation_issues)
        if issue_types[ConsistencyType.LOGICAL] > 3:
            recommendations.append("重点解决逻辑一致性问题，检查推理规则的正确性")
        
        if issue_types[ConsistencyType.FACTUAL] > 3:
            recommendations.append("加强事实验证，使用外部知识源进行交叉验证")
        
        if issue_types[ConsistencyType.TEMPORAL] > 2:
            recommendations.append("改进时间推理机制，确保时间信息的一致性")
        
        return recommendations
    
    def _summarize_issues(self, validation_issues: List[ValidationIssue]) -> Dict[str, Any]:
        """总结验证问题"""
        if not validation_issues:
            return {'total_issues': 0}
        
        # 按严重性统计
        severity_counts = Counter(issue.severity for issue in validation_issues)
        
        # 按类型统计
        type_counts = Counter(issue.issue_type for issue in validation_issues)
        
        # 最严重的问题
        most_critical = max(validation_issues, key=lambda x: x.severity.value) if validation_issues else None
        
        return {
            'total_issues': len(validation_issues),
            'severity_distribution': dict(severity_counts),
            'type_distribution': dict(type_counts),
            'most_critical_issue': {
                'description': most_critical.description if most_critical else None,
                'severity': most_critical.severity.value if most_critical else None
            } if most_critical else None
        }
    
    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """初始化质量指标"""
        return {
            'accuracy': {
                'weight': 0.3,
                'description': '推理的准确性',
                'threshold': 0.7
            },
            'consistency': {
                'weight': 0.25,
                'description': '逻辑一致性',
                'threshold': 0.8
            },
            'completeness': {
                'weight': 0.25,
                'description': '信息完整性',
                'threshold': 0.75
            },
            'confidence': {
                'weight': 0.2,
                'description': '置信度可靠性',
                'threshold': 0.6
            }
        }

class ReasoningValidator:
    """推理结果验证器主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化推理验证器"""
        self.config = config or {}
        
        # 初始化各个检查器
        self.logical_checker = LogicalConsistencyChecker(self.config.get('logical', {}))
        self.factual_checker = FactualConsistencyChecker(self.config.get('factual', {}))
        self.quality_evaluator = ReasoningQualityEvaluator(self.config.get('quality', {}))
        
        # 验证配置
        self.validation_level = ValidationLevel(self.config.get('validation_level', 'comprehensive'))
        self.max_issues = self.config.get('max_issues', 100)
        self.enable_quality_evaluation = self.config.get('enable_quality_evaluation', True)
        
        logger.info("推理验证器初始化完成")
    
    def validate_reasoning_result(self, input_data: Dict[str, Any], 
                                fusion_result: Dict[str, Any],
                                knowledge_graph: Any = None) -> ValidationResult:
        """验证推理结果"""
        try:
            logger.info("开始验证推理结果")
            validation_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 提取输入数据
            entities = input_data.get('entities', [])
            relations = input_data.get('relations', [])
            
            # 执行各类验证
            all_issues = []
            
            # 1. 逻辑一致性检查
            logical_issues = self.logical_checker.check_logical_consistency(entities, relations)
            all_issues.extend(logical_issues)
            
            # 2. 事实一致性检查
            factual_issues = self.factual_checker.check_factual_consistency(
                entities, relations, knowledge_graph)
            all_issues.extend(factual_issues)
            
            # 限制问题数量
            if len(all_issues) > self.max_issues:
                all_issues = all_issues[:self.max_issues]
                logger.warning(f"验证问题数量超过限制，只保留前 {self.max_issues} 个问题")
            
            # 计算总体得分
            overall_score = self._calculate_overall_score(all_issues)
            
            # 判断是否有效
            critical_issues = [issue for issue in all_issues if issue.severity == IssueSeverity.CRITICAL]
            is_valid = len(critical_issues) == 0 and overall_score >= 0.6
            
            # 质量评估
            quality_report = {}
            if self.enable_quality_evaluation:
                quality_report = self.quality_evaluator.evaluate_reasoning_quality(
                    fusion_result, all_issues)
            
            # 生成建议
            recommendations = self._generate_validation_recommendations(all_issues, quality_report)
            
            validation_result = ValidationResult(
                validation_id=validation_id,
                overall_score=overall_score,
                is_valid=is_valid,
                validation_level=self.validation_level,
                issues=all_issues,
                metrics=self._calculate_validation_metrics(all_issues, fusion_result),
                recommendations=recommendations,
                validation_details={
                    'validation_level': self.validation_level.value,
                    'issues_processed': len(all_issues),
                    'knowledge_graph_used': knowledge_graph is not None,
                    'quality_evaluation': quality_report
                }
            )
            
            logger.info(f"推理结果验证完成: 得分={overall_score:.3f}, 有效={is_valid}")
            return validation_result
            
        except Exception as e:
            logger.error(f"推理结果验证失败: {e}")
            return ValidationResult(
                validation_id=f"validation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                overall_score=0.0,
                is_valid=False,
                validation_level=self.validation_level,
                issues=[],
                metrics={},
                recommendations=[f"验证过程发生错误: {str(e)}"],
                validation_details={'error': str(e)}
            )
    
    def _calculate_overall_score(self, issues: List[ValidationIssue]) -> float:
        """计算总体验证得分"""
        if not issues:
            return 1.0
        
        # 基础得分
        base_score = 1.0
        
        # 根据问题严重性扣分
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 0.2
            elif issue.severity == IssueSeverity.ERROR:
                base_score -= 0.1
            elif issue.severity == IssueSeverity.WARNING:
                base_score -= 0.05
            elif issue.severity == IssueSeverity.INFO:
                base_score -= 0.02
        
        # 根据置信度调整
        avg_confidence = sum(issue.confidence for issue in issues) / len(issues)
        score_adjustment = avg_confidence * 0.1
        
        return max(0.0, min(1.0, base_score + score_adjustment))
    
    def _calculate_validation_metrics(self, issues: List[ValidationIssue], 
                                    fusion_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算验证指标"""
        # 问题统计
        severity_stats = Counter(issue.severity for issue in issues)
        type_stats = Counter(issue.issue_type for issue in issues)
        
        # 融合统计
        entity_fusion_stats = fusion_result.get('entity_fusion', {}).get('fusion_statistics', {})
        relation_fusion_stats = fusion_result.get('relation_fusion', {}).get('fusion_statistics', {})
        
        return {
            'total_issues': len(issues),
            'severity_distribution': {k.value: v for k, v in severity_stats.items()},
            'type_distribution': {k.value: v for k, v in type_stats.items()},
            'entity_merge_efficiency': entity_fusion_stats.get('merge_efficiency', 0.0),
            'relation_merge_efficiency': relation_fusion_stats.get('merge_efficiency', 0.0),
            'average_confidence': sum(issue.confidence for issue in issues) / len(issues) if issues else 1.0,
            'critical_issues_count': severity_stats.get(IssueSeverity.CRITICAL, 0),
            'error_issues_count': severity_stats.get(IssueSeverity.ERROR, 0),
            'warning_issues_count': severity_stats.get(IssueSeverity.WARNING, 0)
        }
    
    def _generate_validation_recommendations(self, issues: List[ValidationIssue], 
                                           quality_report: Dict[str, Any]) -> List[str]:
        """生成验证建议"""
        recommendations = []
        
        # 基于问题的建议
        if not issues:
            recommendations.append("验证通过，没有发现问题")
        else:
            # 统计问题类型
            type_counts = Counter(issue.issue_type for issue in issues)
            
            if type_counts[ConsistencyType.LOGICAL] > 0:
                recommendations.append("检查并修正逻辑一致性问题，确保推理规则的正确性")
            
            if type_counts[ConsistencyType.FACTUAL] > 0:
                recommendations.append("验证事实信息的准确性，使用外部知识源进行交叉验证")
            
            if type_counts[ConsistencyType.TEMPORAL] > 0:
                recommendations.append("解决时间信息冲突，确保时间序列的一致性")
            
            critical_issues = [issue for issue in issues if issue.severity == IssueSeverity.CRITICAL]
            if critical_issues:
                recommendations.append(f"优先解决 {len(critical_issues)} 个严重问题，这些问题影响推理结果的有效性")
        
        # 基于质量报告的建议
        if quality_report:
            weaknesses = quality_report.get('weaknesses', [])
            for weakness in weaknesses:
                recommendations.append(f"针对 {weakness}，需要采取改进措施")
        
        return recommendations

def create_reasoning_validator(config: Optional[Dict] = None) -> ReasoningValidator:
    """创建推理验证器实例的工厂函数"""
    return ReasoningValidator(config)

if __name__ == "__main__":
    import hashlib
    
    # 测试代码
    config = {
        'validation_level': 'comprehensive',
        'max_issues': 50,
        'enable_quality_evaluation': True,
        'logical': {'similarity_threshold': 0.8},
        'factual': {'confidence_threshold': 0.7}
    }
    
    validator = create_reasoning_validator(config)
    
    # 测试数据
    test_input = {
        'entities': [
            {'name': '苹果公司', 'type': 'ORG', 'confidence': 0.8},
            {'name': '苹果公司', 'type': 'PERSON', 'confidence': 0.7}  # 类型冲突
        ],
        'relations': [
            {'source_name': '苹果公司', 'target_name': '中国', 'relation_type': 'located_in', 'confidence': 0.7}
        ]
    }
    
    test_fusion_result = {
        'entity_fusion': {
            'fusion_statistics': {
                'original_count': 2,
                'fused_count': 1,
                'merge_efficiency': 0.5
            }
        },
        'relation_fusion': {
            'fusion_statistics': {
                'original_count': 1,
                'fused_count': 1,
                'merge_efficiency': 1.0
            }
        },
        'overall_quality': {
            'overall_quality': 0.75
        }
    }
    
    # 执行验证
    result = validator.validate_reasoning_result(test_input, test_fusion_result)
    
    print("推理验证结果:")
    print(f"总体得分: {result.overall_score:.3f}")
    print(f"是否有效: {result.is_valid}")
    print(f"验证问题数: {len(result.issues)}")
    print(f"验证指标: {result.metrics}")
    print(f"建议: {result.recommendations}")
