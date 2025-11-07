"""
知识融合策略模块
实现实体验证、关系补充和知识图谱集成功能
支持多种融合算法和冲突解决策略
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import Counter, defaultdict
import hashlib

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusionStrategy(Enum):
    """融合策略枚举"""
    EARLY_FUSION = "early_fusion"      # 早期融合
    LATE_FUSION = "late_fusion"        # 晚期融合
    WEIGHTED_FUSION = "weighted_fusion" # 加权融合
    RULE_BASED_FUSION = "rule_based_fusion"  # 基于规则的融合
    ML_BASED_FUSION = "ml_based_fusion"  # 基于机器学习的融合

class ConflictResolution(Enum):
    """冲突解决策略"""
    HIGHEST_CONFIDENCE = "highest_confidence"  # 最高置信度优先
    MAJORITY_VOTE = "majority_vote"           # 多数投票
    WEIGHTED_VOTE = "weighted_vote"           # 加权投票
    RULE_BASED_RESOLUTION = "rule_based"      # 基于规则解决
    MANUAL_RESOLUTION = "manual"              # 人工解决

@dataclass
class EntityRecord:
    """实体记录数据结构"""
    entity_id: str
    name: str
    entity_type: str
    confidence: float
    source: str
    properties: Dict[str, Any]
    timestamp: Optional[str] = None
    quality_score: float = 0.0

@dataclass
class RelationRecord:
    """关系记录数据结构"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    source: str
    evidence: List[str]
    context: Dict[str, Any]
    timestamp: Optional[str] = None
    quality_score: float = 0.0

@dataclass
class FusionDecision:
    """融合决策结果"""
    decision_type: str  # "accept", "reject", "merge", "create_new"
    merged_entity: Optional[EntityRecord] = None
    merged_relation: Optional[RelationRecord] = None
    confidence_score: float = 0.0
    reasoning: str = ""
    alternatives: List[Dict] = None

class EntityFusionEngine:
    """实体重融合引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化实体重融合引擎"""
        self.config = config or {}
        
        # 配置参数
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.merge_threshold = self.config.get('merge_threshold', 0.6)
        
        # 相似度计算器
        self.similarity_calculators = {
            'exact_match': self._exact_match_similarity,
            'fuzzy_match': self._fuzzy_match_similarity,
            'semantic_match': self._semantic_match_similarity,
            'type_based_match': self._type_based_match_similarity
        }
        
        logger.info("实体重融合引擎初始化完成")
    
    def fuse_entities(self, entities: List[Dict], knowledge_graph: Any = None) -> Dict[str, Any]:
        """融合实体集合"""
        try:
            logger.info(f"开始融合 {len(entities)} 个实体")
            
            # 转换为标准格式
            entity_records = [self._convert_to_entity_record(entity) for entity in entities]
            
            # 计算实体相似度
            similarity_matrix = self._calculate_similarity_matrix(entity_records)
            
            # 执行实体聚类
            clusters = self._cluster_similar_entities(entity_records, similarity_matrix)
            
            # 对每个聚类进行融合
            fused_entities = []
            for cluster in clusters:
                fused_entity = self._fuse_entity_cluster(cluster, knowledge_graph)
                if fused_entity:
                    fused_entities.append(fused_entity)
            
            # 处理未聚类的实体
            unclustered_entities = self._process_unclustered_entities(entity_records, clusters)
            fused_entities.extend(unclustered_entities)
            
            fusion_result = {
                'fused_entities': fused_entities,
                'fusion_statistics': {
                    'original_count': len(entities),
                    'fused_count': len(fused_entities),
                    'merge_efficiency': len(entities) / max(len(fused_entities), 1),
                    'clusters_formed': len(clusters),
                    'unclustered_count': len(unclustered_entities)
                },
                'similarity_matrix': similarity_matrix,
                'clustering_results': clusters
            }
            
            logger.info(f"实体融合完成: {len(entities)} -> {len(fused_entities)} 个实体")
            return fusion_result
            
        except Exception as e:
            logger.error(f"实体重融合失败: {e}")
            return {'error': str(e)}
    
    def _convert_to_entity_record(self, entity: Dict) -> EntityRecord:
        """将字典转换为实体记录"""
        return EntityRecord(
            entity_id=entity.get('id', f"entity_{hashlib.md5(str(entity).encode()).hexdigest()[:8]}"),
            name=entity.get('name', ''),
            entity_type=entity.get('type', 'UNKNOWN'),
            confidence=entity.get('confidence', 0.0),
            source=entity.get('source', 'unknown'),
            properties=entity.get('properties', {}),
            timestamp=entity.get('timestamp'),
            quality_score=self._calculate_entity_quality(entity)
        )
    
    def _calculate_entity_quality(self, entity: Dict) -> float:
        """计算实体质量分数"""
        quality_score = 0.0
        
        # 基础置信度权重
        confidence = entity.get('confidence', 0.0)
        quality_score += confidence * 0.4
        
        # 名称完整性
        name = entity.get('name', '')
        if name:
            quality_score += 0.2
        
        # 类型信息
        entity_type = entity.get('type', '')
        if entity_type != 'UNKNOWN':
            quality_score += 0.2
        
        # 属性数量
        properties = entity.get('properties', {})
        if properties:
            quality_score += min(len(properties) * 0.1, 0.2)
        
        return min(quality_score, 1.0)
    
    def _calculate_similarity_matrix(self, entity_records: List[EntityRecord]) -> np.ndarray:
        """计算实体相似度矩阵"""
        n = len(entity_records)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = self._calculate_pairwise_similarity(entity_records[i], entity_records[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        logger.debug(f"相似度矩阵计算完成: {n}x{n}")
        return similarity_matrix
    
    def _calculate_pairwise_similarity(self, entity1: EntityRecord, entity2: EntityRecord) -> float:
        """计算两个实体之间的相似度"""
        similarity_scores = []
        
        # 名称相似度
        name_similarity = self.similarity_calculators['exact_match'](entity1.name, entity2.name) * 0.4
        similarity_scores.append(name_similarity)
        
        # 类型相似度
        type_similarity = self.similarity_calculators['type_based_match'](entity1.entity_type, entity2.entity_type) * 0.3
        similarity_scores.append(type_similarity)
        
        # 模糊匹配相似度
        fuzzy_similarity = self.similarity_calculators['fuzzy_match'](entity1.name, entity2.name) * 0.2
        similarity_scores.append(fuzzy_similarity)
        
        # 属性相似度
        prop_similarity = self._calculate_property_similarity(entity1.properties, entity2.properties) * 0.1
        similarity_scores.append(prop_similarity)
        
        return sum(similarity_scores)
    
    def _exact_match_similarity(self, str1: str, str2: str) -> float:
        """精确匹配相似度"""
        if not str1 or not str2:
            return 0.0
        
        return 1.0 if str1.lower() == str2.lower() else 0.0
    
    def _fuzzy_match_similarity(self, str1: str, str2: str) -> float:
        """模糊匹配相似度"""
        if not str1 or not str2:
            return 0.0
        
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        
        # 检查包含关系
        if str1_lower in str2_lower or str2_lower in str1_lower:
            return 0.8
        
        # 计算编辑距离相似度
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, str1_lower, str2_lower).ratio()
        return similarity
    
    def _semantic_match_similarity(self, str1: str, str2: str) -> float:
        """语义匹配相似度（简化版）"""
        # 简化的语义匹配，实际应用中可以使用词向量或预训练模型
        synonyms = {
            '公司': ['corporation', 'inc', '企业'],
            '机构': ['organization', 'institution', '组织'],
            '个人': ['person', 'individual', '人物']
        }
        
        str1_lower = str1.lower()
        str2_lower = str2.lower()
        
        for base_word, synonym_list in synonyms.items():
            if base_word in str1_lower and any(syn in str2_lower for syn in synonym_list):
                return 0.7
            if base_word in str2_lower and any(syn in str1_lower for syn in synonym_list):
                return 0.7
        
        return 0.0
    
    def _type_based_match_similarity(self, type1: str, type2: str) -> float:
        """基于类型的匹配相似度"""
        if type1 == type2:
            return 1.0
        
        # 类型层次关系
        type_hierarchy = {
            'ORG': ['COMPANY', 'INSTITUTION', 'GOVERNMENT'],
            'PERSON': ['INDIVIDUAL', 'PEOPLE'],
            'LOC': ['PLACE', 'GEOGRAPHY', 'REGION'],
            'MISC': ['THING', 'OBJECT']
        }
        
        for base_type, subtypes in type_hierarchy.items():
            if (type1 == base_type and type2 in subtypes) or (type2 == base_type and type1 in subtypes):
                return 0.8
        
        return 0.0
    
    def _calculate_property_similarity(self, props1: Dict, props2: Dict) -> float:
        """计算属性相似度"""
        if not props1 or not props2:
            return 0.0
        
        common_keys = set(props1.keys()) & set(props2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = props1[key], props2[key]
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 数值相似度
                if max(val1, val2) > 0:
                    similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(similarity)
            else:
                # 字符串相似度
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, str(val1), str(val2)).ratio()
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _cluster_similar_entities(self, entity_records: List[EntityRecord], 
                                similarity_matrix: np.ndarray) -> List[List[EntityRecord]]:
        """聚类相似实体"""
        n = len(entity_records)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if not visited[i]:
                cluster = [entity_records[i]]
                visited[i] = True
                
                # 找到所有相似的实体
                for j in range(i+1, n):
                    if not visited[j] and similarity_matrix[i][j] >= self.similarity_threshold:
                        cluster.append(entity_records[j])
                        visited[j] = True
                
                clusters.append(cluster)
        
        logger.info(f"实体聚类完成: {len(clusters)} 个聚类")
        return clusters
    
    def _fuse_entity_cluster(self, cluster: List[EntityRecord], 
                           knowledge_graph: Any = None) -> Optional[Dict]:
        """融合实体聚类"""
        if len(cluster) == 1:
            # 单个实体，直接返回
            return self._convert_entity_record_to_dict(cluster[0])
        
        # 多个实体需要融合
        fused_name = self._resolve_name_conflict(cluster)
        fused_type = self._resolve_type_conflict(cluster)
        fused_properties = self._merge_properties([e.properties for e in cluster])
        fused_confidence = self._calculate_fused_confidence(cluster)
        
        # 验证融合结果
        if fused_confidence < self.merge_threshold:
            logger.warning(f"实体融合置信度过低: {fused_confidence:.3f}")
            return None
        
        fused_entity = {
            'id': f"fused_{hashlib.md5(fused_name.encode()).hexdigest()[:8]}",
            'name': fused_name,
            'type': fused_type,
            'confidence': fused_confidence,
            'source': 'fusion',
            'properties': fused_properties,
            'original_entities': [asdict(e) for e in cluster],
            'merge_reason': f"融合了 {len(cluster)} 个相似实体"
        }
        
        # 使用知识图谱验证
        if knowledge_graph:
            kg_validation = self._validate_with_knowledge_graph(fused_entity, knowledge_graph)
            fused_entity['kg_validation'] = kg_validation
            if not kg_validation.get('is_valid', True):
                logger.warning(f"知识图谱验证失败: {kg_validation}")
        
        return fused_entity
    
    def _resolve_name_conflict(self, cluster: List[EntityRecord]) -> str:
        """解决名称冲突"""
        names = [(e.name, e.confidence, e.quality_score) for e in cluster]
        
        # 按置信度和质量分数排序
        names.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return names[0][0]
    
    def _resolve_type_conflict(self, cluster: List[EntityRecord]) -> str:
        """解决类型冲突"""
        types = [e.entity_type for e in cluster]
        type_counts = Counter(types)
        
        # 多数投票
        most_common_type = type_counts.most_common(1)[0][0]
        
        # 如果最常见的类型置信度太低，回退到最高置信度的类型
        max_confidence_type = max(cluster, key=lambda e: e.confidence).entity_type
        
        if type_counts[most_common_type] / len(cluster) >= 0.5:
            return most_common_type
        else:
            return max_confidence_type
    
    def _merge_properties(self, properties_list: List[Dict]) -> Dict[str, Any]:
        """合并属性"""
        merged_props = {}
        
        for props in properties_list:
            for key, value in props.items():
                if key not in merged_props:
                    merged_props[key] = []
                merged_props[key].append(value)
        
        # 对每个属性值进行融合
        for key, values in merged_props.items():
            if len(values) == 1:
                merged_props[key] = values[0]
            else:
                # 数值类型取平均，字符串类型取最频繁的
                if all(isinstance(v, (int, float)) for v in values):
                    merged_props[key] = sum(values) / len(values)
                else:
                    from collections import Counter
                    most_common = Counter(values).most_common(1)
                    merged_props[key] = most_common[0][0] if most_common else values[0]
        
        return merged_props
    
    def _calculate_fused_confidence(self, cluster: List[EntityRecord]) -> float:
        """计算融合后的置信度"""
        if not cluster:
            return 0.0
        
        # 加权平均，权重基于质量和置信度
        total_weight = 0.0
        weighted_sum = 0.0
        
        for entity in cluster:
            weight = entity.confidence * entity.quality_score
            weighted_sum += entity.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return sum(e.confidence for e in cluster) / len(cluster)
    
    def _validate_with_knowledge_graph(self, entity: Dict, knowledge_graph: Any) -> Dict[str, Any]:
        """使用知识图谱验证实体"""
        try:
            # 查询知识图谱中的匹配实体
            matches = knowledge_graph.query_entities(entity['name'], entity['type'])
            
            if matches:
                # 计算匹配度
                match_scores = []
                for match in matches:
                    score = self._calculate_entity_match_score(entity, match)
                    match_scores.append(score)
                
                max_score = max(match_scores) if match_scores else 0.0
                
                return {
                    'is_valid': max_score >= self.confidence_threshold,
                    'max_match_score': max_score,
                    'matches_found': len(matches),
                    'validation_details': {
                        'kg_entities': [asdict(match) for match in matches],
                        'match_scores': match_scores
                    }
                }
            else:
                return {
                    'is_valid': True,  # 新实体默认有效
                    'max_match_score': 0.0,
                    'matches_found': 0,
                    'validation_details': {
                        'note': 'No matching entities found in knowledge graph'
                    }
                }
                
        except Exception as e:
            logger.error(f"知识图谱验证失败: {e}")
            return {
                'is_valid': False,
                'error': str(e)
            }
    
    def _calculate_entity_match_score(self, entity: Dict, kg_entity: Any) -> float:
        """计算实体与知识图谱实体的匹配分数"""
        score = 0.0
        
        # 名称匹配
        if entity.get('name', '').lower() == kg_entity.entity_name.lower():
            score += 0.5
        
        # 类型匹配
        if entity.get('type', '') == kg_entity.entity_type:
            score += 0.3
        
        # 置信度权重
        score *= kg_entity.confidence
        
        return score
    
    def _process_unclustered_entities(self, entity_records: List[EntityRecord], 
                                    clusters: List[List[EntityRecord]]) -> List[Dict]:
        """处理未聚类的实体"""
        clustered_ids = set()
        for cluster in clusters:
            for entity in cluster:
                clustered_ids.add(entity.entity_id)
        
        unclustered = []
        for entity in entity_records:
            if entity.entity_id not in clustered_ids:
                entity_dict = self._convert_entity_record_to_dict(entity)
                entity_dict['merge_reason'] = 'singleton_entity'
                unclustered.append(entity_dict)
        
        return unclustered
    
    def _convert_entity_record_to_dict(self, entity_record: EntityRecord) -> Dict[str, Any]:
        """将实体记录转换为字典"""
        return asdict(entity_record)

class RelationFusionEngine:
    """关系融合引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化关系融合引擎"""
        self.config = config or {}
        
        # 配置参数
        self.relation_similarity_threshold = self.config.get('relation_similarity_threshold', 0.8)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        logger.info("关系融合引擎初始化完成")
    
    def fuse_relations(self, relations: List[Dict], entity_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """融合关系集合"""
        try:
            logger.info(f"开始融合 {len(relations)} 个关系")
            
            # 转换为标准格式
            relation_records = [self._convert_to_relation_record(relation) for relation in relations]
            
            # 应用实体映射
            if entity_mapping:
                relation_records = self._apply_entity_mapping(relation_records, entity_mapping)
            
            # 计算关系相似度
            similarity_groups = self._group_similar_relations(relation_records)
            
            # 融合相似关系
            fused_relations = []
            for group in similarity_groups:
                fused_relation = self._fuse_relation_group(group)
                if fused_relation:
                    fused_relations.append(fused_relation)
            
            fusion_result = {
                'fused_relations': fused_relations,
                'fusion_statistics': {
                    'original_count': len(relations),
                    'fused_count': len(fused_relations),
                    'merge_efficiency': len(relations) / max(len(fused_relations), 1),
                    'groups_formed': len(similarity_groups)
                },
                'similarity_groups': similarity_groups
            }
            
            logger.info(f"关系融合完成: {len(relations)} -> {len(fused_relations)} 个关系")
            return fusion_result
            
        except Exception as e:
            logger.error(f"关系融合失败: {e}")
            return {'error': str(e)}
    
    def _convert_to_relation_record(self, relation: Dict) -> RelationRecord:
        """将字典转换为关系记录"""
        return RelationRecord(
            source_id=relation.get('source_id', relation.get('source_name', '')),
            target_id=relation.get('target_id', relation.get('target_name', '')),
            relation_type=relation.get('relation_type', ''),
            confidence=relation.get('confidence', 0.0),
            source=relation.get('source', 'unknown'),
            evidence=relation.get('evidence', []),
            context=relation.get('context', {}),
            timestamp=relation.get('timestamp'),
            quality_score=self._calculate_relation_quality(relation)
        )
    
    def _calculate_relation_quality(self, relation: Dict) -> float:
        """计算关系质量分数"""
        quality_score = 0.0
        
        # 基础置信度权重
        confidence = relation.get('confidence', 0.0)
        quality_score += confidence * 0.5
        
        # 关系类型完整性
        relation_type = relation.get('relation_type', '')
        if relation_type:
            quality_score += 0.2
        
        # 证据数量
        evidence = relation.get('evidence', [])
        if evidence:
            quality_score += min(len(evidence) * 0.1, 0.2)
        
        # 上下文完整性
        context = relation.get('context', {})
        if context:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _apply_entity_mapping(self, relation_records: List[RelationRecord], 
                            entity_mapping: Dict[str, str]) -> List[RelationRecord]:
        """应用实体映射"""
        mapped_relations = []
        
        for relation in relation_records:
            mapped_relation = RelationRecord(
                source_id=entity_mapping.get(relation.source_id, relation.source_id),
                target_id=entity_mapping.get(relation.target_id, relation.target_id),
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                source=relation.source,
                evidence=relation.evidence,
                context=relation.context,
                timestamp=relation.timestamp,
                quality_score=relation.quality_score
            )
            mapped_relations.append(mapped_relation)
        
        return mapped_relations
    
    def _group_similar_relations(self, relation_records: List[RelationRecord]) -> List[List[RelationRecord]]:
        """分组相似关系"""
        groups = []
        used_relations = set()
        
        for i, relation in enumerate(relation_records):
            if i in used_relations:
                continue
            
            group = [relation]
            used_relations.add(i)
            
            for j, other_relation in enumerate(relation_records[i+1:], i+1):
                if j in used_relations:
                    continue
                
                similarity = self._calculate_relation_similarity(relation, other_relation)
                if similarity >= self.relation_similarity_threshold:
                    group.append(other_relation)
                    used_relations.add(j)
            
            groups.append(group)
        
        logger.info(f"关系分组完成: {len(groups)} 个分组")
        return groups
    
    def _calculate_relation_similarity(self, rel1: RelationRecord, rel2: RelationRecord) -> float:
        """计算两个关系之间的相似度"""
        similarity_scores = []
        
        # 源实体相似度
        source_similarity = 1.0 if rel1.source_id == rel2.source_id else 0.0
        similarity_scores.append(source_similarity * 0.3)
        
        # 目标实体相似度
        target_similarity = 1.0 if rel1.target_id == rel2.target_id else 0.0
        similarity_scores.append(target_similarity * 0.3)
        
        # 关系类型相似度
        type_similarity = 1.0 if rel1.relation_type == rel2.relation_type else 0.0
        similarity_scores.append(type_similarity * 0.4)
        
        return sum(similarity_scores)
    
    def _fuse_relation_group(self, group: List[RelationRecord]) -> Optional[Dict]:
        """融合关系分组"""
        if len(group) == 1:
            # 单个关系，直接返回
            return asdict(group[0])
        
        # 多个关系需要融合
        fused_confidence = self._calculate_group_confidence(group)
        
        if fused_confidence < self.confidence_threshold:
            logger.warning(f"关系融合置信度过低: {fused_confidence:.3f}")
            return None
        
        fused_relation = {
            'source_id': group[0].source_id,
            'target_id': group[0].target_id,
            'relation_type': group[0].relation_type,
            'confidence': fused_confidence,
            'source': 'fusion',
            'evidence': list(set().union(*[rel.evidence for rel in group])),
            'context': self._merge_contexts([rel.context for rel in group]),
            'original_relations': [asdict(rel) for rel in group],
            'merge_reason': f"融合了 {len(group)} 个相似关系"
        }
        
        return fused_relation
    
    def _calculate_group_confidence(self, group: List[RelationRecord]) -> float:
        """计算分组置信度"""
        if not group:
            return 0.0
        
        # 加权平均
        total_weight = sum(rel.quality_score for rel in group)
        if total_weight > 0:
            weighted_sum = sum(rel.confidence * rel.quality_score for rel in group)
            return weighted_sum / total_weight
        else:
            return sum(rel.confidence for rel in group) / len(group)
    
    def _merge_contexts(self, contexts: List[Dict]) -> Dict[str, Any]:
        """合并上下文信息"""
        merged_context = {}
        
        for context in contexts:
            for key, value in context.items():
                if key not in merged_context:
                    merged_context[key] = []
                merged_context[key].append(value)
        
        # 对每个上下文值进行融合
        for key, values in merged_context.items():
            if len(values) == 1:
                merged_context[key] = values[0]
            else:
                # 数值类型取平均，字符串类型取最频繁的
                if all(isinstance(v, (int, float)) for v in values):
                    merged_context[key] = sum(values) / len(values)
                else:
                    from collections import Counter
                    most_common = Counter(values).most_common(1)
                    merged_context[key] = most_common[0][0] if most_common else values[0]
        
        return merged_context

class KnowledgeFusionEngine:
    """综合知识融合引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化综合知识融合引擎"""
        self.config = config or {}
        
        # 初始化各个融合引擎
        self.entity_fusion_engine = EntityFusionEngine(self.config.get('entity_fusion', {}))
        self.relation_fusion_engine = RelationFusionEngine(self.config.get('relation_fusion', {}))
        
        # 融合策略
        self.fusion_strategy = FusionStrategy(self.config.get('fusion_strategy', 'weighted_fusion'))
        self.conflict_resolution = ConflictResolution(self.config.get('conflict_resolution', 'majority_vote'))
        
        logger.info("综合知识融合引擎初始化完成")
    
    def perform_fusion(self, input_data: Dict[str, Any], knowledge_graph: Any = None) -> Dict[str, Any]:
        """执行综合知识融合"""
        try:
            logger.info("开始执行综合知识融合")
            
            # 提取输入数据
            entities = input_data.get('entities', [])
            relations = input_data.get('relations', [])
            
            # 执行实体融合
            entity_fusion_result = self.entity_fusion_engine.fuse_entities(entities, knowledge_graph)
            
            # 构建实体映射
            entity_mapping = self._build_entity_mapping(entities, entity_fusion_result)
            
            # 执行关系融合
            relation_fusion_result = self.relation_fusion_engine.fuse_relations(relations, entity_mapping)
            
            # 生成融合决策
            fusion_decisions = self._generate_fusion_decisions(
                entity_fusion_result, relation_fusion_result)
            
            # 计算整体融合质量
            overall_quality = self._calculate_overall_quality(
                entity_fusion_result, relation_fusion_result)
            
            fusion_result = {
                'entity_fusion': entity_fusion_result,
                'relation_fusion': relation_fusion_result,
                'fusion_decisions': fusion_decisions,
                'overall_quality': overall_quality,
                'fusion_metadata': {
                    'strategy': self.fusion_strategy.value,
                    'conflict_resolution': self.conflict_resolution.value,
                    'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else None,
                    'config': self.config
                }
            }
            
            logger.info("综合知识融合完成")
            return fusion_result
            
        except Exception as e:
            logger.error(f"综合知识融合失败: {e}")
            return {'error': str(e)}
    
    def _build_entity_mapping(self, original_entities: List[Dict], 
                            entity_fusion_result: Dict) -> Dict[str, str]:
        """构建实体映射"""
        mapping = {}
        
        # 原始实体ID到融合实体ID的映射
        original_ids = {entity.get('id', entity.get('name', '')): entity for entity in original_entities}
        
        for fused_entity in entity_fusion_result.get('fused_entities', []):
            original_entities_list = fused_entity.get('original_entities', [])
            
            for original_entity in original_entities_list:
                original_id = original_entity.get('entity_id', original_entity.get('name', ''))
                mapping[original_id] = fused_entity.get('id', '')
        
        return mapping
    
    def _generate_fusion_decisions(self, entity_result: Dict, relation_result: Dict) -> List[FusionDecision]:
        """生成融合决策"""
        decisions = []
        
        # 实体融合决策
        for fused_entity in entity_result.get('fused_entities', []):
            if fused_entity.get('merge_reason') != 'singleton_entity':
                decision = FusionDecision(
                    decision_type="merge",
                    merged_entity=asdict(fused_entity) if isinstance(fused_entity, dict) else fused_entity,
                    confidence_score=fused_entity.get('confidence', 0.0),
                    reasoning=f"基于相似度阈值和置信度合并了 {len(fused_entity.get('original_entities', []))} 个实体",
                    alternatives=[]
                )
                decisions.append(decision)
        
        # 关系融合决策
        for fused_relation in relation_result.get('fused_relations', []):
            if fused_relation.get('merge_reason') != 'singleton_relation':
                decision = FusionDecision(
                    decision_type="merge",
                    merged_relation=asdict(fused_relation) if isinstance(fused_relation, dict) else fused_relation,
                    confidence_score=fused_relation.get('confidence', 0.0),
                    reasoning=f"基于实体映射和关系相似度合并了 {len(fused_relation.get('original_relations', []))} 个关系",
                    alternatives=[]
                )
                decisions.append(decision)
        
        return decisions
    
    def _calculate_overall_quality(self, entity_result: Dict, relation_result: Dict) -> Dict[str, Any]:
        """计算整体融合质量"""
        entity_stats = entity_result.get('fusion_statistics', {})
        relation_stats = relation_result.get('fusion_statistics', {})
        
        # 计算质量指标
        entity_efficiency = entity_stats.get('merge_efficiency', 1.0)
        relation_efficiency = relation_stats.get('merge_efficiency', 1.0)
        overall_efficiency = (entity_efficiency + relation_efficiency) / 2
        
        # 计算一致性指标
        clusters_formed = entity_stats.get('clusters_formed', 0)
        original_entities = entity_stats.get('original_count', 1)
        cluster_ratio = clusters_formed / original_entities if original_entities > 0 else 0
        
        quality_metrics = {
            'efficiency_score': overall_efficiency,
            'clustering_score': 1.0 - cluster_ratio,  # 聚类越少，分数越高
            'coverage_score': (entity_stats.get('fused_count', 0) + relation_stats.get('fused_count', 0)) / \
                             max(entity_stats.get('original_count', 1) + relation_stats.get('original_count', 1), 1),
            'overall_quality': (overall_efficiency + (1.0 - cluster_ratio)) / 2
        }
        
        return quality_metrics

def create_fusion_engine(config: Optional[Dict] = None) -> KnowledgeFusionEngine:
    """创建融合引擎实例的工厂函数"""
    return KnowledgeFusionEngine(config)

if __name__ == "__main__":
    # 测试代码
    config = {
        'similarity_threshold': 0.8,
        'confidence_threshold': 0.7,
        'merge_threshold': 0.6,
        'fusion_strategy': 'weighted_fusion',
        'conflict_resolution': 'majority_vote'
    }
    
    fusion_engine = create_fusion_engine(config)
    
    # 测试数据
    test_entities = [
        {'name': '苹果公司', 'type': 'ORG', 'confidence': 0.8, 'source': 'docred'},
        {'name': 'Apple Inc.', 'type': 'ORG', 'confidence': 0.7, 'source': 'kg'},
        {'name': '中国', 'type': 'LOC', 'confidence': 0.9, 'source': 'docred'}
    ]
    
    test_relations = [
        {'source_name': '苹果公司', 'target_name': '中国', 'relation_type': 'located_in', 'confidence': 0.7},
        {'source_name': 'Apple Inc.', 'target_name': 'China', 'relation_type': 'located_in', 'confidence': 0.6}
    ]
    
    test_input = {
        'entities': test_entities,
        'relations': test_relations
    }
    
    # 执行融合
    result = fusion_engine.perform_fusion(test_input)
    
    print("知识融合结果:")
    print(f"实体融合统计: {result['entity_fusion'].get('fusion_statistics', {})}")
    print(f"关系融合统计: {result['relation_fusion'].get('fusion_statistics', {})}")
    print(f"整体质量: {result['overall_quality']}")
    print(f"融合决策: {len(result['fusion_decisions'])} 个")
