"""
格式化器模块
将提取的要素转换为DocRED数据集格式
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from collections import defaultdict
import re

from extractor import ExtractedEntity, ExtractedRelation
from classifier import ElementClassifier

logger = logging.getLogger(__name__)


@dataclass
class DocREDElement:
    """DocRED格式的要素"""
    title: str
    sents: List[List[str]]  # 句子列表，每个句子是词列表
    vertexSet: List[Dict[str, Any]]  # 实体集合
    labels: List[Dict[str, Any]]  # 关系标签集合
    entities_count: int
    relations_count: int
    original_text: str
    extraction_metadata: Dict[str, Any]


class DocREDFormatter:
    """DocRED格式转换器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 实体类型映射
        self.entity_type_mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'LOC': 'LOC',
            'TIME': 'TIME',
            'MISC': 'MISC'
        }
        
        # 关系类型映射到DocRED格式
        self.relation_type_mapping = {
            'PHYSICAL': {
                'LOCATED_IN': '/location/location_contains',
                'NEAR': '/location/location_near',
                'PART_OF': '/physical/physical_part_of'
            },
            'AFFILIATION': {
                'WORKS_FOR': '/people/person/social_position',
                'STUDIES_AT': '/education/education/institution',
                'MEMBER_OF': '/people/person/group_membership'
            },
            'PERSONAL': {
                'MARRIED_TO': '/people/person/spouse_s',
                'FAMILY': '/people/person/parent',
                'SIBLING': '/people/person/sibling_s'
            },
            'TEMPORAL': {
                'BEFORE': '/time/temporal/temporal_after',
                'AFTER': '/time/temporal/temporal_before',
                'SIMULTANEOUS': '/time/temporal/temporal_same_time'
            },
            'CAUSAL': {
                'CAUSES': '/causal_result/causes',
                'AFFECTS': '/causal_result/affects',
                'LEADS_TO': '/causal_result/leads_to'
            }
        }
        
        # 默认DocRED关系类型
        self.default_docred_relations = {
            'P6': '/people/person/affiliation',
            'P17': '/people/person/spouse_s',
            'P19': '/people/person/place_of_birth',
            'P20': '/people/person/place_of_death',
            'P22': '/people/person/place_of_birth',
            'P25': '/people/person/parent',
            'P26': '/people/person/spouse_s',
            'P27': '/people/person/nationality',
            'P30': '/people/person/citizen_of',
            'P31': '/organization/organization_founded_by',
            'P35': '/government/government_position_held',
            'P39': '/people/person/occupation',
            'P40': '/people/person/participant_in',
            'P50': '/organization/organization_founded_by',
            'P54': '/organization/organization_founded_by',
            'P57': '/person/person/educated_at',
            'P58': '/organization/organization_founded_by',
            'P69': '/organization/organization_member_of',
            '74': '/location/location_contains',
            '97': '/organization/organization_founded_by',
            '130': '/location/location_contains'
        }
    
    def format_to_docred(self, extraction_result: Dict[str, Any], 
                        document_id: str, original_text: str = "") -> DocREDElement:
        """
        将提取结果转换为DocRED格式
        
        Args:
            extraction_result: 提取结果字典
            document_id: 文档ID
            original_text: 原始文本（可选）
            
        Returns:
            DocREDElement: DocRED格式的要素
        """
        try:
            logger.info(f"开始转换为DocRED格式，文档ID: {document_id}")
            
            # 获取提取的要素
            classified_entities = extraction_result.get('classified_entities', [])
            classified_relations = extraction_result.get('classified_relations', [])
            events = extraction_result.get('extracted_events', [])
            attributes = extraction_result.get('extracted_attributes', [])
            
            # 转换实体
            docred_entities = self._convert_entities_to_docred_format(
                classified_entities, original_text
            )
            
            # 转换关系
            docred_relations = self._convert_relations_to_docred_format(
                classified_relations, docred_entities
            )
            
            # 构建DocRED文档结构
            docred_element = DocREDElement(
                title=document_id,
                sents=self._split_into_sentences(original_text),
                vertexSet=docred_entities,
                labels=docred_relations,
                entities_count=len(docred_entities),
                relations_count=len(docred_relations),
                original_text=original_text,
                extraction_metadata={
                    'extraction_confidence': extraction_result.get('quality_metrics', {}),
                    'extraction_method': 'ast_based_extraction',
                    'extraction_parameters': self.config
                }
            )
            
            logger.info(f"DocRED转换完成 - 实体: {len(docred_entities)}, 关系: {len(docred_relations)}")
            return docred_element
            
        except Exception as e:
            logger.error(f"DocRED格式转换失败: {str(e)}")
            raise
    
    def _convert_entities_to_docred_format(self, entities: List[Dict[str, Any]], 
                                          original_text: str) -> List[Dict[str, Any]]:
        """转换实体为DocRED格式"""
        docred_entities = []
        
        for i, entity in enumerate(entities):
            # 提取实体信息
            entity_id = entity.get('id', f'entity_{i}')
            text = entity.get('text', '')
            category = entity.get('category', 'MISC')
            confidence = entity.get('confidence', 0.0)
            
            # 获取提及信息
            mentions = entity.get('mentions', [])
            if not mentions:
                # 如果没有提及信息，创建一个默认的
                mentions = [{
                    'sent_idx': 0,
                    'pos': [0, len(text)],
                    'type': str(category).lower()
                }]
            
            # 转换提及格式
            docred_mentions = []
            for mention in mentions:
                docred_mention = {
                    'sent_idx': mention.get('sent_idx', 0),
                    'pos': mention.get('pos', [0, len(text)]),
                    'type': mention.get('type', str(category).lower())
                }
                docred_mentions.append(docred_mention)
            
            # 构建DocRED实体
            docred_entity = {
                'id': i,  # DocRED使用数字ID
                'text': text,
                'label': self.entity_type_mapping.get(category, 'MISC'),
                'mentions': docred_mentions
            }
            
            docred_entities.append(docred_entity)
        
        return docred_entities
    
    def _convert_relations_to_docred_format(self, relations: List[Dict[str, Any]], 
                                           docred_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换关系为DocRED格式"""
        docred_relations = []
        
        # 构建实体ID映射（从原始ID到DocRED数字ID）
        entity_id_mapping = {}
        for i, entity in enumerate(docred_entities):
            # 尝试匹配原始ID
            original_id = None
            for rel in relations:
                # 简化映射逻辑：按顺序匹配
                if rel.get('head_entity_id') and len([e for e in docred_entities[:i] if e.get('original_id') == rel.get('head_entity_id')]) == 0:
                    pass  # 复杂的映射逻辑需要重构
        
        # 简化的关系转换（假设按顺序匹配）
        for i, relation in enumerate(relations):
            try:
                # 获取关系信息
                relation_type = relation.get('relation_type', 'related_to')
                confidence = relation.get('confidence', 0.0)
                
                # 获取参与者
                head_entity_id = relation.get('head_entity_id', '')
                tail_entity_id = relation.get('tail_entity_id', '')
                
                # 简化映射：按出现顺序分配ID
                head_idx = i % len(docred_entities) if docred_entities else 0
                tail_idx = (i + 1) % len(docred_entities) if len(docred_entities) > 1 else 0
                
                # 转换关系类型
                docred_relation_type = self._map_to_docred_relation_type(relation_type)
                
                # 构建DocRED关系
                docred_relation = {
                    'h': head_idx,  # 头实体索引
                    't': tail_idx,  # 尾实体索引
                    'r': docred_relation_type,  # 关系类型
                    'evidence': relation.get('evidence_sentences', [0])  # 证据句子
                }
                
                docred_relations.append(docred_relation)
                
            except Exception as e:
                logger.warning(f"关系转换失败: {relation}, 错误: {str(e)}")
                continue
        
        return docred_relations
    
    def _map_to_docred_relation_type(self, relation_type: str) -> str:
        """将关系类型映射到DocRED格式"""
        # 直接匹配已知的关系类型
        if relation_type in self.default_docred_relations:
            return self.default_docred_relations[relation_type]
        
        # 基于关键词匹配
        relation_lower = relation_type.lower()
        
        # 工作关系
        if 'work' in relation_lower or 'employ' in relation_lower:
            return '/people/person/affiliation'
        
        # 婚姻关系
        if 'marri' in relation_lower or 'spouse' in relation_lower:
            return '/people/person/spouse_s'
        
        # 家庭关系
        if 'parent' in relation_lower or 'father' in relation_lower or 'mother' in relation_lower:
            return '/people/person/parent'
        
        if 'child' in relation_lower or 'son' in relation_lower or 'daughter' in relation_lower:
            return '/people/person/child'
        
        if 'sibling' in relation_lower or 'brother' in relation_lower or 'sister' in relation_lower:
            return '/people/person/sibling_s'
        
        # 位置关系
        if 'located' in relation_lower or 'place' in relation_lower:
            return '/location/location_contains'
        
        if 'born' in relation_lower or 'birth' in relation_lower:
            return '/people/person/place_of_birth'
        
        if 'die' in relation_lower or 'death' in relation_lower:
            return '/people/person/place_of_death'
        
        # 教育关系
        if 'educat' in relation_lower or 'stud' in relation_lower or 'university' in relation_lower:
            return '/person/person/educated_at'
        
        # 组织关系
        if 'found' in relation_lower or 'establish' in relation_lower:
            return '/organization/organization_founded_by'
        
        if 'member' in relation_lower:
            return '/organization/organization_member_of'
        
        # 默认关系
        return '/relation/default'
    
    def _split_into_sentences(self, text: str) -> List[List[str]]:
        """将文本分割为句子（词列表）"""
        if not text:
            return [[]]
        
        # 简单的句子分割
        sentences = re.split(r'[.!?]+', text)
        sentence_words = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 按空格分割词
                words = sentence.split()
                if words:
                    sentence_words.append(words)
        
        # 如果没有句子，返回空列表
        if not sentence_words:
            return [[]]
        
        return sentence_words
    
    def save_docred_format(self, docred_element: DocREDElement, output_path: str):
        """保存DocRED格式到文件"""
        try:
            # 转换为字典格式
            docred_dict = asdict(docred_element)
            
            # 保存为JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(docred_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"DocRED格式已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存DocRED格式失败: {str(e)}")
            raise
    
    def load_docred_format(self, file_path: str) -> DocREDElement:
        """从文件加载DocRED格式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                docred_dict = json.load(f)
            
            # 转换为对象
            docred_element = DocREDElement(**docred_dict)
            
            logger.info(f"已从文件加载DocRED格式: {file_path}")
            return docred_element
            
        except Exception as e:
            logger.error(f"加载DocRED格式失败: {str(e)}")
            raise
    
    def validate_docred_format(self, docred_element: DocREDElement) -> List[str]:
        """验证DocRED格式的有效性"""
        issues = []
        
        # 检查必要字段
        if not docred_element.title:
            issues.append("缺少文档标题")
        
        if not docred_element.sents:
            issues.append("缺少句子数据")
        
        if not docred_element.vertexSet:
            issues.append("缺少实体数据")
        
        if not docred_element.labels:
            issues.append("缺少关系数据")
        
        # 检查实体格式
        for i, entity in enumerate(docred_element.vertexSet):
            if 'id' not in entity:
                issues.append(f"实体 {i} 缺少ID")
            if 'text' not in entity:
                issues.append(f"实体 {i} 缺少文本")
            if 'label' not in entity:
                issues.append(f"实体 {i} 缺少标签")
            if 'mentions' not in entity:
                issues.append(f"实体 {i} 缺少提及信息")
        
        # 检查关系格式
        for i, relation in enumerate(docred_element.labels):
            if 'h' not in relation:
                issues.append(f"关系 {i} 缺少头实体索引")
            if 't' not in relation:
                issues.append(f"关系 {i} 缺少尾实体索引")
            if 'r' not in relation:
                issues.append(f"关系 {i} 缺少关系类型")
            
            # 检查索引有效性
            if 'h' in relation and 't' in relation:
                if relation['h'] >= len(docred_element.vertexSet):
                    issues.append(f"关系 {i} 头实体索引超出范围: {relation['h']}")
                if relation['t'] >= len(docred_element.vertexSet):
                    issues.append(f"关系 {i} 尾实体索引超出范围: {relation['t']}")
                if relation['h'] == relation['t']:
                    issues.append(f"关系 {i} 头尾实体相同（自反关系）")
        
        return issues
    
    def extract_statistics(self, docred_element: DocREDElement) -> Dict[str, Any]:
        """提取DocRED文档统计信息"""
        statistics = {
            'document_id': docred_element.title,
            'total_sentences': len(docred_element.sents),
            'total_entities': len(docred_element.vertexSet),
            'total_relations': len(docred_element.labels),
            'entity_types': defaultdict(int),
            'relation_types': defaultdict(int),
            'avg_entities_per_sentence': 0.0,
            'avg_relations_per_entity': 0.0
        }
        
        # 统计实体类型
        for entity in docred_element.vertexSet:
            entity_type = entity.get('label', 'UNKNOWN')
            statistics['entity_types'][entity_type] += 1
        
        # 统计关系类型
        for relation in docred_element.labels:
            relation_type = relation.get('r', 'UNKNOWN')
            statistics['relation_types'][relation_type] += 1
        
        # 计算平均值
        if statistics['total_sentences'] > 0:
            statistics['avg_entities_per_sentence'] = statistics['total_entities'] / statistics['total_sentences']
        
        if statistics['total_entities'] > 0:
            statistics['avg_relations_per_entity'] = statistics['total_relations'] / statistics['total_entities']
        
        # 转换为标准字典
        statistics['entity_types'] = dict(statistics['entity_types'])
        statistics['relation_types'] = dict(statistics['relation_types'])
        
        return statistics