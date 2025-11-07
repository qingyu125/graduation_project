"""
要素提取器模块
从AST中提取实体、关系、事件等结构化要素
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import re
import json
import logging
from collections import defaultdict

from ast_parser import ASTNode, NodeType, PseudoCodeASTParser

logger = logging.getLogger(__name__)


class ElementType(Enum):
    """要素类型枚举"""
    ENTITY = "entity"
    RELATION = "relation"
    EVENT = "event"
    ATTRIBUTE = "attribute"
    CONTEXT = "context"


class EntityCategory(Enum):
    """实体类别枚举"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    TIME = "TIME"
    MISC = "MISC"


class RelationCategory(Enum):
    """关系类别枚举"""
    PHYSICAL = "physical"
    AFFILIATION = "affiliation"
    PERSONAL = "personal"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    FUNCTIONAL = "functional"
    UNKNOWN = "unknown"


@dataclass
class ExtractedElement:
    """提取的要素数据结构"""
    element_type: ElementType
    category: str
    content: str
    confidence: float
    line_number: int
    context: str
    metadata: Dict[str, Any]
    evidence_sentences: List[int]
    start_position: int
    end_position: int


@dataclass
class ExtractedEntity:
    """提取的实体数据结构"""
    id: str
    text: str
    category: EntityCategory
    mentions: List[Dict[str, Any]]
    context: str
    confidence: float


@dataclass
class ExtractedRelation:
    """提取的关系数据结构"""
    id: str
    relation_type: str
    head_entity_id: str
    tail_entity_id: str
    category: RelationCategory
    context: str
    evidence_sentences: List[int]
    confidence: float


class ElementExtractor:
    """要素提取器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.max_entities = self.config.get('max_entities', 100)
        self.max_relations = self.config.get('max_relations', 100)
        
        # 初始化AST解析器
        self.ast_parser = PseudoCodeASTParser()
        
        # 实体识别规则
        self.entity_rules = {
            EntityCategory.PERSON: [
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:said|stated|announced|claimed|argued)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:CEO|president|director|manager|chairman)\b'
            ],
            EntityCategory.ORGANIZATION: [
                r'\b[A-Z][A-Z][A-Za-z]*(?:\s+[A-Z][A-Z][A-Za-z]*)*\b',
                r'\b(?:Inc|Corp|Ltd|LLC|Company|Organization)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|Institute)\b'
            ],
            EntityCategory.LOCATION: [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|State|Country|Street|Road|Avenue)\b',
                r'\b(?:United States|USA|China|Japan|Germany|UK|France|Italy|Spain)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b'
            ],
            EntityCategory.TIME: [
                r'\b\d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
            ]
        }
        
        # 关系识别规则
        self.relation_rules = {
            RelationCategory.PERSONAL: [
                r'(\w+)\s+(?:and|&)\s+(\w+)\s+(?:are|is)\s+(?:married|coupled|related|family)',
                r'(\w+)\s+(?:works?|works?)\s+for\s+(\w+)',
                r'(\w+)\s+(?:is|are)\s+(?:father|mother|son|daughter|brother|sister)\s+of\s+(\w+)'
            ],
            RelationCategory.AFFILIATION: [
                r'(\w+)\s+(?:works?|works?)\s+for\s+(\w+)',
                r'(\w+)\s+(?:studied|study)\s+at\s+(\w+)',
                r'(\w+)\s+(?:is|are)\s+(?:member|employee)\s+of\s+(\w+)'
            ],
            RelationCategory.PHYSICAL: [
                r'(\w+)\s+(?:located|situated)\s+in\s+(\w+)',
                r'(\w+)\s+(?:near|close to)\s+(\w+)',
                r'(\w+)\s+(?:part of)\s+(\w+)'
            ],
            RelationCategory.TEMPORAL: [
                r'(\w+)\s+(?:before|after|at the same time as)\s+(\w+)',
                r'(\w+)\s+(?:occurred|occurs)\s+(?:before|after)\s+(\w+)',
                r'(\w+)\s+(?:duration|lasts?)\s+(?:for\s+)?(\w+)'
            ],
            RelationCategory.CAUSAL: [
                r'(\w+)\s+(?:causes?|caused)\s+(\w+)',
                r'(\w+)\s+(?:affects?|effected)\s+(\w+)',
                r'(\w+)\s+(?:leads? to|result in)\s+(\w+)'
            ]
        }
    
    def extract_elements_from_pseudocode(self, pseudocode: str, document_id: str) -> Dict[str, Any]:
        """
        从伪代码中提取所有要素
        
        Args:
            pseudocode: 伪代码文本
            document_id: 文档ID
            
        Returns:
            包含提取结果的字典
        """
        try:
            logger.info(f"开始从伪代码中提取要素，文档ID: {document_id}")
            
            # 解析AST
            ast_root = self.ast_parser.parse_pseudocode(pseudocode)
            
            # 提取各类要素
            entities = self.extract_entities(ast_root, document_id)
            relations = self.extract_relations(ast_root, document_id, entities)
            events = self.extract_events(ast_root, document_id)
            attributes = self.extract_attributes(ast_root, document_id)
            
            # 构建结果
            result = {
                'document_id': document_id,
                'extracted_entities': [asdict(entity) for entity in entities],
                'extracted_relations': [asdict(relation) for relation in relations],
                'extracted_events': events,
                'extracted_attributes': attributes,
                'statistics': self._generate_statistics(entities, relations, events, attributes),
                'metadata': {
                    'extraction_method': 'ast_based',
                    'min_confidence': self.min_confidence,
                    'total_elements': len(entities) + len(relations) + len(events) + len(attributes)
                }
            }
            
            logger.info(f"要素提取完成 - 实体: {len(entities)}, 关系: {len(relations)}, 事件: {len(events)}, 属性: {len(attributes)}")
            return result
            
        except Exception as e:
            logger.error(f"要素提取失败: {str(e)}")
            raise
    
    def extract_entities(self, ast_root: ASTNode, document_id: str) -> List[ExtractedEntity]:
        """从AST中提取实体"""
        entities = []
        seen_entities = set()
        
        def traverse_node(node: ASTNode):
            if node.node_type == NodeType.VARIABLE and node.metadata.get('is_entity'):
                entity = self._extract_entity_from_node(node, document_id)
                if entity and entity.text not in seen_entities:
                    entities.append(entity)
                    seen_entities.add(entity.text)
            
            elif node.node_type == NodeType.RELATION:
                # 从关系节点中提取实体
                participants = node.metadata.get('participants', [])
                for participant in participants:
                    if participant not in seen_entities:
                        entity = self._create_entity_from_text(participant, document_id)
                        if entity:
                            entities.append(entity)
                            seen_entities.add(participant)
            
            elif node.node_type in [NodeType.IDENTIFIER, NodeType.STRING_LITERAL]:
                # 从文本中识别实体
                detected_entities = self._detect_entities_in_text(node.content)
                for entity_text in detected_entities:
                    if entity_text not in seen_entities:
                        entity = self._create_entity_from_text(entity_text, document_id)
                        if entity:
                            entities.append(entity)
                            seen_entities.add(entity_text)
            
            # 递归遍历子节点
            for child in node.children:
                traverse_node(child)
        
        traverse_node(ast_root)
        return entities
    
    def extract_relations(self, ast_root: ASTNode, document_id: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """从AST中提取关系"""
        relations = []
        entity_map = {entity.text: entity.id for entity in entities}
        seen_relations = set()
        
        def traverse_node(node: ASTNode):
            if node.node_type == NodeType.RELATION:
                relation = self._extract_relation_from_node(node, entity_map, document_id)
                if relation and self._is_valid_relation(relation):
                    relation_key = f"{relation.head_entity_id}_{relation.tail_entity_id}_{relation.relation_type}"
                    if relation_key not in seen_relations:
                        relations.append(relation)
                        seen_relations.add(relation_key)
            
            # 递归遍历子节点
            for child in node.children:
                traverse_node(child)
        
        traverse_node(ast_root)
        return relations
    
    def extract_events(self, ast_root: ASTNode, document_id: str) -> List[Dict[str, Any]]:
        """从AST中提取事件"""
        events = []
        
        def traverse_node(node: ASTNode):
            if self._is_event_node(node):
                event = self._extract_event_from_node(node, document_id)
                if event:
                    events.append(event)
            
            # 递归遍历子节点
            for child in node.children:
                traverse_node(child)
        
        traverse_node(ast_root)
        return events
    
    def extract_attributes(self, ast_root: ASTNode, document_id: str) -> List[Dict[str, Any]]:
        """从AST中提取属性"""
        attributes = []
        
        def traverse_node(node: ASTNode):
            if node.node_type == NodeType.VARIABLE:
                attribute = self._extract_attribute_from_node(node, document_id)
                if attribute:
                    attributes.append(attribute)
            
            # 递归遍历子节点
            for child in node.children:
                traverse_node(child)
        
        traverse_node(ast_root)
        return attributes
    
    def _extract_entity_from_node(self, node: ASTNode, document_id: str) -> Optional[ExtractedEntity]:
        """从节点中提取实体"""
        value = node.metadata.get('value', '')
        entity_type = node.metadata.get('entity_type')
        
        if entity_type:
            entity_id = f"{document_id}_entity_{len(node.metadata.get('entities', []))}"
            mentions = [{
                'text': value,
                'sent_idx': 0,
                'pos': [0, len(value)],
                'type': entity_type.lower()
            }]
            
            return ExtractedEntity(
                id=entity_id,
                text=value,
                category=EntityCategory(entity_type),
                mentions=mentions,
                context=node.content,
                confidence=0.8
            )
        
        return None
    
    def _create_entity_from_text(self, text: str, document_id: str) -> Optional[ExtractedEntity]:
        """从文本创建实体"""
        # 检测实体类型
        category = self._detect_entity_type(text)
        if not category:
            return None
        
        entity_id = f"{document_id}_entity_{hash(text) % 10000}"
        mentions = [{
            'text': text,
            'sent_idx': 0,
            'pos': [0, len(text)],
            'type': category.value.lower()
        }]
        
        return ExtractedEntity(
            id=entity_id,
            text=text,
            category=category,
            mentions=mentions,
            context=f"实体: {text}",
            confidence=0.7
        )
    
    def _extract_relation_from_node(self, node: ASTNode, entity_map: Dict[str, str], document_id: str) -> Optional[ExtractedRelation]:
        """从节点中提取关系"""
        participants = node.metadata.get('participants', [])
        relation_type = node.metadata.get('relation_type', 'related_to')
        
        if len(participants) >= 2:
            head_entity = participants[0]
            tail_entity = participants[1]
            
            # 查找对应的实体ID
            head_id = entity_map.get(head_entity)
            tail_id = entity_map.get(tail_entity)
            
            if head_id and tail_id:
                relation_id = f"{document_id}_rel_{hash(node.content) % 10000}"
                category = self._classify_relation_type(relation_type)
                
                return ExtractedRelation(
                    id=relation_id,
                    relation_type=relation_type,
                    head_entity_id=head_id,
                    tail_entity_id=tail_id,
                    category=category,
                    context=node.content,
                    evidence_sentences=[0],
                    confidence=0.8
                )
        
        return None
    
    def _extract_event_from_node(self, node: ASTNode, document_id: str) -> Optional[Dict[str, Any]]:
        """从节点中提取事件"""
        if self._is_event_node(node):
            event_id = f"{document_id}_event_{hash(node.content) % 10000}"
            
            return {
                'id': event_id,
                'text': node.content,
                'type': 'event',
                'trigger': node.content.split()[0] if node.content.split() else node.content,
                'arguments': [],
                'confidence': 0.6,
                'context': node.content,
                'metadata': node.metadata
            }
        
        return None
    
    def _extract_attribute_from_node(self, node: ASTNode, document_id: str) -> Optional[Dict[str, Any]]:
        """从节点中提取属性"""
        var_name = node.metadata.get('variable', '')
        value = node.metadata.get('value', '')
        
        if var_name and value:
            return {
                'id': f"{document_id}_attr_{hash(var_name) % 10000}",
                'entity': var_name,
                'attribute': value,
                'type': 'attribute',
                'confidence': 0.7,
                'context': node.content
            }
        
        return None
    
    def _detect_entities_in_text(self, text: str) -> List[str]:
        """在文本中检测实体"""
        entities = []
        
        for category, patterns in self.entity_rules.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities.extend(matches)
        
        return list(set(entities))
    
    def _detect_entity_type(self, text: str) -> Optional[EntityCategory]:
        """检测实体类型"""
        for category, patterns in self.entity_rules.items():
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return category
        return None
    
    def _classify_relation_type(self, relation_type: str) -> RelationCategory:
        """分类关系类型"""
        relation_lower = relation_type.lower()
        
        for category, patterns in self.relation_rules.items():
            for pattern in patterns:
                if any(word in relation_lower for word in pattern.split()):
                    return category
        
        return RelationCategory.UNKNOWN
    
    def _is_event_node(self, node: ASTNode) -> bool:
        """判断是否为事件节点"""
        event_keywords = ['event', 'happens', 'occurs', 'takes place', 'happening']
        content_lower = node.content.lower()
        
        return any(keyword in content_lower for keyword in event_keywords) or \
               node.node_type == NodeType.ACTION
    
    def _is_valid_relation(self, relation: ExtractedRelation) -> bool:
        """验证关系有效性"""
        return (relation.confidence >= self.min_confidence and
                relation.head_entity_id != relation.tail_entity_id and
                len(relation.context.strip()) > 0)
    
    def _generate_statistics(self, entities: List[ExtractedEntity], 
                           relations: List[ExtractedRelation],
                           events: List[Dict], 
                           attributes: List[Dict]) -> Dict[str, Any]:
        """生成统计信息"""
        entity_categories = defaultdict(int)
        relation_categories = defaultdict(int)
        
        for entity in entities:
            entity_categories[entity.category.value] += 1
        
        for relation in relations:
            relation_categories[relation.category.value] += 1
        
        return {
            'total_entities': len(entities),
            'total_relations': len(relations),
            'total_events': len(events),
            'total_attributes': len(attributes),
            'entity_distribution': dict(entity_categories),
            'relation_distribution': dict(relation_categories),
            'avg_confidence': {
                'entities': sum(e.confidence for e in entities) / len(entities) if entities else 0,
                'relations': sum(r.confidence for r in relations) / len(relations) if relations else 0
            }
        }