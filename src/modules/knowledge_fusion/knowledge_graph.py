"""
知识图谱加载和查询功能模块
支持DocRED数据集的知识图谱构建、查询和验证功能
"""

import json
import logging
import sqlite3
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """知识图谱节点数据结构"""
    node_id: str
    entity_name: str
    entity_type: str
    aliases: List[str]
    properties: Dict[str, Any]
    confidence: float
    source: str
    
@dataclass
class KnowledgeEdge:
    """知识图谱边（关系）数据结构"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    source: str
    evidence: List[str]
    temporal_info: Optional[Dict[str, Any]] = None

class KnowledgeGraph:
    """知识图谱管理类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化知识图谱"""
        self.config = config or {}
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        self.entity_index: Dict[str, Set[str]] = {}  # 实体名称到节点ID的映射
        self.relation_index: Dict[str, List[Tuple[str, str]]] = {}  # 关系类型到边的映射
        
        # 配置参数
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        self.enable_temporal_reasoning = self.config.get('enable_temporal_reasoning', True)
        
        logger.info("知识图谱模块初始化完成")
    
    def load_docred_knowledge(self, docred_data_path: str) -> bool:
        """从DocRED数据构建知识图谱"""
        try:
            logger.info(f"开始从 {docred_data_path} 加载DocRED知识图谱")
            
            with open(docred_data_path, 'r', encoding='utf-8') as f:
                docred_data = json.load(f)
            
            nodes_processed = 0
            edges_processed = 0
            
            for document in docred_data:
                document_id = document.get('title', f'doc_{hashlib.md5(document.get("text", "").encode()).hexdigest()[:8]}')
                
                # 处理实体（从vertexSet中提取）
                for entity_set in document.get('vertexSet', []):
                    for entity in entity_set:
                        node_id = f"{document_id}_{entity['id']}"
                        entity_name = entity.get('name', '')
                        entity_type = self._infer_entity_type(entity)
                        
                        if entity_name:
                            node = KnowledgeNode(
                                node_id=node_id,
                                entity_name=entity_name,
                                entity_type=entity_type,
                                aliases=[],
                                properties={
                                    'positions': entity.get('pos', []),
                                    'sentences': [document['sents'][pos[0]][pos[1]:pos[2]] for pos in entity.get('pos', [])],
                                    'document_id': document_id
                                },
                                confidence=1.0,
                                source='docred'
                            )
                            
                            self.add_node(node)
                            nodes_processed += 1
                
                # 处理关系（从labels中提取）
                for label in document.get('labels', []):
                    h_id = label.get('h', -1)
                    t_id = label.get('t', -1)
                    r_id = label.get('r', -1)
                    
                    if h_id >= 0 and t_id >= 0 and r_id >= 0:
                        source_node = f"{document_id}_{h_id}"
                        target_node = f"{document_id}_{t_id}"
                        relation_type = f"relation_{r_id}"
                        
                        edge = KnowledgeEdge(
                            source_id=source_node,
                            target_id=target_node,
                            relation_type=relation_type,
                            confidence=0.8,
                            source='docred',
                            evidence=[],
                            temporal_info=None
                        )
                        
                        self.add_edge(edge)
                        edges_processed += 1
            
            logger.info(f"DocRED知识图谱加载完成: 节点={nodes_processed}, 边={edges_processed}")
            return True
            
        except Exception as e:
            logger.error(f"加载DocRED知识图谱失败: {e}")
            return False
    
    def add_node(self, node: KnowledgeNode) -> bool:
        """添加节点到知识图谱"""
        try:
            self.graph.add_node(node.node_id, **asdict(node))
            self.nodes[node.node_id] = node
            
            # 更新实体索引
            if node.entity_name not in self.entity_index:
                self.entity_index[node.entity_name] = set()
            self.entity_index[node.entity_name].add(node.node_id)
            
            return True
        except Exception as e:
            logger.error(f"添加节点失败 {node.node_id}: {e}")
            return False
    
    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """添加边到知识图谱"""
        try:
            edge_id = (edge.source_id, edge.target_id)
            self.graph.add_edge(edge.source_id, edge.target_id, **asdict(edge))
            self.edges[edge_id] = edge
            
            # 更新关系索引
            if edge.relation_type not in self.relation_index:
                self.relation_index[edge.relation_type] = []
            self.relation_index[edge.relation_type].append((edge.source_id, edge.target_id))
            
            return True
        except Exception as e:
            logger.error(f"添加边失败 {edge.source_id} -> {edge.target_id}: {e}")
            return False
    
    def query_entities(self, entity_name: str, entity_type: Optional[str] = None) -> List[KnowledgeNode]:
        """根据名称和类型查询实体"""
        try:
            results = []
            
            # 直接名称匹配
            if entity_name in self.entity_index:
                for node_id in self.entity_index[entity_name]:
                    node = self.nodes[node_id]
                    if entity_type is None or node.entity_type == entity_type:
                        results.append(node)
            
            # 模糊匹配（包含关系）
            for name, node_ids in self.entity_index.items():
                if entity_name.lower() in name.lower() or name.lower() in entity_name.lower():
                    for node_id in node_ids:
                        node = self.nodes[node_id]
                        if entity_type is None or node.entity_type == entity_type:
                            if node not in results:  # 避免重复
                                results.append(node)
            
            logger.info(f"实体查询结果: {entity_name} -> {len(results)}个实体")
            return results
            
        except Exception as e:
            logger.error(f"实体查询失败 {entity_name}: {e}")
            return []
    
    def query_relations(self, source_id: Optional[str] = None, 
                       target_id: Optional[str] = None, 
                       relation_type: Optional[str] = None) -> List[KnowledgeEdge]:
        """根据条件查询关系"""
        try:
            results = []
            
            for edge in self.edges.values():
                if source_id and edge.source_id != source_id:
                    continue
                if target_id and edge.target_id != target_id:
                    continue
                if relation_type and edge.relation_type != relation_type:
                    continue
                results.append(edge)
            
            logger.info(f"关系查询结果: {len(results)}个关系")
            return results
            
        except Exception as e:
            logger.error(f"关系查询失败: {e}")
            return []
    
    def validate_entity_consistency(self, entities: List[Dict]) -> Dict[str, Any]:
        """验证实体一致性"""
        try:
            validation_results = {
                'total_entities': len(entities),
                'consistent_entities': 0,
                'inconsistent_entities': 0,
                'validation_details': []
            }
            
            for entity in entities:
                entity_name = entity.get('name', '')
                entity_type = entity.get('type', '')
                confidence = entity.get('confidence', 0.0)
                
                # 查询知识图谱中的匹配实体
                matched_entities = self.query_entities(entity_name, entity_type)
                
                if matched_entities:
                    # 计算一致性得分
                    consistency_score = self._calculate_consistency_score(entity, matched_entities)
                    validation_details = {
                        'input_entity': entity,
                        'matched_entities': [asdict(me) for me in matched_entities],
                        'consistency_score': consistency_score,
                        'is_consistent': consistency_score >= self.confidence_threshold
                    }
                    validation_results['validation_details'].append(validation_details)
                    
                    if consistency_score >= self.confidence_threshold:
                        validation_results['consistent_entities'] += 1
                    else:
                        validation_results['inconsistent_entities'] += 1
                else:
                    # 新实体
                    validation_details = {
                        'input_entity': entity,
                        'matched_entities': [],
                        'consistency_score': 0.0,
                        'is_consistent': True,  # 新实体默认一致
                        'is_new': True
                    }
                    validation_results['validation_details'].append(validation_details)
                    validation_results['consistent_entities'] += 1
            
            logger.info(f"实体一致性验证完成: 总数={validation_results['total_entities']}, "
                       f"一致={validation_results['consistent_entities']}, "
                       f"不一致={validation_results['inconsistent_entities']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"实体一致性验证失败: {e}")
            return {'error': str(e)}
    
    def complete_missing_relations(self, entities: List[Dict], 
                                 relations: List[Dict]) -> List[Dict]:
        """补充缺失的关系"""
        try:
            completed_relations = relations.copy()
            
            # 构建实体映射
            entity_map = {entity.get('name', ''): entity for entity in entities}
            
            # 查找可能的缺失关系
            for source_entity in entities:
                for target_entity in entities:
                    if source_entity == target_entity:
                        continue
                    
                    # 检查是否已存在关系
                    existing_relation = self._find_existing_relation(
                        relations, source_entity, target_entity)
                    
                    if not existing_relation:
                        # 查询知识图谱中的类似关系
                        knowledge_relations = self._query_knowledge_relations(
                            source_entity, target_entity)
                        
                        for knowledge_relation in knowledge_relations:
                            if knowledge_relation['confidence'] >= self.confidence_threshold:
                                completed_relations.append(knowledge_relation)
            
            logger.info(f"关系补充完成: {len(relations)} -> {len(completed_relations)}个关系")
            return completed_relations
            
        except Exception as e:
            logger.error(f"关系补充失败: {e}")
            return relations
    
    def get_entity_context(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """获取实体的上下文信息"""
        try:
            matched_entities = self.query_entities(entity_name)
            if not matched_entities:
                return {'error': f'未找到实体: {entity_name}'}
            
            # 选择最匹配的实体（第一个）
            entity = matched_entities[0]
            context = {
                'entity': asdict(entity),
                'neighbors': {},
                'paths': []
            }
            
            # 获取邻居实体
            for depth in range(1, max_depth + 1):
                depth_key = f'depth_{depth}'
                context['neighbors'][depth_key] = []
                
                if depth == 1:
                    # 直接邻居
                    neighbors = self.graph.neighbors(entity.node_id)
                    for neighbor_id in neighbors:
                        neighbor = self.nodes[neighbor_id]
                        edge = self.edges.get((entity.node_id, neighbor_id))
                        context['neighbors'][depth_key].append({
                            'entity': asdict(neighbor),
                            'relation': asdict(edge) if edge else None
                        })
                else:
                    # 多跳路径
                    paths = self._find_paths(entity.node_id, depth, max_depth)
                    for path in paths:
                        path_info = self._format_path(path, max_depth)
                        if path_info:
                            context['paths'].append(path_info)
            
            return context
            
        except Exception as e:
            logger.error(f"获取实体上下文失败 {entity_name}: {e}")
            return {'error': str(e)}
    
    def _infer_entity_type(self, entity: Dict) -> str:
        """推断实体类型"""
        # 简化的实体类型推断逻辑
        name = entity.get('name', '').lower()
        if any(word in name for word in ['人', '先生', '女士', '博士', '教授']):
            return 'PERSON'
        elif any(word in name for word in ['公司', '组织', '机构', '大学', '政府']):
            return 'ORG'
        elif any(word in name for word in ['城市', '国家', '地区', '省', '州']):
            return 'LOC'
        else:
            return 'MISC'
    
    def _calculate_consistency_score(self, input_entity: Dict, 
                                   matched_entities: List[KnowledgeNode]) -> float:
        """计算实体一致性得分"""
        if not matched_entities:
            return 0.0
        
        max_score = 0.0
        
        for matched_entity in matched_entities:
            score = 0.0
            
            # 名称完全匹配
            if input_entity.get('name', '').lower() == matched_entity.entity_name.lower():
                score += 0.8
            
            # 类型匹配
            if input_entity.get('type', '') == matched_entity.entity_type:
                score += 0.2
            
            # 置信度权重
            score *= matched_entity.confidence
            
            max_score = max(max_score, score)
        
        return max_score
    
    def _find_existing_relation(self, relations: List[Dict], 
                              source_entity: Dict, target_entity: Dict) -> Optional[Dict]:
        """查找已存在的关系"""
        for relation in relations:
            if (relation.get('source_name') == source_entity.get('name') and
                relation.get('target_name') == target_entity.get('name')):
                return relation
        return None
    
    def _query_knowledge_relations(self, source_entity: Dict, 
                                 target_entity: Dict) -> List[Dict]:
        """查询知识图谱中的类似关系"""
        relations = []
        
        # 查询知识图谱中的匹配实体
        source_matches = self.query_entities(source_entity.get('name', ''))
        target_matches = self.query_entities(target_entity.get('name', ''))
        
        for source_match in source_matches:
            for target_match in target_matches:
                # 查询关系
                relations.extend(self.query_relations(
                    source_id=source_match.node_id,
                    target_id=target_match.node_id
                ))
        
        return [asdict(rel) for rel in relations]
    
    def _find_paths(self, start_node: str, min_depth: int, max_depth: int) -> List[List[str]]:
        """查找路径"""
        paths = []
        
        for depth in range(min_depth, max_depth + 1):
            try:
                for target_node in self.nodes:
                    if target_node != start_node:
                        try:
                            path = nx.shortest_path(self.graph, start_node, target_node)
                            if len(path) == depth:
                                paths.append(path)
                        except nx.NetworkXNoPath:
                            continue
            except Exception as e:
                logger.debug(f"路径查找错误: {e}")
                continue
        
        return paths
    
    def _format_path(self, path: List[str], max_depth: int) -> Optional[Dict]:
        """格式化路径信息"""
        if len(path) > max_depth:
            return None
        
        path_info = {
            'nodes': [],
            'relations': [],
            'length': len(path) - 1
        }
        
        for i, node_id in enumerate(path):
            if node_id in self.nodes:
                node = self.nodes[node_id]
                path_info['nodes'].append({
                    'id': node_id,
                    'name': node.entity_name,
                    'type': node.entity_type
                })
                
                # 添加关系信息
                if i < len(path) - 1:
                    next_node_id = path[i + 1]
                    edge = self.edges.get((node_id, next_node_id))
                    if edge:
                        path_info['relations'].append({
                            'from': node.entity_name,
                            'to': self.nodes[next_node_id].entity_name if next_node_id in self.nodes else 'Unknown',
                            'relation': edge.relation_type,
                            'confidence': edge.confidence
                        })
        
        return path_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': len(set(node.entity_type for node in self.nodes.values())),
            'relation_types': len(set(edge.relation_type for edge in self.edges.values())),
            'entity_index_size': len(self.entity_index),
            'relation_index_size': len(self.relation_index),
            'graph_density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph) if self.graph.number_of_nodes() > 0 else 0.0
        }
    
    def save_to_file(self, output_path: str) -> bool:
        """保存知识图谱到文件"""
        try:
            data = {
                'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
                'edges': {f"{src}-{tgt}": asdict(edge) for (src, tgt), edge in self.edges.items()},
                'config': self.config,
                'statistics': self.get_statistics()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"知识图谱保存成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")
            return False
    
    def load_from_file(self, input_path: str) -> bool:
        """从文件加载知识图谱"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建节点
            for node_id, node_data in data.get('nodes', {}).items():
                node = KnowledgeNode(**node_data)
                self.add_node(node)
            
            # 重建边
            for edge_id, edge_data in data.get('edges', {}).items():
                edge = KnowledgeEdge(**edge_data)
                self.add_edge(edge)
            
            # 加载配置
            if 'config' in data:
                self.config.update(data['config'])
            
            logger.info(f"知识图谱加载成功: {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            return False


def create_knowledge_graph(config: Optional[Dict] = None) -> KnowledgeGraph:
    """创建知识图谱实例的工厂函数"""
    return KnowledgeGraph(config)


if __name__ == "__main__":
    # 测试代码
    config = {
        'confidence_threshold': 0.7,
        'max_cache_size': 1000,
        'enable_temporal_reasoning': True
    }
    
    kg = create_knowledge_graph(config)
    
    # 测试实体查询
    test_entities = [
        {'name': '苹果公司', 'type': 'ORG', 'confidence': 0.8},
        {'name': '中国', 'type': 'LOC', 'confidence': 0.9}
    ]
    
    for entity in test_entities:
        results = kg.query_entities(entity['name'], entity['type'])
        print(f"查询实体 {entity['name']}: {len(results)} 个结果")
    
    # 测试一致性验证
    validation_result = kg.validate_entity_consistency(test_entities)
    print(f"验证结果: {validation_result}")
    
    print(f"知识图谱统计: {kg.get_statistics()}")