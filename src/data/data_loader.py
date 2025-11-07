#!/usr/bin/env python3
"""
DocRED数据加载器
处理DocRED数据集的加载、验证和预处理
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ..utils.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体类"""
    id: int
    name: str
    type: str
    mentions: List[Dict[str, Any]]
    confidence: float = 1.0

@dataclass
class Relation:
    """关系类"""
    head_entity_id: int
    tail_entity_id: int
    head_entity_name: str
    tail_entity_name: str
    relation_id: str
    relation_name: str
    evidence_sentences: List[int]
    confidence: float = 1.0

@dataclass
class Document:
    """文档类"""
    doc_id: str
    title: str
    original_text: str
    sentences: List[List[str]]
    entities: List[Entity]
    relations: List[Relation]
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class TrainingSample:
    """训练样本类"""
    doc_id: str
    text: str
    entities: List[Entity]
    relations: List[Relation]
    pseudo_code: Optional[str] = None
    extraction_target: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class DocRedDataLoader:
    """DocRED数据加载器"""
    
    def __init__(self, data_dir: str = None):
        self.config = get_config()
        self.data_config = self.config.data
        self.preprocessing_config = self.config.preprocessing
        
        if data_dir is None:
            self.data_dir = Path(self.data_config.get('raw.docred_path', './data/raw/docred'))
        else:
            self.data_dir = Path(data_dir)
            
        self.rel_info = {}
        self.train_data = []
        self.processed_data = []
        
    def load_rel_info(self) -> Dict[str, str]:
        """加载关系信息"""
        try:
            rel_info_path = self.data_dir / "rel_info.json"
            if not rel_info_path.exists():
                logger.error(f"关系信息文件不存在: {rel_info_path}")
                return {}
                
            with open(rel_info_path, 'r', encoding='utf-8') as f:
                self.rel_info = json.load(f)
                
            logger.info(f"成功加载 {len(self.rel_info)} 个关系类型")
            return self.rel_info
            
        except Exception as e:
            logger.error(f"加载关系信息失败: {e}")
            return {}
    
    def load_train_data(self) -> List[Dict[str, Any]]:
        """加载训练数据"""
        try:
            train_path = self.data_dir / "train_annotated.json"
            if not train_path.exists():
                logger.error(f"训练数据文件不存在: {train_path}")
                return []
                
            self.train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            doc_data = json.loads(line)
                            self.train_data.append(doc_data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"第 {line_num} 行JSON解析失败: {e}")
                            continue
                            
            logger.info(f"成功加载 {len(self.train_data)} 条训练数据")
            return self.train_data
            
        except Exception as e:
            logger.error(f"加载训练数据失败: {e}")
            return []
    
    def validate_docred_format(self, doc_data: Dict[str, Any]) -> bool:
        """验证DocRED数据格式"""
        required_fields = ['title', 'sents', 'vertexSet', 'labels']
        
        for field in required_fields:
            if field not in doc_data:
                logger.warning(f"缺少必需字段: {field}")
                return False
        
        # 验证句子格式
        if not isinstance(doc_data['sents'], list):
            logger.warning("sents字段应为列表")
            return False
            
        # 验证实体格式
        if not isinstance(doc_data['vertexSet'], list):
            logger.warning("vertexSet字段应为列表")
            return False
            
        # 验证关系格式
        if not isinstance(doc_data['labels'], list):
            logger.warning("labels字段应为列表")
            return False
            
        return True
    
    def convert_to_document(self, doc_data: Dict[str, Any], doc_id: str) -> Optional[Document]:
        """将DocRED数据转换为Document对象"""
        try:
            if not self.validate_docred_format(doc_data):
                return None
                
            title = doc_data.get('title', '')
            sents = doc_data.get('sents', [])
            vertex_set = doc_data.get('vertexSet', [])
            labels = doc_data.get('labels', [])
            
            # 重建完整文本
            original_text = ' '.join([' '.join(sent) for sent in sents])
            
            # 转换实体
            entities = []
            for i, entity_mentions in enumerate(vertex_set):
                if not entity_mentions:
                    continue
                    
                # 取第一个提及作为主要实体
                main_mention = entity_mentions[0]
                entity_name = main_mention.get('name', f'Entity_{i}')
                entity_type = main_mention.get('type', 'UNKNOWN')
                
                entity = Entity(
                    id=i,
                    name=entity_name,
                    type=entity_type,
                    mentions=entity_mentions
                )
                entities.append(entity)
            
            # 转换关系
            relations = []
            for label in labels:
                h = label.get('h', -1)
                t = label.get('t', -1)
                r = label.get('r', '')
                evidence = label.get('evidence', [])
                
                # 验证实体ID有效性
                if h < 0 or h >= len(entities) or t < 0 or t >= len(entities):
                    continue
                    
                head_entity = entities[h]
                tail_entity = entities[t]
                relation_name = self.rel_info.get(r, 'Unknown')
                
                relation = Relation(
                    head_entity_id=h,
                    tail_entity_id=t,
                    head_entity_name=head_entity.name,
                    tail_entity_name=tail_entity.name,
                    relation_id=r,
                    relation_name=relation_name,
                    evidence_sentences=evidence
                )
                relations.append(relation)
            
            document = Document(
                doc_id=doc_id,
                title=title,
                original_text=original_text,
                sentences=sents,
                entities=entities,
                relations=relations,
                raw_data=doc_data
            )
            
            return document
            
        except Exception as e:
            logger.error(f"转换文档失败: {e}")
            return None
    
    def process_all_documents(self) -> List[Document]:
        """处理所有文档"""
        self.processed_data = []
        
        for i, doc_data in enumerate(self.train_data):
            document = self.convert_to_document(doc_data, f"doc_{i:06d}")
            if document:
                self.processed_data.append(document)
        
        logger.info(f"成功处理 {len(self.processed_data)} 个文档")
        return self.processed_data
    
    def split_data(self, 
                   data: List[Document], 
                   train_ratio: float = None,
                   val_ratio: float = None, 
                   test_ratio: float = None) -> Tuple[List[Document], List[Document], List[Document]]:
        """划分数据集"""
        if train_ratio is None:
            train_ratio = self.preprocessing_config.train_ratio
        if val_ratio is None:
            val_ratio = self.preprocessing_config.val_ratio
        if test_ratio is None:
            test_ratio = self.preprocessing_config.test_ratio
            
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # 随机打乱
        import random
        random.seed(42)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        logger.info(f"数据划分完成:")
        logger.info(f"  训练集: {len(train_data)} 文档 ({len(train_data)/total_size:.1%})")
        logger.info(f"  验证集: {len(val_data)} 文档 ({len(val_data)/total_size:.1%})")
        logger.info(f"  测试集: {len(test_data)} 文档 ({len(test_data)/total_size:.1%})")
        
        return train_data, val_data, test_data
    
    def analyze_data_statistics(self, data: List[Document]) -> Dict[str, Any]:
        """分析数据统计信息"""
        stats = {
            'total_documents': len(data),
            'total_entities': sum(len(doc.entities) for doc in data),
            'total_relations': sum(len(doc.relations) for doc in data),
            'entity_types': {},
            'relation_types': {},
            'doc_lengths': {
                'sentences': [],
                'entities': [],
                'relations': []
            }
        }
        
        for doc in data:
            # 文档长度统计
            stats['doc_lengths']['sentences'].append(len(doc.sentences))
            stats['doc_lengths']['entities'].append(len(doc.entities))
            stats['doc_lengths']['relations'].append(len(doc.relations))
            
            # 实体类型统计
            for entity in doc.entities:
                entity_type = entity.type
                stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
            
            # 关系类型统计
            for relation in doc.relations:
                rel_type = relation.relation_name
                stats['relation_types'][rel_type] = stats['relation_types'].get(rel_type, 0) + 1
        
        # 计算统计指标
        for length_type in stats['doc_lengths']:
            lengths = stats['doc_lengths'][length_type]
            if lengths:
                stats['doc_lengths'][f'{length_type}_stats'] = {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'min': np.min(lengths),
                    'max': np.max(lengths),
                    'median': np.median(lengths)
                }
        
        return stats
    
    def filter_data_by_criteria(self, 
                               data: List[Document], 
                               min_entities: int = 1,
                               min_relations: int = 0,
                               max_entities: int = None,
                               max_relations: int = None) -> List[Document]:
        """根据条件过滤数据"""
        if max_entities is None:
            max_entities = self.preprocessing_config.max_entities
        if max_relations is None:
            max_relations = self.preprocessing_config.max_relations
            
        filtered_data = []
        
        for doc in data:
            entity_count = len(doc.entities)
            relation_count = len(doc.relations)
            
            if (entity_count >= min_entities and 
                relation_count >= min_relations and
                entity_count <= max_entities and
                relation_count <= max_relations):
                filtered_data.append(doc)
        
        logger.info(f"过滤前: {len(data)} 文档，过滤后: {len(filtered_data)} 文档")
        return filtered_data
    
    def save_processed_data(self, 
                           data: List[Document], 
                           output_path: str) -> bool:
        """保存处理后的数据"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化的格式
            serializable_data = []
            for doc in data:
                doc_dict = {
                    'doc_id': doc.doc_id,
                    'title': doc.title,
                    'original_text': doc.original_text,
                    'sentences': doc.sentences,
                    'entities': [
                        {
                            'id': e.id,
                            'name': e.name,
                            'type': e.type,
                            'mentions': e.mentions,
                            'confidence': e.confidence
                        } for e in doc.entities
                    ],
                    'relations': [
                        {
                            'head_entity_id': r.head_entity_id,
                            'tail_entity_id': r.tail_entity_id,
                            'head_entity_name': r.head_entity_name,
                            'tail_entity_name': r.tail_entity_name,
                            'relation_id': r.relation_id,
                            'relation_name': r.relation_name,
                            'evidence_sentences': r.evidence_sentences,
                            'confidence': r.confidence
                        } for r in doc.relations
                    ]
                }
                serializable_data.append(doc_dict)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"处理后的数据已保存到: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return False


class RelationExtractionDataset(Dataset):
    """关系抽取数据集"""
    
    def __init__(self, 
                 documents: List[Document], 
                 tokenizer,
                 max_length: int = 512,
                 max_entities: int = 50,
                 max_relations: int = 100):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_entities = max_entities
        self.max_relations = max_relations
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # 截断文本
        text = doc.original_text[:self.max_length*4]  # 粗略估计
        tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length)
        
        # 准备标签
        entity_labels = [0] * self.max_entities
        relation_labels = [0] * self.max_relations
        
        # 实体标签
        for i, entity in enumerate(doc.entities[:self.max_entities]):
            entity_labels[i] = 1  # 假设1表示实体存在
            
        # 关系标签
        for i, relation in enumerate(doc.relations[:self.max_relations]):
            relation_labels[i] = 1  # 假设1表示关系存在
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor([1] * len(tokens), dtype=torch.long),
            'entity_labels': torch.tensor(entity_labels, dtype=torch.long),
            'relation_labels': torch.tensor(relation_labels, dtype=torch.long),
            'doc_id': doc.doc_id,
            'text': doc.original_text,
            'entities': doc.entities,
            'relations': doc.relations
        }


def create_data_loaders(train_docs: List[Document], 
                       val_docs: List[Document], 
                       test_docs: List[Document],
                       tokenizer,
                       batch_size: int = 4,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    config = get_config()
    training_config = config.training
    preprocessing_config = config.preprocessing
    
    # 创建数据集
    train_dataset = RelationExtractionDataset(
        train_docs, tokenizer,
        max_length=training_config.max_seq_length,
        max_entities=preprocessing_config.max_entities,
        max_relations=preprocessing_config.max_relations
    )
    
    val_dataset = RelationExtractionDataset(
        val_docs, tokenizer,
        max_length=training_config.max_seq_length,
        max_entities=preprocessing_config.max_entities,
        max_relations=preprocessing_config.max_relations
    )
    
    test_dataset = RelationExtractionDataset(
        test_docs, tokenizer,
        max_length=training_config.max_seq_length,
        max_entities=preprocessing_config.max_entities,
        max_relations=preprocessing_config.max_relations
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader