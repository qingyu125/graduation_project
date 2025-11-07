#!/usr/bin/env python3
"""
数据预处理器
将DocRED数据转换为统一的训练格式
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import random

from .data_loader import Document, Entity, Relation, TrainingSample

logger = logging.getLogger(__name__)

@dataclass
class DataPreprocessorConfig:
    """数据预处理器配置"""
    min_entities: int = 1
    min_relations: int = 0
    max_entities: int = 50
    max_relations: int = 100
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shuffle_seed: int = 42

class DocRedPreprocessor:
    """DocRED数据预处理器"""
    
    def __init__(self, config: Optional[DataPreprocessorConfig] = None):
        self.config = config or DataPreprocessorConfig()
        self.documents = []
        self.training_samples = []
        
    def load_documents(self, documents: List[Document]):
        """加载文档数据"""
        self.documents = documents
        logger.info(f"加载了 {len(self.documents)} 个文档")
    
    def filter_documents(self, 
                        min_entities: int = None,
                        min_relations: int = None,
                        max_entities: int = None,
                        max_relations: int = None) -> List[Document]:
        """过滤文档"""
        if min_entities is None:
            min_entities = self.config.min_entities
        if min_relations is None:
            min_relations = self.config.min_relations
        if max_entities is None:
            max_entities = self.config.max_entities
        if max_relations is None:
            max_relations = self.config.max_relations
        
        filtered_docs = []
        for doc in self.documents:
            entity_count = len(doc.entities)
            relation_count = len(doc.relations)
            
            if (entity_count >= min_entities and 
                relation_count >= min_relations and
                entity_count <= max_entities and
                relation_count <= max_relations):
                filtered_docs.append(doc)
        
        logger.info(f"过滤前: {len(self.documents)} 文档，过滤后: {len(filtered_docs)} 文档")
        return filtered_docs
    
    def create_training_samples(self, 
                               documents: List[Document],
                               include_metadata: bool = True) -> List[TrainingSample]:
        """创建训练样本"""
        training_samples = []
        
        for doc in documents:
            # 准备实体数据
            entities_data = []
            for entity in doc.entities:
                entity_data = {
                    'id': entity.id,
                    'name': entity.name,
                    'type': entity.type,
                    'mentions': entity.mentions,
                    'confidence': entity.confidence
                }
                entities_data.append(entity_data)
            
            # 准备关系数据
            relations_data = []
            for relation in doc.relations:
                relation_data = {
                    'head_entity_id': relation.head_entity_id,
                    'tail_entity_id': relation.tail_entity_id,
                    'head_entity_name': relation.head_entity_name,
                    'tail_entity_name': relation.tail_entity_name,
                    'relation_id': relation.relation_id,
                    'relation_name': relation.relation_name,
                    'evidence_sentences': relation.evidence_sentences,
                    'confidence': relation.confidence
                }
                relations_data.append(relation_data)
            
            # 创建训练样本
            sample = TrainingSample(
                doc_id=doc.doc_id,
                text=doc.original_text,
                entities=[Entity(**e) for e in entities_data],
                relations=[Relation(**r) for r in relations_data],
                metadata={
                    'title': doc.title,
                    'sentences': doc.sentences,
                    'raw_data': doc.raw_data
                } if include_metadata else None
            )
            
            training_samples.append(sample)
        
        logger.info(f"创建了 {len(training_samples)} 个训练样本")
        return training_samples
    
    def generate_pseudo_code_template(self, sample: TrainingSample) -> str:
        """为样本生成伪代码模板"""
        lines = ["# 自动生成的伪代码模板"]
        lines.append(f"# 文档ID: {sample.doc_id}")
        lines.append(f"# 标题: {sample.metadata['title'] if sample.metadata else 'Unknown'}")
        lines.append("")
        
        # 定义实体
        lines.append("# 定义实体")
        for entity in sample.entities:
            lines.append(f"{entity.name.lower().replace(' ', '_')} = \"{entity.name}\"")
            lines.append(f"{entity.name.lower().replace(' ', '_')}_type = \"{entity.type}\"")
        lines.append("")
        
        # 定义关系
        if sample.relations:
            lines.append("# 定义关系")
            for relation in sample.relations:
                head_var = relation.head_entity_name.lower().replace(' ', '_')
                tail_var = relation.tail_entity_name.lower().replace(' ', '_')
                lines.append(f"relation_{head_var}_{tail_var} = (\"{head_var}\", \"{relation.relation_name}\", \"{tail_var}\")")
            lines.append("")
        
        # 推理步骤
        lines.append("# 推理步骤")
        lines.append("if len(entities) > 0:")
        lines.append("    # 进行实体关系推理")
        lines.append("    for entity in entities:")
        lines.append("        analyze_entity_relations(entity)")
        lines.append("")
        
        return "\n".join(lines)
    
    def create_extraction_target(self, sample: TrainingSample) -> Dict[str, Any]:
        """创建要素抽取目标"""
        target = {
            'entities': [],
            'relations': []
        }
        
        # 实体目标
        for entity in sample.entities:
            entity_target = {
                'id': entity.id,
                'name': entity.name,
                'type': entity.type,
                'required': True
            }
            target['entities'].append(entity_target)
        
        # 关系目标
        for relation in sample.relations:
            relation_target = {
                'head_entity_id': relation.head_entity_id,
                'tail_entity_id': relation.tail_entity_id,
                'head_entity_name': relation.head_entity_name,
                'tail_entity_name': relation.tail_entity_name,
                'relation_id': relation.relation_id,
                'relation_name': relation.relation_name,
                'required': True
            }
            target['relations'].append(relation_target)
        
        return target
    
    def augment_training_data(self, 
                             samples: List[TrainingSample], 
                             augmentation_ratio: float = 0.2) -> List[TrainingSample]:
        """数据增强"""
        augmented_samples = []
        num_augment = int(len(samples) * augmentation_ratio)
        
        # 随机选择样本进行增强
        selected_samples = random.sample(samples, min(num_augment, len(samples)))
        
        for sample in selected_samples:
            augmented_sample = self.augment_sample(sample)
            if augmented_sample:
                augmented_samples.append(augmented_sample)
        
        logger.info(f"生成了 {len(augmented_samples)} 个增强样本")
        return samples + augmented_samples
    
    def augment_sample(self, sample: TrainingSample) -> Optional[TrainingSample]:
        """增强单个样本"""
        try:
            # 简单的文本扰动增强
            original_text = sample.text
            words = original_text.split()
            
            if len(words) < 10:
                return None  # 太短的文本不进行增强
            
            # 随机选择增强方式
            augmentation_type = random.choice(['shuffle', 'dropout', 'synonym'])
            
            if augmentation_type == 'shuffle':
                # 随机打乱部分词语
                shuffled_words = words.copy()
                start_idx = random.randint(0, max(0, len(words) - 10))
                end_idx = min(len(words), start_idx + 10)
                segment = shuffled_words[start_idx:end_idx]
                random.shuffle(segment)
                shuffled_words[start_idx:end_idx] = segment
                augmented_text = ' '.join(shuffled_words)
                
            elif augmentation_type == 'dropout':
                # 随机删除一些词语
                dropout_words = [w for w in words if random.random() > 0.1]
                augmented_text = ' '.join(dropout_words) if dropout_words else original_text
                
            else:  # synonym
                # 保持原样（简化实现）
                augmented_text = original_text
            
            # 创建增强样本
            augmented_sample = TrainingSample(
                doc_id=f"{sample.doc_id}_aug",
                text=augmented_text,
                entities=sample.entities.copy(),
                relations=sample.relations.copy(),
                metadata=sample.metadata.copy() if sample.metadata else None
            )
            
            return augmented_sample
            
        except Exception as e:
            logger.warning(f"增强样本失败: {e}")
            return None
    
    def split_samples(self, 
                     samples: List[TrainingSample],
                     train_ratio: float = None,
                     val_ratio: float = None,
                     test_ratio: float = None) -> Tuple[List[TrainingSample], List[TrainingSample], List[TrainingSample]]:
        """划分训练样本"""
        if train_ratio is None:
            train_ratio = self.config.train_ratio
        if val_ratio is None:
            val_ratio = self.config.val_ratio
        if test_ratio is None:
            test_ratio = self.config.test_ratio
        
        # 设置随机种子
        random.seed(self.config.shuffle_seed)
        
        # 复制并打乱
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        total_size = len(shuffled_samples)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_samples = shuffled_samples[:train_size]
        val_samples = shuffled_samples[train_size:train_size + val_size]
        test_samples = shuffled_samples[train_size + val_size:]
        
        logger.info(f"样本划分完成:")
        logger.info(f"  训练集: {len(train_samples)} 样本 ({len(train_samples)/total_size:.1%})")
        logger.info(f"  验证集: {len(val_samples)} 样本 ({len(val_samples)/total_size:.1%})")
        logger.info(f"  测试集: {len(test_samples)} 样本 ({len(test_samples)/total_size:.1%})")
        
        return train_samples, val_samples, test_samples
    
    def save_samples(self, 
                    samples: List[TrainingSample], 
                    output_path: str,
                    format_type: str = 'json') -> bool:
        """保存样本数据"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type == 'json':
                # 转换为可序列化格式
                serializable_samples = []
                for sample in samples:
                    sample_dict = asdict(sample)
                    serializable_samples.append(sample_dict)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_samples, f, ensure_ascii=False, indent=2)
            
            elif format_type == 'jsonl':
                # JSONL格式（每行一个样本）
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        sample_dict = asdict(sample)
                        f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
            
            else:
                raise ValueError(f"不支持的格式类型: {format_type}")
            
            logger.info(f"样本数据已保存到: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存样本数据失败: {e}")
            return False
    
    def load_samples(self, input_path: str) -> List[TrainingSample]:
        """加载样本数据"""
        try:
            input_file = Path(input_path)
            if not input_file.exists():
                logger.error(f"文件不存在: {input_file}")
                return []
            
            samples = []
            
            if input_path.endswith('.jsonl'):
                # JSONL格式
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                sample_dict = json.loads(line)
                                sample = TrainingSample(**sample_dict)
                                samples.append(sample)
                            except Exception as e:
                                logger.warning(f"第 {line_num} 行加载失败: {e}")
                                continue
            else:
                # JSON格式
                with open(input_file, 'r', encoding='utf-8') as f:
                    samples_data = json.load(f)
                    for sample_dict in samples_data:
                        try:
                            sample = TrainingSample(**sample_dict)
                            samples.append(sample)
                        except Exception as e:
                            logger.warning(f"加载样本失败: {e}")
                            continue
            
            logger.info(f"成功加载 {len(samples)} 个样本")
            return samples
            
        except Exception as e:
            logger.error(f"加载样本数据失败: {e}")
            return []
    
    def create_pseudo_code_dataset(self, 
                                  samples: List[TrainingSample],
                                  output_dir: str = "./data/processed") -> bool:
        """创建伪代码数据集"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 创建伪代码示例
            pseudo_code_data = []
            for sample in samples:
                pseudo_code = self.generate_pseudo_code_template(sample)
                extraction_target = self.create_extraction_target(sample)
                
                pseudo_code_sample = {
                    'doc_id': sample.doc_id,
                    'original_text': sample.text,
                    'pseudo_code': pseudo_code,
                    'extraction_target': extraction_target,
                    'entities': [asdict(e) for e in sample.entities],
                    'relations': [asdict(r) for r in sample.relations]
                }
                pseudo_code_data.append(pseudo_code_sample)
            
            # 保存伪代码数据
            pseudo_code_path = output_path / "pseudo_code_dataset.json"
            with open(pseudo_code_path, 'w', encoding='utf-8') as f:
                json.dump(pseudo_code_data, f, ensure_ascii=False, indent=2)
            
            # 创建提示模板示例
            prompt_examples = []
            for sample in pseudo_code_data[:50]:  # 取前50个作为示例
                prompt_example = {
                    'input': f"文本: {sample['original_text'][:200]}...",
                    'pseudo_code': sample['pseudo_code'],
                    'extraction_target': sample['extraction_target']
                }
                prompt_examples.append(prompt_example)
            
            prompt_path = output_path / "prompt_examples.json"
            with open(prompt_path, 'w', encoding='utf-8') as f:
                json.dump(prompt_examples, f, ensure_ascii=False, indent=2)
            
            logger.info(f"伪代码数据集已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建伪代码数据集失败: {e}")
            return False
    
    def analyze_sample_quality(self, samples: List[TrainingSample]) -> Dict[str, Any]:
        """分析样本质量"""
        if not samples:
            return {}
        
        stats = {
            'total_samples': len(samples),
            'avg_entities_per_sample': 0,
            'avg_relations_per_sample': 0,
            'entity_type_distribution': {},
            'relation_type_distribution': {},
            'text_length_stats': {
                'min': float('inf'),
                'max': 0,
                'total': 0
            }
        }
        
        total_entities = 0
        total_relations = 0
        text_lengths = []
        
        for sample in samples:
            # 统计实体和关系
            num_entities = len(sample.entities)
            num_relations = len(sample.relations)
            total_entities += num_entities
            total_relations += num_relations
            
            # 统计文本长度
            text_length = len(sample.text)
            text_lengths.append(text_length)
            stats['text_length_stats']['min'] = min(stats['text_length_stats']['min'], text_length)
            stats['text_length_stats']['max'] = max(stats['text_length_stats']['max'], text_length)
            stats['text_length_stats']['total'] += text_length
            
            # 统计实体类型
            for entity in sample.entities:
                entity_type = entity.type
                stats['entity_type_distribution'][entity_type] = stats['entity_type_distribution'].get(entity_type, 0) + 1
            
            # 统计关系类型
            for relation in sample.relations:
                rel_type = relation.relation_name
                stats['relation_type_distribution'][rel_type] = stats['relation_type_distribution'].get(rel_type, 0) + 1
        
        # 计算平均值
        stats['avg_entities_per_sample'] = total_entities / len(samples)
        stats['avg_relations_per_sample'] = total_relations / len(samples)
        stats['text_length_stats']['mean'] = stats['text_length_stats']['total'] / len(samples)
        
        # 删除临时字段
        del stats['text_length_stats']['total']
        
        return stats