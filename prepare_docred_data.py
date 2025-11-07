#!/usr/bin/env python3
"""
DocRED数据集准备脚本
用于下载、解压和处理DocRED数据集
"""

import os
import json
# import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocRedDataProcessor:
    """DocRED数据处理器"""
    
    def __init__(self, data_dir: str = "data/raw/docred_sample"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rel_info_path = None
        self.train_annotated_path = None
        
    def download_docred_dataset(self, 
                              docred_url: str = "https://github.com/thunlp/DocRED/archive/refs/heads/master.zip",
                              target_dir: str = "./data/raw") -> bool:
        """
        下载DocRED数据集
        
        Args:
            docred_url: DocRED数据集的下载链接
            target_dir: 目标下载目录
            
        Returns:
            bool: 下载是否成功
        """
        try:
            logger.info("开始下载DocRED数据集...")
            response = requests.get(docred_url, stream=True)
            response.raise_for_status()
            
            zip_path = Path(target_dir) / "docred_master.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("解压DocRED数据集...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # 移动文件到正确位置
            extracted_dir = Path(target_dir) / "DocRED-master"
            if extracted_dir.exists():
                # 移动到目标位置
                import shutil
                if self.data_dir.exists():
                    shutil.rmtree(self.data_dir)
                shutil.move(str(extracted_dir), str(self.data_dir))
                
                # 删除zip文件
                zip_path.unlink()
                
            logger.info(f"DocRED数据集已成功下载到: {self.data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"下载DocRED数据集失败: {e}")
            return False
    
    def setup_file_paths(self):
        """设置文件路径"""
        # 查找rel_info.json文件
        for file_path in self.data_dir.rglob("rel_info.json"):
            self.rel_info_path = file_path
            break
            
        # 查找train_annotated.json文件
        for file_path in self.data_dir.rglob("train_annotated.json"):
            self.train_annotated_path = file_path
            break
            
        if not self.rel_info_path or not self.train_annotated_path:
            logger.error("无法找到必要的DocRED文件，请检查数据集结构")
            return False
            
        logger.info(f"找到rel_info.json: {self.rel_info_path}")
        logger.info(f"找到train_annotated.json: {self.train_annotated_path}")
        return True
    
    def load_rel_info(self) -> Dict[str, str]:
        """
        加载关系信息文件
        
        Returns:
            Dict[str, str]: 关系ID到关系名称的映射
        """
        try:
            with open(self.rel_info_path, 'r', encoding='utf-8') as f:
                rel_info = json.load(f)
            logger.info(f"加载了 {len(rel_info)} 个关系类型")
            return rel_info
        except Exception as e:
            logger.error(f"加载关系信息失败: {e}")
            return {}
    
    def load_train_data(self) -> List[Dict[str, Any]]:
        """
        加载训练数据
        
        Returns:
            List[Dict[str, Any]]: 训练数据列表
        """
        try:
            with open(self.train_annotated_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.error("训练数据文件为空")
                    return []
                
                # 尝试解析为JSON数组
                try:
                    train_data = json.loads(content)
                    if isinstance(train_data, list):
                        logger.info(f"从JSON数组加载了 {len(train_data)} 条训练数据")
                        return train_data
                except json.JSONDecodeError:
                    pass
                
                # 如果不是JSON数组，尝试按行解析
                train_data = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            train_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"跳过无法解析的行: {line[:50]}... (错误: {e})")
                            continue
                
                logger.info(f"按行解析加载了 {len(train_data)} 条训练数据")
                return train_data
                
        except Exception as e:
            logger.error(f"加载训练数据失败: {e}")
            return []
    
    def analyze_dataset(self, train_data: List[Dict[str, Any]], rel_info: Dict[str, str]):
        """
        分析数据集统计信息
        
        Args:
            train_data: 训练数据
            rel_info: 关系信息
        """
        logger.info("=== DocRED数据集分析 ===")
        
        if not train_data:
            logger.error("训练数据为空")
            return
        
        # 验证数据类型
        valid_docs = []
        for i, doc in enumerate(train_data):
            if isinstance(doc, dict):
                valid_docs.append(doc)
            else:
                logger.warning(f"跳过第 {i+1} 条无效数据: {type(doc)}")
        
        if not valid_docs:
            logger.error("没有有效的训练数据")
            return
        
        # 基本统计
        total_docs = len(valid_docs)
        total_entities = sum(len(doc.get('vertexSet', [])) for doc in valid_docs)
        total_relations = sum(len(doc.get('labels', [])) for doc in valid_docs)
        
        logger.info(f"总文档数: {total_docs}")
        logger.info(f"总实体数: {total_entities}")
        logger.info(f"总关系数: {total_relations}")
        
        # 关系类型分布
        relation_counts = {}
        for doc in valid_docs:
            for label in doc.get('labels', []):
                if isinstance(label, dict):
                    rel_type = label.get('r', '')
                    relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
        
        logger.info("=== 关系类型分布 ===")
        for rel_id, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
            rel_name = rel_info.get(rel_id, 'Unknown')
            logger.info(f"{rel_id} ({rel_name}): {count}")
        
        # 文档长度分布
        doc_lengths = []
        for doc in valid_docs:
            sents = doc.get('sents', [])
            sent_count = len(sents) if isinstance(sents, list) else 0
            doc_lengths.append(sent_count)
        
        if doc_lengths:
            logger.info(f"=== 文档长度分布 ===")
            logger.info(f"平均句子数: {sum(doc_lengths) / len(doc_lengths):.2f}")
            logger.info(f"最短文档: {min(doc_lengths)} 句")
            logger.info(f"最长文档: {max(doc_lengths)} 句")
        else:
            logger.warning("没有有效的文档长度数据")
        
    def create_sample_data(self, train_data: List[Dict[str, Any]], 
                          rel_info: Dict[str, Any], 
                          output_path: str = "data/processed/sample_data.json",
                          sample_size: int = 100):
        """
        创建样本数据用于测试
        
        Args:
            train_data: 完整训练数据
            rel_info: 关系信息
            output_path: 输出文件路径
            sample_size: 样本大小
        """
        try:
            # 随机选择样本
            import random
            random.seed(42)
            sample_data = random.sample(train_data, min(sample_size, len(train_data)))
            
            # 添加关系名称
            for doc in sample_data:
                for label in doc.get('labels', []):
                    rel_id = label.get('r', '')
                    label['relation_name'] = rel_info.get(rel_id, 'Unknown')
            
            # 保存样本数据
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"样本数据已保存到: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"创建样本数据失败: {e}")
            return False
    
    def process_docred_format(self, train_data: List[Dict[str, Any]], 
                            rel_info: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        将DocRED格式转换为统一训练格式
        
        Args:
            train_data: DocRED训练数据
            rel_info: 关系信息
            
        Returns:
            List[Dict[str, Any]]: 转换后的训练数据
        """
        processed_data = []
        
        for doc in train_data:
            try:
                # 提取基本信息
                title = doc.get('title', '')
                sents = doc.get('sents', [])
                vertex_set = doc.get('vertexSet', [])
                labels = doc.get('labels', [])
                
                # 构建实体映射
                entity_mapping = {}
                for i, entity_mentions in enumerate(vertex_set):
                    entity_name = entity_mentions[0]['name'] if entity_mentions else f"Entity_{i}"
                    entity_mapping[i] = entity_name
                
                # 重建完整文本
                full_text = ' '.join([' '.join(sent) for sent in sents])
                
                # 提取结构化要素
                entities = []
                for i, entity_mentions in enumerate(vertex_set):
                    entity_name = entity_mapping[i]
                    entity_type = entity_mentions[0].get('type', 'UNKNOWN') if entity_mentions else 'UNKNOWN'
                    entities.append({
                        'id': i,
                        'name': entity_name,
                        'type': entity_type,
                        'mentions': entity_mentions
                    })
                
                # 提取关系
                relations = []
                for label in labels:
                    h = label.get('h', -1)
                    t = label.get('t', -1)
                    r = label.get('r', '')
                    evidence = label.get('evidence', [])
                    
                    if h in entity_mapping and t in entity_mapping:
                        relations.append({
                            'head_entity_id': h,
                            'tail_entity_id': t,
                            'head_entity_name': entity_mapping[h],
                            'tail_entity_name': entity_mapping[t],
                            'relation_id': r,
                            'relation_name': rel_info.get(r, 'Unknown'),
                            'evidence_sentences': evidence
                        })
                
                # 构建处理后的数据
                processed_doc = {
                    'original_format': 'DocRED',
                    'doc_id': len(processed_data),
                    'title': title,
                    'original_text': full_text,
                    'sentences': sents,
                    'entities': entities,
                    'relations': relations,
                    'raw_doc': doc
                }
                
                processed_data.append(processed_doc)
                
            except Exception as e:
                logger.warning(f"处理文档时出错: {e}")
                continue
        
        logger.info(f"处理了 {len(processed_data)} 个文档")
        return processed_data
    
    def split_data(self, data: List[Dict[str, Any]], 
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1, 
                   test_ratio: float = 0.1):
        """
        划分训练集、验证集和测试集
        
        Args:
            data: 完整数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        total_size = len(data)
        if total_size == 0:
            logger.error("数据为空，无法划分")
            return [], [], []
        
        # 确保比例加起来不超过1
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio > 1.0:
            # 重新标准化比例
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
            logger.info("重新标准化数据划分比例")
        
        # 计算各集合大小
        train_size = max(1, int(total_size * train_ratio))
        val_size = max(0, int(total_size * val_ratio))
        test_size = max(1, total_size - train_size - val_size) if total_size > 2 else max(0, total_size - train_size)
        
        # 调整大小以确保总数不变
        if train_size + val_size + test_size != total_size:
            # 优先保证训练集，然后保证测试集
            if train_size >= total_size:
                train_size = total_size
                val_size = 0
                test_size = 0
            elif train_size + val_size >= total_size:
                val_size = total_size - train_size
                test_size = 0
            else:
                test_size = total_size - train_size - val_size
        
        logger.info(f"原始比例: 训练{train_ratio:.1%}, 验证{val_ratio:.1%}, 测试{test_ratio:.1%}")
        logger.info(f"划分大小: 训练{train_size}, 验证{val_size}, 测试{test_size} (总计: {total_size})")
        
        # 随机打乱
        import random
        random.seed(42)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        # 安全地划分数据
        train_data = shuffled_data[:train_size] if train_size > 0 else []
        val_start = train_size
        val_end = val_start + val_size
        test_start = val_end
        
        val_data = shuffled_data[val_start:val_end] if val_size > 0 else []
        test_data = shuffled_data[test_start:] if test_size > 0 else []
        
        logger.info(f"实际划分:")
        logger.info(f"  训练集: {len(train_data)} 样本")
        logger.info(f"  验证集: {len(val_data)} 样本")
        logger.info(f"  测试集: {len(test_data)} 样本")
        
        return train_data, val_data, test_data
    
    def save_splits(self, train_data, val_data, test_data, 
                   output_dir: str = "data/processed"):
        """
        保存数据划分
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            test_data: 测试数据
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存训练集
        train_path = output_path / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # 保存验证集
        val_path = output_path / "val.json"
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # 保存测试集
        test_path = output_path / "test.json"
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到: {output_path}")
    
    def generate_pseudo_code_examples(self, processed_data: List[Dict[str, Any]], 
                                    output_path: str = "data/processed/pseudocode_examples.json",
                                    num_examples: int = 50):
        """
        为要素抽取任务生成伪代码示例
        
        Args:
            processed_data: 处理后的数据
            output_path: 输出路径
            num_examples: 示例数量
        """
        examples = []
        
        for i, doc in enumerate(processed_data[:num_examples]):
            try:
                title = doc.get('title', '')
                text = doc.get('original_text', '')
                entities = doc.get('entities', [])
                relations = doc.get('relations', [])
                
                # 生成伪代码示例
                pseudo_code_lines = ["# 定义实体"]
                for entity in entities:
                    entity_id = entity['id']
                    entity_name = entity['name']
                    entity_type = entity['type']
                    pseudo_code_lines.append(f"entity_{entity_id} = \"{entity_name}\"")
                    pseudo_code_lines.append(f"entity_{entity_id}_type = \"{entity_type}\"")
                
                pseudo_code_lines.append("\n# 定义关系")
                for relation in relations:
                    head_id = relation['head_entity_id']
                    tail_id = relation['tail_entity_id']
                    rel_name = relation['relation_name']
                    pseudo_code_lines.append(f"relationship_{head_id}_{tail_id} = (\"entity_{head_id}\", \"{rel_name}\", \"entity_{tail_id}\")")
                
                pseudo_code = "\n".join(pseudo_code_lines)
                
                example = {
                    'doc_id': i,
                    'title': title,
                    'original_text': text,
                    'entities': entities,
                    'relations': relations,
                    'pseudo_code': pseudo_code,
                    'extraction_target': {
                        'entities': [{'id': e['id'], 'name': e['name'], 'type': e['type']} for e in entities],
                        'relations': [{'head': r['head_entity_name'], 'relation': r['relation_name'], 'tail': r['tail_entity_name']} for r in relations]
                    }
                }
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"生成伪代码示例失败 (doc {i}): {e}")
                continue
        
        # 保存示例
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已生成 {len(examples)} 个伪代码示例，保存到: {output_path}")
        
        return examples


def main():
    """主函数"""
    logger.info("开始处理DocRED真实数据集...")
    
    # 创建数据处理器
    processor = DocRedDataProcessor()
    
    # 设置文件路径
    if not processor.setup_file_paths():
        logger.error("无法找到DocRED数据文件，请确保数据文件在正确位置")
        logger.info(f"预期位置: {processor.data_dir}")
        logger.info("需要文件: train_annotated.json, rel_info.json")
        return
    
    # 加载数据
    rel_info = processor.load_rel_info()
    train_data = processor.load_train_data()
    
    if not rel_info:
        logger.error("关系信息加载失败")
        return
    
    if not train_data:
        logger.error("训练数据加载失败")
        return
    
    if len(train_data) == 0:
        logger.error("训练数据为空")
        return
    
    logger.info(f"成功加载 {len(train_data)} 条训练数据和 {len(rel_info)} 个关系类型")
    
    # 分析数据集
    try:
        processor.analyze_dataset(train_data, rel_info)
    except Exception as e:
        logger.error(f"数据分析失败: {e}")
        # 继续处理，但记录错误
        logger.info("跳过数据分析，继续数据处理...")
    
    # 处理数据格式 (处理真实数据)
    logger.info("正在处理DocRED真实数据...")
    try:
        processed_data = processor.process_docred_format(train_data, rel_info)
    except Exception as e:
        logger.error(f"数据格式转换失败: {e}")
        return
    
    if not processed_data:
        logger.error("数据处理后为空")
        return
    
    # 划分数据 - 根据数据量调整比例
    data_size = len(processed_data)
    if data_size <= 2:
        # 小数据集处理：每个数据作为不同的集合
        train_ratio = 0.5
        val_ratio = 0.3
        test_ratio = 0.2
        logger.info("检测到小数据集，调整数据划分比例")
    else:
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
    
    try:
        train_data_split, val_data_split, test_data_split = processor.split_data(
            processed_data, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )
    except Exception as e:
        logger.error(f"数据划分失败: {e}")
        return
    
    # 保存数据划分到processed目录
    output_dir = "data/processed"
    try:
        processor.save_splits(train_data_split, val_data_split, test_data_split, output_dir)
    except Exception as e:
        logger.error(f"保存数据划分失败: {e}")
        return
    
    # 为训练系统生成训练数据
    logger.info("生成训练数据...")
    try:
        training_data = {
            'train_data': train_data_split,
            'val_data': val_data_split, 
            'test_data': test_data_split,
            'rel_info': rel_info,
            'total_documents': len(processed_data),
            'entity_types': list(set([entity['type'] for doc in processed_data for entity in doc.get('entities', [])])),
            'relation_types': list(rel_info.keys())
        }
        
        # 保存完整训练数据
        training_path = Path(output_dir) / "docred_training_data.json"
        with open(training_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # 为GUI系统生成可视化数据
        gui_data = []
        for doc in processed_data:
            gui_doc = {
                'doc_id': doc['doc_id'],
                'title': doc['title'],
                'text': doc['original_text'][:1000] + "..." if len(doc['original_text']) > 1000 else doc['original_text'],
                'entities': doc['entities'],
                'relations': doc['relations'],
                'sentences': doc['sentences']
            }
            gui_data.append(gui_doc)
        
        gui_path = Path(output_dir) / "gui_data.json"
        with open(gui_path, 'w', encoding='utf-8') as f:
            json.dump(gui_data, f, ensure_ascii=False, indent=2)
        
        # 生成统计信息
        stats = {
            'total_documents': len(processed_data),
            'train_size': len(train_data_split),
            'val_size': len(val_data_split),
            'test_size': len(test_data_split),
            'total_entities': sum(len(doc.get('entities', [])) for doc in processed_data),
            'total_relations': sum(len(doc.get('relations', [])) for doc in processed_data),
            'entity_types': list(set([entity['type'] for doc in processed_data for entity in doc.get('entities', [])])),
            'relation_distribution': {
                rel_info.get(rel_id, f'Relation_{rel_id}'): sum(1 for doc in processed_data for rel in doc.get('relations', []) if rel.get('relation_id') == rel_id)
                for rel_id in rel_info.keys()
            }
        }
        
        stats_path = Path(output_dir) / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("=== 数据处理完成 ===")
        logger.info(f"总文档数: {stats['total_documents']}")
        logger.info(f"训练集: {stats['train_size']} 样本")
        logger.info(f"验证集: {stats['val_size']} 样本") 
        logger.info(f"测试集: {stats['test_size']} 样本")
        logger.info(f"总实体数: {stats['total_entities']}")
        logger.info(f"总关系数: {stats['total_relations']}")
        logger.info(f"数据保存位置:")
        logger.info(f"  训练数据: {training_path}")
        logger.info(f"  GUI数据: {gui_path}")
        logger.info(f"  数据统计: {stats_path}")
        logger.info(f"  训练/验证/测试集: {output_dir}/train.json, val.json, test.json")
        logger.info("DocRED数据集处理完成！数据已准备就绪用于训练和GUI系统。")
        
    except Exception as e:
        logger.error(f"生成输出文件失败: {e}")
        return


if __name__ == "__main__":
    main()