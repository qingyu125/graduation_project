#!/usr/bin/env python3
"""
数据管理模块
整合所有数据处理功能的主入口
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .data_loader import DocRedDataLoader, Document
from .preprocessor import DocRedPreprocessor, DataPreprocessorConfig, TrainingSample

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class DataManager:
    """数据管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_loader = None
        self.preprocessor = None
        self.documents = []
        self.training_samples = []
        
        # 输出目录
        self.output_dir = Path("./data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize(self, data_dir: str = None) -> bool:
        """初始化数据管理器"""
        try:
            # 初始化数据加载器
            self.data_loader = DocRedDataLoader(data_dir)
            
            # 初始化预处理器
            preprocessor_config = DataPreprocessorConfig(
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                min_entities=1,
                min_relations=0,
                max_entities=50,
                max_relations=100
            )
            self.preprocessor = DocRedPreprocessor(preprocessor_config)
            
            logger.info("数据管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"数据管理器初始化失败: {e}")
            return False
    
    def load_and_process_docred(self, rel_info_path: str = None, train_data_path: str = None) -> bool:
        """加载和处理DocRED数据"""
        try:
            # 加载关系信息
            rel_info = self.data_loader.load_rel_info()
            if not rel_info:
                logger.error("加载关系信息失败")
                return False
            
            # 加载训练数据
            train_data = self.data_loader.load_train_data()
            if not train_data:
                logger.error("加载训练数据失败")
                return False
            
            # 处理文档
            self.documents = self.data_loader.process_all_documents()
            if not self.documents:
                logger.error("处理文档失败")
                return False
            
            logger.info(f"成功加载和处理 {len(self.documents)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"加载DocRED数据失败: {e}")
            return False
    
    def create_training_dataset(self, 
                               filter_criteria: Optional[Dict[str, Any]] = None,
                               augment_data: bool = True,
                               augmentation_ratio: float = 0.2) -> bool:
        """创建训练数据集"""
        try:
            # 过滤文档
            if filter_criteria is None:
                filter_criteria = {}
            
            filtered_docs = self.data_loader.filter_data_by_criteria(
                self.documents, **filter_criteria
            )
            
            if not filtered_docs:
                logger.error("过滤后没有有效文档")
                return False
            
            # 创建训练样本
            self.training_samples = self.preprocessor.create_training_samples(filtered_docs)
            
            # 数据增强
            if augment_data:
                self.training_samples = self.preprocessor.augment_training_data(
                    self.training_samples, augmentation_ratio
                )
            
            # 划分数据集
            train_samples, val_samples, test_samples = self.preprocessor.split_samples(
                self.training_samples
            )
            
            # 保存数据集
            self._save_datasets(train_samples, val_samples, test_samples)
            
            # 创建伪代码数据集
            self.preprocessor.create_pseudo_code_dataset(self.training_samples)
            
            logger.info("训练数据集创建完成")
            return True
            
        except Exception as e:
            logger.error(f"创建训练数据集失败: {e}")
            return False
    
    def _save_datasets(self, 
                      train_samples: List[TrainingSample], 
                      val_samples: List[TrainingSample], 
                      test_samples: List[TrainingSample]):
        """保存数据集"""
        # 保存训练集
        train_path = self.output_dir / "train_samples.json"
        self.preprocessor.save_samples(train_samples, str(train_path))
        
        # 保存验证集
        val_path = self.output_dir / "val_samples.json"
        self.preprocessor.save_samples(val_samples, str(val_path))
        
        # 保存测试集
        test_path = self.output_dir / "test_samples.json"
        self.preprocessor.save_samples(test_samples, str(test_path))
        
        # 保存完整数据集
        all_samples_path = self.output_dir / "all_samples.json"
        self.preprocessor.save_samples(self.training_samples, str(all_samples_path))
        
        # 生成数据统计信息
        self._generate_data_statistics()
    
    def _generate_data_statistics(self):
        """生成数据统计信息"""
        try:
            # 样本质量分析
            quality_stats = self.preprocessor.analyze_sample_quality(self.training_samples)
            
            # 文档统计
            doc_stats = self.data_loader.analyze_data_statistics(self.documents)
            
            # 合并统计信息
            all_stats = {
                'generation_time': datetime.now().isoformat(),
                'total_documents': len(self.documents),
                'total_training_samples': len(self.training_samples),
                'document_statistics': doc_stats,
                'sample_quality': quality_stats
            }
            
            # 保存统计信息
            stats_path = self.output_dir / "data_statistics.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(all_stats, f, ensure_ascii=False, indent=2)
            
            # 生成可视化报告
            self._create_visualization_report(all_stats)
            
            logger.info(f"数据统计信息已生成: {stats_path}")
            
        except Exception as e:
            logger.error(f"生成数据统计信息失败: {e}")
    
    def _create_visualization_report(self, stats: Dict[str, Any]):
        """创建可视化报告"""
        try:
            # 设置图表样式
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('DocRED数据集分析报告', fontsize=16, fontweight='bold')
            
            # 1. 文档长度分布
            doc_lengths = stats['document_statistics']['doc_lengths']['sentences']
            axes[0, 0].hist(doc_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('文档句子数分布')
            axes[0, 0].set_xlabel('句子数')
            axes[0, 0].set_ylabel('文档数')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 实体数量分布
            entity_counts = stats['document_statistics']['doc_lengths']['entities']
            axes[0, 1].hist(entity_counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('文档实体数分布')
            axes[0, 1].set_xlabel('实体数')
            axes[0, 1].set_ylabel('文档数')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 关系数量分布
            relation_counts = stats['document_statistics']['doc_lengths']['relations']
            axes[0, 2].hist(relation_counts, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 2].set_title('文档关系数分布')
            axes[0, 2].set_xlabel('关系数')
            axes[0, 2].set_ylabel('文档数')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 实体类型分布（前10个）
            entity_types = stats['document_statistics']['entity_types']
            top_entity_types = dict(sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10])
            axes[1, 0].bar(range(len(top_entity_types)), list(top_entity_types.values()), 
                          color='orange', alpha=0.7)
            axes[1, 0].set_title('实体类型分布（前10个）')
            axes[1, 0].set_xlabel('实体类型')
            axes[1, 0].set_ylabel('数量')
            axes[1, 0].set_xticks(range(len(top_entity_types)))
            axes[1, 0].set_xticklabels(list(top_entity_types.keys()), rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. 关系类型分布（前10个）
            relation_types = stats['document_statistics']['relation_types']
            top_relation_types = dict(sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10])
            axes[1, 1].bar(range(len(top_relation_types)), list(top_relation_types.values()), 
                          color='purple', alpha=0.7)
            axes[1, 1].set_title('关系类型分布（前10个）')
            axes[1, 1].set_xlabel('关系类型')
            axes[1, 1].set_ylabel('数量')
            axes[1, 1].set_xticks(range(len(top_relation_types)))
            axes[1, 1].set_xticklabels(list(top_relation_types.keys()), rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. 文本长度分布
            text_lengths = [len(sample.text) for sample in self.training_samples]
            axes[1, 2].hist(text_lengths, bins=30, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 2].set_title('训练样本文本长度分布')
            axes[1, 2].set_xlabel('字符数')
            axes[1, 2].set_ylabel('样本数')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = self.output_dir / "data_analysis_charts.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"可视化报告已保存: {chart_path}")
            
        except Exception as e:
            logger.error(f"创建可视化报告失败: {e}")
    
    def create_data_summary_report(self, output_path: str = None) -> str:
        """创建数据摘要报告"""
        if output_path is None:
            output_path = self.output_dir / "data_summary_report.md"
        else:
            output_path = Path(output_path)
        
        try:
            # 加载统计信息
            stats_path = self.output_dir / "data_statistics.json"
            if not stats_path.exists():
                logger.error("统计信息文件不存在")
                return ""
            
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            # 生成报告
            report_lines = [
                "# DocRED数据集处理摘要报告",
                "",
                f"**生成时间**: {stats['generation_time']}",
                f"**数据来源**: DocRED数据集",
                "",
                "## 数据概览",
                "",
                f"- **总文档数**: {stats['total_documents']:,}",
                f"- **总训练样本数**: {stats['total_training_samples']:,}",
                "",
                "## 文档统计",
                "",
                f"- **平均句子数**: {stats['document_statistics']['doc_lengths']['sentences_stats']['mean']:.2f}",
                f"- **平均实体数**: {stats['document_statistics']['doc_lengths']['entities_stats']['mean']:.2f}",
                f"- **平均关系数**: {stats['document_statistics']['doc_lengths']['relations_stats']['mean']:.2f}",
                "",
                "### 实体类型分布（前10个）",
                ""
            ]
            
            # 添加实体类型分布
            entity_types = stats['document_statistics']['entity_types']
            top_entity_types = dict(sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10])
            for entity_type, count in top_entity_types.items():
                report_lines.append(f"- **{entity_type}**: {count:,}")
            
            report_lines.extend([
                "",
                "### 关系类型分布（前10个）",
                ""
            ])
            
            # 添加关系类型分布
            relation_types = stats['document_statistics']['relation_types']
            top_relation_types = dict(sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10])
            for rel_type, count in top_relation_types.items():
                report_lines.append(f"- **{rel_type}**: {count:,}")
            
            report_lines.extend([
                "",
                "## 训练样本质量",
                "",
                f"- **平均每样本实体数**: {stats['sample_quality']['avg_entities_per_sample']:.2f}",
                f"- **平均每样本关系数**: {stats['sample_quality']['avg_relations_per_sample']:.2f}",
                f"- **平均文本长度**: {stats['sample_quality']['text_length_stats']['mean']:.0f} 字符",
                "",
                "## 数据文件",
                "",
                "处理后的数据文件位于 `data/processed/` 目录下：",
                "",
                "- `train_samples.json`: 训练集样本",
                "- `val_samples.json`: 验证集样本", 
                "- `test_samples.json`: 测试集样本",
                "- `all_samples.json`: 完整样本集",
                "- `pseudo_code_dataset.json`: 伪代码数据集",
                "- `prompt_examples.json`: 提示模板示例",
                "- `data_statistics.json`: 详细统计信息",
                "- `data_analysis_charts.png`: 可视化图表",
                "",
                "## 下一步",
                "",
                "1. 使用 `pseudo_code_dataset.json` 训练文本到伪代码转换模型",
                "2. 使用训练样本训练要素抽取和关系抽取模型",
                "3. 在验证集上评估模型性能",
                "4. 在测试集上进行最终测试",
                ""
            ])
            
            # 保存报告
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"数据摘要报告已保存: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"创建数据摘要报告失败: {e}")
            return ""
    
    def load_processed_data(self, 
                           data_type: str = 'all') -> Optional[List[TrainingSample]]:
        """加载已处理的数据"""
        try:
            if data_type == 'train':
                samples = self.preprocessor.load_samples(str(self.output_dir / "train_samples.json"))
            elif data_type == 'val':
                samples = self.preprocessor.load_samples(str(self.output_dir / "val_samples.json"))
            elif data_type == 'test':
                samples = self.preprocessor.load_samples(str(self.output_dir / "test_samples.json"))
            elif data_type == 'all':
                samples = self.preprocessor.load_samples(str(self.output_dir / "all_samples.json"))
            else:
                logger.error(f"未知的数据类型: {data_type}")
                return None
            
            return samples
            
        except Exception as e:
            logger.error(f"加载处理数据失败: {e}")
            return None
    
    def export_for_model_training(self, 
                                 output_dir: str = None,
                                 formats: List[str] = ['json', 'jsonl']) -> bool:
        """导出用于模型训练的数据"""
        try:
            if output_dir is None:
                output_dir = self.output_dir / "for_training"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 导出不同格式的数据
            for format_type in formats:
                # 训练集
                train_path = output_dir / f"train.{format_type}"
                if format_type == 'json':
                    # JSON格式需要特殊处理
                    train_samples = self.load_processed_data('train')
                    if train_samples:
                        with open(train_path, 'w', encoding='utf-8') as f:
                            json.dump([sample.__dict__ for sample in train_samples], f, 
                                     ensure_ascii=False, indent=2)
                else:
                    # JSONL格式
                    train_path = output_dir / f"train.{format_type}"
                    with open(train_path, 'w', encoding='utf-8') as f:
                        for sample in self.training_samples:
                            f.write(json.dumps(sample.__dict__, ensure_ascii=False) + '\n')
            
            # 创建数据说明文件
            readme_path = output_dir / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("# 训练数据说明\n\n")
                f.write("这是为模型训练准备的数据集。\n\n")
                f.write("## 文件说明\n\n")
                f.write("- `train.json/jsonl`: 训练数据\n")
                f.write("- `README.md`: 本说明文件\n\n")
                f.write("## 数据格式\n\n")
                f.write("每个样本包含以下字段：\n")
                f.write("- `doc_id`: 文档ID\n")
                f.write("- `text`: 原始文本\n")
                f.write("- `entities`: 实体列表\n")
                f.write("- `relations`: 关系列表\n")
                f.write("- `metadata`: 元数据（可选）\n")
            
            logger.info(f"训练数据已导出到: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"导出训练数据失败: {e}")
            return False


def main():
    """主函数 - 演示数据管理器的使用"""
    # 创建数据管理器
    data_manager = DataManager()
    
    # 初始化
    if not data_manager.initialize():
        print("数据管理器初始化失败")
        return
    
    # 加载DocRED数据（需要先下载数据）
    print("请确保已下载DocRED数据集到 data/raw/docred 目录")
    print("或者修改 data_dir 参数指向正确的数据路径")
    
    # 如果数据存在，处理数据
    # if data_manager.load_and_process_docred():
    #     # 创建训练数据集
    #     if data_manager.create_training_dataset():
    #         # 生成报告
    #         report_path = data_manager.create_data_summary_report()
    #         print(f"数据处理完成，报告保存在: {report_path}")


if __name__ == "__main__":
    main()