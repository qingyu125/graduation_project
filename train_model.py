#!/usr/bin/env python3
"""
DocRED关系抽取模型训练脚本
使用处理后的数据进行模型训练
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.trainer import DocRedTrainer
from src.models.model_loader import ModelLoader
from src.data.manager import DataManager
from src.utils.config import get_config
import json
import torch

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiments/logs/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_processed_data():
    """加载处理后的训练数据"""
    logger = logging.getLogger(__name__)
    
    data_path = Path("data/processed/docred_training_data.json")
    if not data_path.exists():
        logger.error(f"训练数据文件不存在: {data_path}")
        return None
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        logger.info(f"成功加载训练数据:")
        logger.info(f"  训练样本: {len(training_data['train_data'])}")
        logger.info(f"  验证样本: {len(training_data['val_data'])}")
        logger.info(f"  测试样本: {len(training_data['test_data'])}")
        logger.info(f"  关系类型: {len(training_data['rel_info'])}")
        logger.info(f"  实体类型: {len(training_data['entity_types'])}")
        
        return training_data
    except Exception as e:
        logger.error(f"加载训练数据失败: {e}")
        return None

def create_training_config(args):
    """创建训练配置"""
    config = {
        'model': {
            'name': args.model_name,
            'use_lora': args.use_lora,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'precision': args.precision,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'max_seq_length': args.max_seq_length,
            'warmup_steps': 100,
            'save_steps': 500,
            'eval_steps': 250,
            'logging_steps': 50
        },
        'output': {
            'output_dir': args.output_dir,
            'logging_dir': f"{args.output_dir}/logs",
            'save_total_limit': 3
        },
        'data': {
            'train_path': "data/processed/train.json",
            'val_path': "data/processed/val.json", 
            'test_path': "data/processed/test.json",
            'rel_info_path': "data/processed/rel_info.json"
        }
    }
    return config

def main():
    parser = argparse.ArgumentParser(description="DocRED关系抽取模型训练")
    
    # 模型配置
    parser.add_argument('--model-name', type=str, default='codellama/CodeLlama-7b-Instruct-hf',
                       help='基础模型名称或路径')
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='是否使用LoRA微调')
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA rank参数')
    parser.add_argument('--lora-alpha', type=int, default=32,
                       help='LoRA alpha参数')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                       help='LoRA dropout参数')
    parser.add_argument('--precision', type=str, default='4bit',
                       choices=['4bit', '8bit', 'fp16', 'fp32'],
                       help='模型精度')
    
    # 训练配置
    parser.add_argument('--batch-size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='学习率')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='最大序列长度')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='experiments/checkpoints',
                       help='输出目录')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("=== DocRED模型训练开始 ===")
    logger.info(f"模型: {args.model_name}")
    logger.info(f"LoRA: {args.use_lora}, r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"精度: {args.precision}")
    logger.info(f"批次: {args.batch_size}, 学习率: {args.learning_rate}")
    logger.info(f"轮数: {args.num_epochs}")
    
    # 加载训练数据
    training_data = load_processed_data()
    if not training_data:
        logger.error("无法加载训练数据，请先运行数据处理脚本")
        return 1
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    if device.type == 'cpu':
        logger.warning("使用CPU训练，速度较慢，建议使用GPU")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info("正在加载模型...")
    model_loader = ModelLoader()
    try:
        model, tokenizer = model_loader.load_model(
            model_path=args.model_name,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            precision=args.precision
        )
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.info("请检查模型名称是否正确，或者先下载预训练模型")
        return 1
    
    # 创建训练器
    logger.info("创建训练器...")
    config = create_training_config(args)
    trainer = DocRedTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
    
    # 开始训练
    logger.info("开始训练...")
    try:
        # 准备训练数据
        train_data = training_data['train_data']
        val_data = training_data['val_data'] if training_data['val_data'] else []
        
        # 如果验证集为空，使用训练集的一部分作为验证集
        if not val_data and train_data:
            val_size = max(1, len(train_data) // 5)  # 20%作为验证集
            val_data = train_data[-val_size:]
            train_data = train_data[:-val_size]
            logger.info(f"自动划分验证集: {len(train_data)} 训练, {len(val_data)} 验证")
        
        # 训练模型
        trainer.train(
            train_dataset=train_data,
            eval_dataset=val_data if val_data else None
        )
        
        # 保存最终模型
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        logger.info(f"模型已保存到: {final_model_path}")
        
        # 生成训练报告
        generate_training_report(trainer, training_data, config)
        
        logger.info("=== 训练完成 ===")
        logger.info(f"最终模型保存在: {output_dir}")
        logger.info(f"训练日志: {output_dir}/logs/training.log")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def generate_training_report(trainer, training_data, config):
    """生成训练报告"""
    logger = logging.getLogger(__name__)
    
    report = {
        'training_config': config,
        'data_info': {
            'total_documents': training_data['total_documents'],
            'entity_types': training_data['entity_types'],
            'relation_types': training_data['relation_types']
        },
        'model_info': {
            'model_name': config['model']['name'],
            'use_lora': config['model']['use_lora'],
            'precision': config['model']['precision']
        },
        'training_results': {
            'final_loss': getattr(trainer, 'final_loss', 'N/A'),
            'training_steps': getattr(trainer, 'training_steps', 'N/A'),
            'best_epoch': getattr(trainer, 'best_epoch', 'N/A')
        }
    }
    
    report_path = Path(config['output']['output_dir']) / "training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练报告已保存: {report_path}")

if __name__ == "__main__":
    sys.exit(main())