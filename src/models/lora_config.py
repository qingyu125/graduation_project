"""
LoRA配置管理模块

本模块提供完整的LoRA（Low-Rank Adaptation）配置管理功能，
支持动态参数调整、预设模板、参数验证、自动调优等功能。

主要功能：
- LoRA配置类的定义和管理
- 多种预设配置模板
- 动态参数调整和验证
- 配置文件的保存和加载
- 自动调优功能
- 详细的参数说明和文档

Author: DocRED Paper Team
Date: 2025-11-06
"""

import json
import yaml
import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from enum import Enum


class LoraTaskType(Enum):
    """LoRA任务类型枚举"""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_CLASSIFICATION = "relation_classification" 
    JOINT_LEARNING = "joint_learning"
    DOCUMENT_QA = "document_qa"
    CUSTOM = "custom"


class LoraModelSize(Enum):
    """LoRA模型规模枚举"""
    SMALL = "small"      # < 1B parameters
    MEDIUM = "medium"    # 1B - 7B parameters
    LARGE = "large"      # 7B - 70B parameters
    XL = "xl"           # > 70B parameters


@dataclass
class LoraConfig:
    """
    LoRA配置类
    
    Args:
        rank (int): LoRA秩，默认为8。控制低秩矩阵的维度。
                   更大的rank提供更强的表达能力但增加计算成本。
        lora_alpha (int): LoRA缩放参数，默认为32。用于缩放LoRA更新。
        lora_dropout (float): LoRA层的dropout率，默认为0.1。
        learning_rate (float): 学习率，默认为2e-4。
        target_modules (List[str]): 需要应用LoRA的模块名称列表。
        task_type (str): 任务类型，影响默认参数设置。
        model_size (str): 模型规模，影响默认参数设置。
    """
    
    # LoRA核心参数
    rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    
    # 模块配置
    target_modules: List[str] = None
    exclude_modules: List[str] = None
    
    # 任务和模型配置
    task_type: str = "custom"
    model_size: str = "medium"
    
    # 训练配置
    max_epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # LoRA高级配置
    use_rslora: bool = False  # 使用Rank-Stabilized LoRA
    lora_init_type: str = "standard"  # standard, gaussian, scaled
    lora_scaling: float = 1.0
    
    # 其他配置
    seed: int = 42
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    report_to: str = "none"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.target_modules is None:
            self.target_modules = self._get_default_target_modules()
        if self.exclude_modules is None:
            self.exclude_modules = []
        
        # 参数验证
        self._validate_config()
    
    def _get_default_target_modules(self) -> List[str]:
        """根据任务类型和模型规模获取默认的目标模块"""
        default_modules = {
            LoraTaskType.ENTITY_EXTRACTION: [
                "query", "key", "value", "output.dense", "intermediate.dense"
            ],
            LoraTaskType.RELATION_CLASSIFICATION: [
                "query", "key", "value", "output.dense", "intermediate.dense"
            ],
            LoraTaskType.JOINT_LEARNING: [
                "query", "key", "value", "output.dense", "intermediate.dense"
            ],
            LoraTaskType.DOCUMENT_QA: [
                "query", "key", "value", "output.dense", "intermediate.dense", "attention.self"
            ]
        }
        
        # 处理CUSTOM和其他未明确指定的任务类型
        if self.task_type == "CUSTOM":
            modules = ["query", "key", "value", "output.dense", "intermediate.dense"]
        else:
            try:
                task_enum = LoraTaskType(self.task_type)
                modules = default_modules.get(task_enum, 
                                            ["query", "key", "value", "output.dense", "intermediate.dense"])
            except ValueError:
                # 如果任务类型不在枚举中，使用CUSTOM配置
                modules = ["query", "key", "value", "output.dense", "intermediate.dense"]
        
        # 根据模型规模调整
        if self.model_size == LoraModelSize.SMALL.value:
            # 小模型使用更多模块
            modules.extend(["attention.output.dense", "hidden.dense"])
        elif self.model_size == LoraModelSize.XL.value:
            # 大模型只使用关键模块
            modules = ["query", "key", "value"]
        
        return modules
    
    def _validate_config(self):
        """验证配置参数"""
        errors = []
        
        # 基本参数验证
        if self.rank <= 0:
            errors.append("rank必须为正整数")
        if self.lora_alpha <= 0:
            errors.append("lora_alpha必须为正整数")
        if not 0 <= self.lora_dropout <= 1:
            errors.append("lora_dropout必须在[0,1]范围内")
        if self.learning_rate <= 0:
            errors.append("learning_rate必须为正数")
        if self.weight_decay < 0:
            errors.append("weight_decay不能为负数")
        
        # 任务类型验证
        if self.task_type not in [t.value for t in LoraTaskType]:
            errors.append(f"不支持的任务类型: {self.task_type}")
        
        # 模型规模验证
        if self.model_size not in [s.value for s in LoraModelSize]:
            errors.append(f"不支持的模型规模: {self.model_size}")
        
        # 训练参数验证
        if self.max_epochs <= 0:
            errors.append("max_epochs必须为正整数")
        if self.batch_size <= 0:
            errors.append("batch_size必须为正整数")
        if self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps必须为正整数")
        if self.warmup_steps < 0:
            errors.append("warmup_steps不能为负数")
        if self.max_grad_norm <= 0:
            errors.append("max_grad_norm必须为正数")
        
        # LoRA高级配置验证
        if self.lora_init_type not in ["standard", "gaussian", "scaled"]:
            errors.append("lora_init_type必须是: standard, gaussian, scaled之一")
        if self.lora_scaling <= 0:
            errors.append("lora_scaling必须为正数")
        
        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
    
    def update_params(self, **kwargs):
        """
        动态更新配置参数
        
        Args:
            **kwargs: 要更新的参数键值对
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logging.warning(f"未知参数: {key}")
        
        # 重新验证配置
        self._validate_config()
    
    def get_effective_rank(self) -> int:
        """获取有效秩，考虑缩放因子"""
        if self.use_rslora:
            return int(self.rank * (self.lora_alpha / (self.rank + 1)))
        return self.rank
    
    def get_memory_estimate(self, model_params: int) -> float:
        """
        估算LoRA额外内存消耗（MB）
        
        Args:
            model_params: 模型参数量
            
        Returns:
            float: 估计的额外内存消耗（MB）
        """
        # LoRA参数内存 = 2 * rank * (in_dim + out_dim) * alpha_scaling
        # 这里使用简化的估算方法
        base_params = len(self.target_modules) * self.rank * 1024  # 假设平均每模块1024参数
        
        # 考虑dropout和梯度
        gradient_overhead = base_params * 2  # 梯度通常为参数的2倍
        optimizer_overhead = base_params * 4  # Adam优化器需要4倍参数内存
        
        total_mb = (base_params + gradient_overhead + optimizer_overhead) * 4 / 1024 / 1024  # float32 = 4 bytes
        return total_mb
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def copy(self) -> 'LoraConfig':
        """创建配置的深拷贝"""
        return LoraConfig(**self.to_dict())
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"LoraConfig(rank={self.rank}, alpha={self.lora_alpha}, lr={self.learning_rate}, " \
               f"task={self.task_type}, model_size={self.model_size})"
    
    def __repr__(self) -> str:
        """详细的字符串表示"""
        return self.to_json()


class LoraConfigManager:
    """LoRA配置管理器"""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件保存目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 预设配置模板
        self.presets = self._initialize_presets()
        
        # 参数范围定义（用于自动调优）
        self.param_ranges = {
            'rank': (1, 128),
            'lora_alpha': (8, 256),
            'lora_dropout': (0.0, 0.5),
            'learning_rate': (1e-6, 1e-2),
            'weight_decay': (0.0, 0.1),
            'batch_size': [1, 2, 4, 8, 16, 32],
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_presets(self) -> Dict[str, Dict[str, Any]]:
        """初始化预设配置模板"""
        return {
            # 轻量级配置 - 适合小模型和快速实验
            "lightweight": {
                "rank": 4,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "learning_rate": 3e-4,
                "max_epochs": 5,
                "batch_size": 32,
                "task_type": "custom",
                "model_size": "small",
                "description": "轻量级配置，适合快速实验和小模型"
            },
            
            # 平衡配置 - 默认配置
            "balanced": {
                "rank": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "max_epochs": 10,
                "batch_size": 16,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 100,
                "task_type": "custom",
                "model_size": "medium",
                "description": "平衡配置，兼顾性能和效率"
            },
            
            # 高性能配置 - 适合大模型
            "high_performance": {
                "rank": 16,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "max_epochs": 15,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 500,
                "use_rslora": True,
                "task_type": "custom",
                "model_size": "large",
                "description": "高性能配置，适合大模型和高质量任务"
            },
            
            # 实体抽取专用配置
            "entity_extraction": {
                "rank": 12,
                "lora_alpha": 48,
                "lora_dropout": 0.15,
                "learning_rate": 1.5e-4,
                "weight_decay": 0.01,
                "max_epochs": 12,
                "batch_size": 16,
                "target_modules": ["query", "key", "value", "output.dense"],
                "task_type": "entity_extraction",
                "model_size": "medium",
                "description": "实体抽取任务专用配置"
            },
            
            # 关系分类专用配置
            "relation_classification": {
                "rank": 10,
                "lora_alpha": 40,
                "lora_dropout": 0.1,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "max_epochs": 15,
                "batch_size": 24,
                "target_modules": ["query", "key", "value", "intermediate.dense"],
                "task_type": "relation_classification",
                "model_size": "medium",
                "description": "关系分类任务专用配置"
            },
            
            # 联合学习配置
            "joint_learning": {
                "rank": 16,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "max_epochs": 20,
                "batch_size": 12,
                "gradient_accumulation_steps": 2,
                "target_modules": ["query", "key", "value", "output.dense", "intermediate.dense"],
                "task_type": "joint_learning",
                "model_size": "large",
                "description": "实体和关系联合学习配置"
            },
            
            # 文档问答配置
            "document_qa": {
                "rank": 20,
                "lora_alpha": 80,
                "lora_dropout": 0.15,
                "learning_rate": 8e-5,
                "weight_decay": 0.01,
                "max_epochs": 25,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 1000,
                "target_modules": ["query", "key", "value", "output.dense", "intermediate.dense", "attention.self"],
                "task_type": "document_qa",
                "model_size": "large",
                "description": "文档问答任务专用配置"
            },
            
            # 超轻量配置 - 极低资源消耗
            "ultra_light": {
                "rank": 2,
                "lora_alpha": 8,
                "lora_dropout": 0.01,
                "learning_rate": 5e-4,
                "max_epochs": 3,
                "batch_size": 64,
                "task_type": "custom",
                "model_size": "small",
                "description": "超轻量配置，极低资源消耗"
            },
            
            # 质量优先配置
            "quality_focused": {
                "rank": 32,
                "lora_alpha": 128,
                "lora_dropout": 0.1,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "max_epochs": 30,
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "warmup_steps": 2000,
                "use_rslora": True,
                "lora_init_type": "scaled",
                "task_type": "custom",
                "model_size": "xl",
                "description": "质量优先配置，追求最佳性能"
            }
        }
    
    def get_preset(self, name: str) -> LoraConfig:
        """
        获取预设配置
        
        Args:
            name: 预设配置名称
            
        Returns:
            LoraConfig: 配置对象
        """
        if name not in self.presets:
            raise ValueError(f"未知预设配置: {name}")
        
        config_dict = self.presets[name].copy()
        # 移除description字段
        config_dict.pop('description', None)
        
        return LoraConfig(**config_dict)
    
    def list_presets(self) -> List[str]:
        """获取所有可用预设配置名称"""
        return list(self.presets.keys())
    
    def get_preset_info(self, name: str = None) -> Dict[str, str]:
        """
        获取预设配置信息
        
        Args:
            name: 具体配置名称，如果为None则返回所有配置信息
            
        Returns:
            Dict[str, str]: 配置名称和描述的映射
        """
        if name:
            if name not in self.presets:
                raise ValueError(f"未知预设配置: {name}")
            return {name: self.presets[name]['description']}
        
        return {name: config['description'] for name, config in self.presets.items()}
    
    def save_config(self, config: LoraConfig, filename: str, format: str = "json"):
        """
        保存配置到文件
        
        Args:
            config: LoRA配置对象
            filename: 文件名
            format: 文件格式（json或yaml）
        """
        filepath = self.config_dir / f"{filename}.{format}"
        
        config_dict = config.to_dict()
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif format.lower() == "yaml":
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        self.logger.info(f"配置已保存到: {filepath}")
    
    def load_config(self, filename: str) -> LoraConfig:
        """
        从文件加载配置
        
        Args:
            filename: 文件名（不包含扩展名）
            
        Returns:
            LoraConfig: LoRA配置对象
        """
        # 尝试不同的文件格式
        for ext in ["json", "yaml", "yml"]:
            filepath = self.config_dir / f"{filename}.{ext}"
            if filepath.exists():
                break
        else:
            raise FileNotFoundError(f"配置文件不存在: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # 处理列表类型参数
        for key in ['target_modules', 'exclude_modules']:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = config_dict[key].split(',')
        
        return LoraConfig(**config_dict)
    
    def create_custom_config(self, 
                           task_type: str = "CUSTOM",
                           model_size: str = "MEDIUM",
                           priority: str = "balanced",
                           **kwargs) -> LoraConfig:
        """
        创建自定义配置
        
        Args:
            task_type: 任务类型
            model_size: 模型规模
            priority: 参数优先级（speed, balanced, quality）
            **kwargs: 额外参数
            
        Returns:
            LoraConfig: 自定义配置对象
        """
        # 基于优先级选择初始模板
        if priority == "speed":
            template = "lightweight"
        elif priority == "quality":
            template = "quality_focused"
        else:
            template = "balanced"
        
        config = self.get_preset(template)
        
        # 更新任务和模型相关参数
        config.update_params(task_type=task_type, model_size=model_size)
        
        # 应用额外参数
        if kwargs:
            config.update_params(**kwargs)
        
        return config
    
    def auto_tune(self, 
                 config: LoraConfig,
                 metric_history: List[Tuple[int, float]],
                 tuning_strategy: str = "adaptive") -> LoraConfig:
        """
        基于训练历史自动调优配置
        
        Args:
            config: 初始配置
            metric_history: 训练历史 [(epoch, metric_value), ...]
            tuning_strategy: 调优策略 ("adaptive", "grid", "random")
            
        Returns:
            LoraConfig: 调优后的配置
        """
        new_config = config.copy()
        
        if len(metric_history) < 2:
            self.logger.warning("训练历史数据不足，无法进行自动调优")
            return new_config
        
        # 分析训练趋势
        recent_metrics = [m[1] for m in metric_history[-3:]]  # 最近3个epoch
        trend = np.polyfit(range(len(recent_metrics)), recent_metrics, 1)[0]
        
        if tuning_strategy == "adaptive":
            if trend < 0:  # 性能下降
                # 降低学习率，增加正则化
                new_config.learning_rate *= 0.5
                new_config.lora_dropout = min(new_config.lora_dropout * 1.1, 0.3)
                new_config.weight_decay *= 1.2
                self.logger.info("检测到性能下降，降低学习率并增加正则化")
            else:  # 性能提升
                # 可以适当增加学习率或减少正则化
                if config.learning_rate < 1e-3:
                    new_config.learning_rate *= 1.1
        
        elif tuning_strategy == "grid":
            # 网格搜索下一个参数组合
            if len(metric_history) % 3 == 0:  # 每3个epoch调整一次
                self._grid_search_adjustment(new_config, metric_history)
        
        elif tuning_strategy == "random":
            # 随机搜索参数
            self._random_search_adjustment(new_config)
        
        # 参数边界检查
        self._apply_parameter_bounds(new_config)
        
        self.logger.info(f"自动调优完成: {config} -> {new_config}")
        return new_config
    
    def _grid_search_adjustment(self, config: LoraConfig, history: List[Tuple[int, float]]):
        """网格搜索调优策略"""
        best_metric = max(h[1] for h in history)
        
        # 如果性能不错，可以尝试更大的rank
        if best_metric > 0.8:
            if config.rank < 32:
                config.rank = min(config.rank * 2, 32)
                config.lora_alpha = config.rank * 4
        else:
            # 性能不佳时，减小参数
            if config.rank > 2:
                config.rank = max(config.rank // 2, 2)
                config.lora_alpha = config.rank * 4
    
    def _random_search_adjustment(self, config: LoraConfig):
        """随机搜索调优策略"""
        import random
        random.seed(42)  # 确保可重复性
        
        # 随机调整1-2个参数
        params_to_tune = random.sample(['rank', 'learning_rate', 'lora_dropout'], 
                                     random.randint(1, 2))
        
        for param in params_to_tune:
            if param == 'rank':
                current = getattr(config, param)
                range_min, range_max = self.param_ranges['rank']
                new_value = random.randint(range_min, range_max)
                setattr(config, param, new_value)
            elif param == 'learning_rate':
                current = getattr(config, param)
                range_min, range_max = self.param_ranges['learning_rate']
                # 在对数尺度上随机选择
                log_min, log_max = np.log10(range_min), np.log10(range_max)
                log_value = random.uniform(log_min, log_max)
                new_value = 10 ** log_value
                setattr(config, param, new_value)
            elif param == 'lora_dropout':
                current = getattr(config, param)
                range_min, range_max = self.param_ranges['lora_dropout']
                new_value = random.uniform(range_min, range_max)
                setattr(config, param, new_value)
    
    def _apply_parameter_bounds(self, config: LoraConfig):
        """应用参数边界"""
        # 应用rank边界
        range_min, range_max = self.param_ranges['rank']
        config.rank = max(range_min, min(range_max, config.rank))
        
        # 应用alpha边界
        range_min, range_max = self.param_ranges['lora_alpha']
        config.lora_alpha = max(range_min, min(range_max, config.lora_alpha))
        
        # 应用dropout边界
        range_min, range_max = self.param_ranges['lora_dropout']
        config.lora_dropout = max(range_min, min(range_max, config.lora_dropout))
    
    def compare_configs(self, configs: List[LoraConfig], 
                       model_params: int) -> Dict[str, Any]:
        """
        比较多个配置的性能特征
        
        Args:
            configs: 配置列表
            model_params: 模型参数量
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        results = []
        
        for i, config in enumerate(configs):
            result = {
                'config_name': f"config_{i+1}",
                'config': config.to_dict(),
                'memory_mb': config.get_memory_estimate(model_params),
                'effective_rank': config.get_effective_rank(),
                'complexity_score': config.rank * len(config.target_modules),
            }
            results.append(result)
        
        return {
            'comparison': results,
            'summary': {
                'min_memory': min(r['memory_mb'] for r in results),
                'max_memory': max(r['memory_mb'] for r in results),
                'min_complexity': min(r['complexity_score'] for r in results),
                'max_complexity': max(r['complexity_score'] for r in results),
            }
        }
    
    def get_parameter_importance(self, config: LoraConfig) -> Dict[str, float]:
        """
        计算参数重要性评分
        
        Args:
            config: LoRA配置对象
            
        Returns:
            Dict[str, float]: 参数重要性评分
        """
        importance_scores = {}
        
        # 基于参数值计算重要性
        importance_scores['rank'] = min(config.rank / 32, 1.0)  # rank越高越重要
        importance_scores['lora_alpha'] = min(config.lora_alpha / 128, 1.0)  # alpha越高越重要
        importance_scores['learning_rate'] = 1.0 - min(abs(np.log10(config.learning_rate) + 4) / 4, 1.0)  # 学习率适中最好
        
        # 基于任务类型调整
        task_weights = {
            'ENTITY_EXTRACTION': {'rank': 1.2, 'lora_alpha': 1.0, 'learning_rate': 1.1},
            'RELATION_CLASSIFICATION': {'rank': 1.0, 'lora_alpha': 1.1, 'learning_rate': 1.0},
            'JOINT_LEARNING': {'rank': 1.3, 'lora_alpha': 1.2, 'learning_rate': 1.2},
            'DOCUMENT_QA': {'rank': 1.1, 'lora_alpha': 1.3, 'learning_rate': 1.3},
        }
        
        if config.task_type in task_weights:
            weights = task_weights[config.task_type]
            for param in importance_scores:
                if param in weights:
                    importance_scores[param] *= weights[param]
        
        # 归一化
        total_score = sum(importance_scores.values())
        if total_score > 0:
            importance_scores = {k: v/total_score for k, v in importance_scores.items()}
        
        return importance_scores
    
    def export_config_report(self, config: LoraConfig, 
                           output_path: str, 
                           model_params: Optional[int] = None) -> str:
        """
        导出配置报告
        
        Args:
            config: LoRA配置对象
            output_path: 输出文件路径
            model_params: 模型参数量（可选）
            
        Returns:
            str: 报告内容
        """
        report_lines = []
        report_lines.append("# LoRA配置报告")
        report_lines.append(f"生成时间: {Path(__file__).parent.name} - 2025-11-06")
        report_lines.append("")
        
        # 基本信息
        report_lines.append("## 基本信息")
        report_lines.append(f"- 任务类型: {config.task_type}")
        report_lines.append(f"- 模型规模: {config.model_size}")
        report_lines.append(f"- 随机种子: {config.seed}")
        report_lines.append("")
        
        # LoRA参数
        report_lines.append("## LoRA参数")
        report_lines.append(f"- Rank: {config.rank}")
        report_lines.append(f"- Alpha: {config.lora_alpha}")
        report_lines.append(f"- Dropout: {config.lora_dropout}")
        report_lines.append(f"- 有效Rank: {config.get_effective_rank()}")
        if config.use_rslora:
            report_lines.append("- 使用RSLoRA: 是")
        report_lines.append(f"- 初始化类型: {config.lora_init_type}")
        report_lines.append("")
        
        # 训练参数
        report_lines.append("## 训练参数")
        report_lines.append(f"- 学习率: {config.learning_rate}")
        report_lines.append(f"- 权重衰减: {config.weight_decay}")
        report_lines.append(f"- 最大轮数: {config.max_epochs}")
        report_lines.append(f"- 批次大小: {config.batch_size}")
        report_lines.append(f"- 梯度累积步数: {config.gradient_accumulation_steps}")
        report_lines.append(f"- 预热步数: {config.warmup_steps}")
        report_lines.append(f"- 最大梯度范数: {config.max_grad_norm}")
        report_lines.append("")
        
        # 目标模块
        report_lines.append("## 目标模块")
        for module in config.target_modules:
            report_lines.append(f"- {module}")
        report_lines.append("")
        
        if config.exclude_modules:
            report_lines.append("## 排除模块")
            for module in config.exclude_modules:
                report_lines.append(f"- {module}")
            report_lines.append("")
        
        # 性能预估
        if model_params:
            memory_mb = config.get_memory_estimate(model_params)
            report_lines.append("## 性能预估")
            report_lines.append(f"- 模型参数量: {model_params:,}")
            report_lines.append(f"- 预估额外内存: {memory_mb:.2f} MB")
            report_lines.append(f"- 复杂度评分: {config.rank * len(config.target_modules)}")
            report_lines.append("")
        
        # 参数重要性
        importance = self.get_parameter_importance(config)
        report_lines.append("## 参数重要性")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- {param}: {score:.3f}")
        report_lines.append("")
        
        # 建议
        report_lines.append("## 使用建议")
        suggestions = self._get_usage_suggestions(config)
        for suggestion in suggestions:
            report_lines.append(f"- {suggestion}")
        
        # 写入文件
        report_content = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content
    
    def _get_usage_suggestions(self, config: LoraConfig) -> List[str]:
        """生成使用建议"""
        suggestions = []
        
        # 基于配置参数的建议
        if config.rank > 16:
            suggestions.append("当前rank较高，建议在资源充足时使用")
        
        if config.learning_rate > 1e-3:
            suggestions.append("学习率较高，注意监控训练稳定性")
        elif config.learning_rate < 1e-5:
            suggestions.append("学习率较低，可能需要更多训练轮数")
        
        if config.lora_dropout > 0.2:
            suggestions.append("dropout较高，有助于防止过拟合")
        
        # 基于任务类型的建议
        task_suggestions = {
            'ENTITY_EXTRACTION': "实体抽取任务建议重点关注Precision和Recall的平衡",
            'RELATION_CLASSIFICATION': "关系分类任务注意处理类别不平衡问题",
            'JOINT_LEARNING': "联合学习任务需要平衡实体和关系的学习进度",
            'DOCUMENT_QA': "文档问答任务建议使用较大的batch size和较长的预热",
        }
        
        if config.task_type in task_suggestions:
            suggestions.append(task_suggestions[config.task_type])
        
        # 基于模型规模的建议
        if config.model_size == 'LARGE' or config.model_size == 'XL':
            suggestions.append("大模型建议使用梯度检查点以节省内存")
            suggestions.append("考虑使用混合精度训练以提高效率")
        
        return suggestions


# 便利函数
def create_balanced_config() -> LoraConfig:
    """创建平衡配置的便利函数"""
    manager = LoraConfigManager()
    return manager.get_preset("balanced")


def create_entity_config() -> LoraConfig:
    """创建实体抽取配置的便利函数"""
    manager = LoraConfigManager()
    return manager.get_preset("entity_extraction")


def create_relation_config() -> LoraConfig:
    """创建关系分类配置的便利函数"""
    manager = LoraConfigManager()
    return manager.get_preset("relation_classification")


def create_joint_config() -> LoraConfig:
    """创建联合学习配置的便利函数"""
    manager = LoraConfigManager()
    return manager.get_preset("joint_learning")


if __name__ == "__main__":
    # 示例用法
    print("LoRA配置管理系统示例")
    print("=" * 50)
    
    # 创建配置管理器
    manager = LoraConfigManager()
    
    # 列出可用预设
    print("可用预设配置:")
    for name, desc in manager.get_preset_info().items():
        print(f"  - {name}: {desc}")
    
    print()
    
    # 创建不同类型的配置
    configs = {
        "平衡配置": create_balanced_config(),
        "实体抽取": create_entity_config(),
        "关系分类": create_relation_config(),
        "联合学习": create_joint_config()
    }
    
    for name, config in configs.items():
        print(f"{name}:")
        print(f"  Rank: {config.rank}, Alpha: {config.lora_alpha}")
        print(f"  学习率: {config.learning_rate}, 任务: {config.task_type}")
        print()
    
    # 保存配置示例
    balanced_config = create_balanced_config()
    manager.save_config(balanced_config, "example_config")
    print("示例配置已保存到 example_config.json")
    
    # 生成报告示例
    try:
        report = manager.export_config_report(balanced_config, "config_report.md", model_params=1000000000)
        print("配置报告已生成: config_report.md")
    except Exception as e:
        print(f"生成报告时出错: {e}")
    
    print("\n配置管理模块就绪！")