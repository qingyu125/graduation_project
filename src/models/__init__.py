"""
DocRED 论文项目模型模块
===================

本模块提供 DocRED 关系抽取任务的完整模型系统，包括基线模型、微调模型和实用工具。

Author: DocRED Project Team
Version: 1.0.0
"""

# 模块版本信息
__version__ = "1.0.0"
__author__ = "DocRED Project Team"
__email__ = "docred@example.com"
__description__ = "DocRED 关系抽取模型系统"

# 核心依赖库导入
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. 基础数据结构和配置
# =============================================================================

@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str
    model_type: str
    hidden_size: int = 768
    num_classes: int = 97  # DocRED关系类型数量
    max_length: int = 512
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "max_length": self.max_length,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "device": self.device
        }

@dataclass  
class TrainingConfig:
    """训练配置类"""
    output_dir: str = "./output"
    log_dir: str = "./logs"
    model_save_dir: str = "./models"
    save_interval: int = 5
    eval_interval: int = 1
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    save_total_limit: int = 3

# =============================================================================
# 2. 抽象基类
# =============================================================================

class BaseModel(ABC, nn.Module):
    """所有模型的基础抽象类"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.model_type = config.model_type
        
    @abstractmethod
    def forward(self, **kwargs):
        """前向传播"""
        pass
        
    @abstractmethod
    def predict(self, **kwargs):
        """预测"""
        pass
        
    @abstractmethod
    def load_pretrained(self, path: str):
        """加载预训练权重"""
        pass
        
    def save_model(self, save_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'model_name': self.model_name,
            'model_type': self.model_type
        }, save_path)
        logger.info(f"模型已保存到: {save_path}")

class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(self, model: BaseModel, train_config: TrainingConfig):
        self.model = model
        self.config = train_config
        self.device = model.config.device
        
    @abstractmethod
    def train(self, train_loader, val_loader=None):
        """训练"""
        pass
        
    @abstractmethod
    def evaluate(self, test_loader):
        """评估"""
        pass
        
    def save_checkpoint(self, epoch: int, save_path: str, metrics: Dict = None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config.to_dict(),
            'train_config': self.config.__dict__,
            'metrics': metrics or {}
        }
        torch.save(checkpoint, save_path)
        logger.info(f"检查点已保存: {save_path}")

# =============================================================================
# 3. 模型工厂和工具函数
# =============================================================================

class ModelFactory:
    """模型工厂类，用于创建各种模型实例"""
    
    _MODELS = {}
    
    @classmethod
    def register(cls, model_type: str):
        """装饰器：注册模型类型"""
        def decorator(model_class):
            cls._MODELS[model_type] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create_model(cls, model_type: str, config: ModelConfig) -> BaseModel:
        """根据类型创建模型"""
        if model_type not in cls._MODELS:
            raise ValueError(f"未知的模型类型: {model_type}")
        return cls._MODELS[model_type](config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取可用的模型类型"""
        return list(cls._MODELS.keys())

class ModelLoader:
    """模型加载器"""
    
    @staticmethod
    def load_model(model_path: str, device: str = "auto") -> Tuple[BaseModel, Dict]:
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=device)
        
        # 创建配置
        config = ModelConfig(**checkpoint['config'])
        if device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            config.device = device
            
        # 创建模型
        model = ModelFactory.create_model(checkpoint['model_type'], config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        
        logger.info(f"模型加载成功: {checkpoint['model_name']} ({checkpoint['model_type']})")
        return model, checkpoint
    
    @staticmethod
    def load_pretrained_model(model_name: str, cache_dir: str = None) -> BaseModel:
        """加载预训练模型"""
        # 这里是预训练模型的逻辑
        logger.info(f"正在加载预训练模型: {model_name}")
        # 实际实现时需要根据具体模型调整
        raise NotImplementedError("预训练模型加载功能待实现")

class ModelUtils:
    """模型工具函数集合"""
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, Any]:
        """获取模型大小信息"""
        param_size = 0
        param_size_grad = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            if param.requires_grad:
                param_size_grad += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = 1024 ** 2
        return {
            "model_size_mb": param_size / size_mb,
            "model_size_grad_mb": param_size_grad / size_mb,
            "buffer_size_mb": buffer_size / size_mb,
            "total_size_mb": (param_size + buffer_size) / size_mb,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters())
        }
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str] = None):
        """冻结指定层"""
        if layer_names is None:
            # 冻结所有参数
            for param in model.parameters():
                param.requires_grad = False
        else:
            # 冻结指定层
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
        
        logger.info(f"已冻结层: {layer_names or '所有层'}")
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: List[str] = None):
        """解冻指定层"""
        if layer_names is None:
            # 解冻所有参数
            for param in model.parameters():
                param.requires_grad = True
        else:
            # 解冻指定层
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
        
        logger.info(f"已解冻层: {layer_names or '所有层'}")

# =============================================================================
# 4. 便捷接口函数
# =============================================================================

def create_model(model_type: str, **config_kwargs) -> BaseModel:
    """
    创建模型的便捷接口
    
    Args:
        model_type: 模型类型
        **config_kwargs: 配置参数
        
    Returns:
        创建的模型实例
    """
    config = ModelConfig(model_name=f"{model_type}_model", model_type=model_type, **config_kwargs)
    return ModelFactory.create_model(model_type, config)

def load_model_from_path(model_path: str, device: str = "auto") -> BaseModel:
    """
    从路径加载模型的便捷接口
    
    Args:
        model_path: 模型文件路径
        device: 设备类型
        
    Returns:
        加载的模型实例
    """
    model, _ = ModelLoader.load_model(model_path, device)
    return model

def get_model_info(model: BaseModel) -> Dict[str, Any]:
    """
    获取模型信息的便捷接口
    
    Args:
        model: 模型实例
        
    Returns:
        模型信息字典
    """
    size_info = ModelUtils.get_model_size(model)
    config_info = model.config.to_dict()
    
    return {
        "model_name": model.model_name,
        "model_type": model.model_type,
        "config": config_info,
        "size_info": size_info
    }

def list_available_models() -> List[str]:
    """
    获取可用模型列表的便捷接口
    
    Returns:
        可用模型类型列表
    """
    return ModelFactory.get_available_models()

def create_trainer(model: BaseModel, train_config: TrainingConfig = None) -> BaseTrainer:
    """
    创建训练器的便捷接口
    
    Args:
        model: 模型实例
        train_config: 训练配置
        
    Returns:
        训练器实例
    """
    if train_config is None:
        train_config = TrainingConfig()
    
    # 如果有可用的训练器模块
    if TRAINER_AVAILABLE:
        return DocRedTrainer(model, train_config)
    else:
        # 返回基础训练器
        raise NotImplementedError("训练器模块不可用，请检查trainer模块导入")

def create_evaluator(model: BaseModel = None) -> 'DocREDEvaluator':
    """
    创建评估器的便捷接口
    
    Args:
        model: 可选的模型实例
        
    Returns:
        评估器实例
    """
    if EVALUATOR_AVAILABLE:
        return DocREDEvaluator(model)
    else:
        raise NotImplementedError("评估器模块不可用，请检查evaluator模块导入")

# =============================================================================
# 5. 快速原型支持
# =============================================================================

class QuickModel:
    """快速模型原型类"""
    
    @staticmethod
    def bert_based_model(config: ModelConfig = None) -> BaseModel:
        """创建基于BERT的快速模型"""
        if config is None:
            config = ModelConfig(
                model_name="quick_bert_model",
                model_type="bert_based",
                hidden_size=768
            )
        # 这里应该返回实际的BERT模型实例
        raise NotImplementedError("BERT模型待实现")
    
    @staticmethod
    def cnn_model(config: ModelConfig = None) -> BaseModel:
        """创建基于CNN的快速模型"""
        if config is None:
            config = ModelConfig(
                model_name="quick_cnn_model", 
                model_type="cnn_based",
                hidden_size=256
            )
        # 这里应该返回实际的CNN模型实例
        raise NotImplementedError("CNN模型待实现")
    
    @staticmethod
    def lstm_model(config: ModelConfig = None) -> BaseModel:
        """创建基于LSTM的快速模型"""
        if config is None:
            config = ModelConfig(
                model_name="quick_lstm_model",
                model_type="lstm_based", 
                hidden_size=512
            )
        # 这里应该返回实际的LSTM模型实例
        raise NotImplementedError("LSTM模型待实现")

# =============================================================================
# 6. 导入子模块
# =============================================================================

# 导入基线模型
try:
    from . import baselines
    BASELINES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"基线模型模块导入失败: {e}")
    BASELINES_AVAILABLE = False

# 导入微调模型
try:
    from . import fine_tuned
    FINE_TUNED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"微调模型模块导入失败: {e}")
    FINE_TUNED_AVAILABLE = False

# 导入工具模块
try:
    from . import utils
    UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"工具模块导入失败: {e}")
    UTILS_AVAILABLE = False

# 导入现有模块
try:
    from .evaluator import DocREDEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"评估器模块导入失败: {e}")
    EVALUATOR_AVAILABLE = False

try:
    from .trainer import DocRedTrainer
    TRAINER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"训练器模块导入失败: {e}")
    TRAINER_AVAILABLE = False

try:
    from .model_loader import ModelLoader as CoreModelLoader
    CORE_MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"核心模型加载器导入失败: {e}")
    CORE_MODEL_LOADER_AVAILABLE = False

try:
    from .loss_functions import *
    LOSS_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"损失函数模块导入失败: {e}")
    LOSS_FUNCTIONS_AVAILABLE = False

# =============================================================================
# 7. 统一导出接口
# =============================================================================

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__", 
    "__description__",
    
    # 配置类
    "ModelConfig",
    "TrainingConfig",
    
    # 抽象基类
    "BaseModel",
    "BaseTrainer",
    
    # 工厂和加载器
    "ModelFactory",
    "ModelLoader", 
    "ModelUtils",
    
    # 便捷接口
    "create_model",
    "load_model_from_path", 
    "get_model_info",
    "list_available_models",
    "create_trainer",
    "create_evaluator",
    
    # 快速原型
    "QuickModel",
    
    # 状态标识
    "BASELINES_AVAILABLE",
    "FINE_TUNED_AVAILABLE", 
    "UTILS_AVAILABLE",
    "EVALUATOR_AVAILABLE",
    "TRAINER_AVAILABLE",
    "CORE_MODEL_LOADER_AVAILABLE",
    "LOSS_FUNCTIONS_AVAILABLE"
]

# =============================================================================
# 8. 模块初始化
# =============================================================================

def _check_dependencies():
    """检查依赖项"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        logger.warning(f"缺少以下依赖: {missing_deps}")
        logger.warning("请使用 pip install 安装这些依赖")
    
    return len(missing_deps) == 0

def _print_banner():
    """打印模块信息横幅"""
    banner = f"""
    ═══════════════════════════════════════════════════════════
    DocRED 关系抽取模型系统 v{__version__}
    
    模型类型: {len(ModelFactory._MODELS)} 种
    设备: {'GPU' if torch.cuda.is_available() else 'CPU'}
    PyTorch版本: {torch.__version__}
    ═══════════════════════════════════════════════════════════
    """
    print(banner)

# 模块初始化
def _init_module():
    """模块初始化"""
    _print_banner()
    _check_dependencies()
    
    if BASELINES_AVAILABLE:
        logger.info("基线模型模块可用")
    if FINE_TUNED_AVAILABLE:
        logger.info("微调模型模块可用") 
    if UTILS_AVAILABLE:
        logger.info("工具模块可用")
    if EVALUATOR_AVAILABLE:
        logger.info("评估器模块可用")
    if TRAINER_AVAILABLE:
        logger.info("训练器模块可用")
    if CORE_MODEL_LOADER_AVAILABLE:
        logger.info("核心模型加载器可用")
    if LOSS_FUNCTIONS_AVAILABLE:
        logger.info("损失函数模块可用")
    
    logger.info("DocRED 模型系统初始化完成")

# 自动执行模块初始化
_init_module()

# =============================================================================
# 使用示例
# =============================================================================

"""
使用示例:

# 1. 创建配置
config = ModelConfig(
    model_name="my_model",
    model_type="bert_based", 
    hidden_size=768,
    num_classes=97
)

# 2. 创建模型
model = create_model("bert_based", **config.to_dict())

# 3. 获取模型信息
info = get_model_info(model)
print(f"模型参数: {info['size_info']['total_params']}")

# 4. 加载预训练模型
model = load_model_from_path("./path/to/model.pth")

# 5. 快速原型
quick_model = QuickModel.bert_based_model()
"""

print("DocRED 模型模块已成功导入！")