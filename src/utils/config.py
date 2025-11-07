#!/usr/bin/env python3
"""
配置管理模块
用于加载和管理系统配置
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProjectConfig:
    """项目配置"""
    name: str
    version: str
    author: str
    date: str

@dataclass
class DataConfig:
    """数据配置"""
    raw: Dict[str, str]
    processed: Dict[str, str]
    output: Dict[str, str]
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径访问"""
        keys = key.split('.')
        value = self.__dict__
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

@dataclass
class ModelConfig:
    """模型配置"""
    base: Dict[str, Any]
    fine_tuned: Dict[str, Any]

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_steps: int
    max_seq_length: int
    gradient_accumulation_steps: int
    save_steps: int
    eval_steps: int
    early_stopping: Dict[str, Any]

@dataclass
class PreprocessingConfig:
    """预处理配置"""
    max_length: int
    max_entities: int
    max_relations: int
    train_ratio: float
    val_ratio: float
    test_ratio: float

@dataclass
class PromptConfig:
    """提示模板配置"""
    text2pseudocode: Dict[str, Any]

@dataclass
class ExtractionConfig:
    """要素抽取配置"""
    ast: Dict[str, Any]
    entity_types: list

@dataclass
class KnowledgeConfig:
    """知识融合配置"""
    knowledge_graph: Dict[str, Any]
    confidence_threshold: float
    max_candidates: int

@dataclass
class InferenceConfig:
    """推理配置"""
    lmulator: Dict[str, Any]
    validation: Dict[str, Any]

@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: list
    thresholds: Dict[str, float]

@dataclass
class GUIConfig:
    """GUI配置"""
    window_size: list
    theme: str
    font_size: int
    max_text_length: int
    supported_models: list

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str
    format: str
    file_path: str
    max_size: str
    backup_count: int

@dataclass
class HardwareConfig:
    """硬件配置"""
    gpu_memory_threshold: float
    batch_processing: bool
    num_workers: int
    pin_memory: bool

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self._loaded = False
        
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                logger.error(f"配置文件不存在: {self.config_path}")
                return False
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            self._config = config_data
            self._loaded = True
            
            logger.info(f"配置文件加载成功: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False
    
    @property
    def project(self) -> ProjectConfig:
        """获取项目配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['project']
        return ProjectConfig(**data)
    
    @property
    def data(self) -> DataConfig:
        """获取数据配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['data']
        return DataConfig(**data)
    
    @property
    def models(self) -> ModelConfig:
        """获取模型配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['models']
        return ModelConfig(**data)
    
    @property
    def training(self) -> TrainingConfig:
        """获取训练配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['training']
        return TrainingConfig(**data)
    
    @property
    def preprocessing(self) -> PreprocessingConfig:
        """获取预处理配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['preprocessing']
        return PreprocessingConfig(**data)
    
    @property
    def prompt_templates(self) -> PromptConfig:
        """获取提示模板配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['prompt_templates']
        return PromptConfig(**data)
    
    @property
    def extraction(self) -> ExtractionConfig:
        """获取要素抽取配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['extraction']
        return ExtractionConfig(**data)
    
    @property
    def knowledge_fusion(self) -> KnowledgeConfig:
        """获取知识融合配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['knowledge_fusion']
        return KnowledgeConfig(**data)
    
    @property
    def inference(self) -> InferenceConfig:
        """获取推理配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['inference']
        return InferenceConfig(**data)
    
    @property
    def evaluation(self) -> EvaluationConfig:
        """获取评估配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['evaluation']
        return EvaluationConfig(**data)
    
    @property
    def gui(self) -> GUIConfig:
        """获取GUI配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['gui']
        return GUIConfig(**data)
    
    @property
    def logging(self) -> LoggingConfig:
        """获取日志配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['logging']
        return LoggingConfig(**data)
    
    @property
    def hardware(self) -> HardwareConfig:
        """获取硬件配置"""
        if not self._loaded:
            self.load_config()
        data = self._config['hardware']
        return HardwareConfig(**data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if not self._loaded:
            self.load_config()
        
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        if not self._loaded:
            self.load_config()
            
        keys = key.split('.')
        config = self._config
        
        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None):
        """保存配置到文件"""
        if not self._loaded:
            logger.error("配置未加载，无法保存")
            return False
            
        path = output_path or str(self.config_path)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logger.info(f"配置已保存到: {path}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False


# 全局配置实例
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """获取全局配置管理器实例"""
    return config_manager


# 便捷函数
def get_project_config() -> ProjectConfig:
    """获取项目配置"""
    return config_manager.project

def get_data_config() -> DataConfig:
    """获取数据配置"""
    return config_manager.data

def get_model_config() -> ModelConfig:
    """获取模型配置"""
    return config_manager.models

def get_training_config() -> TrainingConfig:
    """获取训练配置"""
    return config_manager.training

def get_gui_config() -> GUIConfig:
    """获取GUI配置"""
    return config_manager.gui