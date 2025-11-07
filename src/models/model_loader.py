"""
模型加载器模块

此模块提供了完整的CodeLlama-7B模型加载功能，包括：
- 4位量化加载（bitsandbytes）
- 分片加载和内存优化
- 模型安全检查和状态监控
- 多GPU并行加载
- 详细的异常处理和错误恢复

Author: DocRED Project Team
Date: 2025-11-06
"""

import os
import gc
import logging
import psutil
import torch
import time
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
from functools import wraps

# 尝试导入必要的库
try:
    import bitsandbytes as bnb
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        AutoConfig
    )
    BNB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}. 部分功能将受限。")
    BNB_AVAILABLE = False

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ModelLoadStatus:
    """模型加载状态跟踪"""
    model_name: str
    status: str  # 'loading', 'loaded', 'failed', 'partial'
    start_time: float
    end_time: Optional[float] = None
    memory_usage: Dict[str, float] = None
    gpu_usage: Dict[int, float] = None
    error_message: Optional[str] = None
    checkpoints_loaded: List[str] = None
    total_parameters: Optional[int] = None
    loaded_parameters: Optional[int] = None
    
    def __post_init__(self):
        if self.memory_usage is None:
            self.memory_usage = {}
        if self.gpu_usage is None:
            self.gpu_usage = {}
        if self.checkpoints_loaded is None:
            self.checkpoints_loaded = []
    
    @property
    def duration(self) -> Optional[float]:
        """获取加载耗时"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['duration'] = self.duration
        return data


class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.peak_memory = {}
        self.current_memory = {}
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控内存使用"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("内存监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU内存
                memory_info = process.memory_info()
                self.current_memory['cpu_rss'] = memory_info.rss / (1024**3)  # GB
                self.current_memory['cpu_vms'] = memory_info.vms / (1024**3)  # GB
                
                # GPU内存
                if torch.cuda.is_available():
                    for gpu_id in range(torch.cuda.device_count()):
                        gpu_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                        self.current_memory[f'gpu_{gpu_id}'] = gpu_memory
                        
                        # 更新峰值
                        if f'gpu_{gpu_id}' not in self.peak_memory:
                            self.peak_memory[f'gpu_{gpu_id}'] = 0.0
                        self.peak_memory[f'gpu_{gpu_id}'] = max(
                            self.peak_memory[f'gpu_{gpu_id}'], 
                            gpu_memory
                        )
                
                # 系统内存
                system_memory = psutil.virtual_memory()
                self.current_memory['system_used'] = system_memory.used / (1024**3)
                self.current_memory['system_available'] = system_memory.available / (1024**3)
                
                time.sleep(0.5)  # 0.5秒间隔
                
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                break
    
    def get_memory_report(self) -> Dict[str, float]:
        """获取内存报告"""
        report = {
            'current': self.current_memory.copy(),
            'peak': self.peak_memory.copy()
        }
        
        # CPU峰值内存
        if 'cpu_rss' not in self.peak_memory:
            self.peak_memory['cpu_rss'] = self.current_memory.get('cpu_rss', 0.0)
        else:
            self.peak_memory['cpu_rss'] = max(
                self.peak_memory['cpu_rss'], 
                self.current_memory.get('cpu_rss', 0.0)
            )
        
        return report


def error_handler(func):
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            # 根据错误类型提供具体的恢复建议
            if isinstance(e, torch.cuda.OutOfMemoryError):
                logger.warning("CUDA内存不足，建议减少批处理大小或使用更激进的量化")
            elif isinstance(e, FileNotFoundError):
                logger.error("模型文件未找到，请检查模型路径")
            elif isinstance(e, PermissionError):
                logger.error("文件权限不足，请检查文件访问权限")
            
            raise
    return wrapper


class ModelLoader:
    """CodeLlama-7B模型加载器
    
    提供高效率的CodeLlama-7B模型加载功能，支持：
    - 4位量化以节省内存
    - 分片加载以处理大模型
    - 多GPU并行加载
    - 实时内存监控
    - 安全检查和错误恢复
    """
    
    SUPPORTED_MODELS = [
        "codellama/CodeLlama-7b-Instruct-hf",
        "codellama/CodeLlama-7b-Python-hf",
        "codellama/CodeLlama-7b-hf",
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "meta-llama/CodeLlama-7b-Python-hf",
        "meta-llama/CodeLlama-7b-hf"
    ]
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
        device_map: Optional[Union[str, Dict[str, str]]] = "auto",
        trust_remote_code: bool = True,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: torch.dtype = torch.float16,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        max_memory_per_gpu: Optional[str] = "6GB",
        offload_folder: Optional[str] = None,
        max_shard_size: str = "1GB"
    ):
        """
        初始化模型加载器
        
        Args:
            model_name: 模型名称或路径
            device_map: 设备映射配置
            trust_remote_code: 是否信任远程代码
            use_auth_token: Hugging Face认证token
            cache_dir: 模型缓存目录
            load_in_8bit: 是否使用8位量化
            load_in_4bit: 是否使用4位量化
            bnb_4bit_compute_dtype: 4位量化计算数据类型
            bnb_4bit_use_double_quant: 是否使用双量化
            bnb_4bit_quant_type: 4位量化类型
            max_memory_per_gpu: 每GPU最大内存限制
            offload_folder: CPU卸载文件夹
            max_shard_size: 分片最大大小
        """
        self.model_name = model_name
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_auth_token = use_auth_token
        self.cache_dir = cache_dir
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.max_memory_per_gpu = max_memory_per_gpu
        self.offload_folder = offload_folder
        self.max_shard_size = max_shard_size
        
        self.tokenizer = None
        self.model = None
        self.config = None
        self.load_status = None
        self.memory_monitor = MemoryMonitor()
        
        logger.info(f"ModelLoader初始化完成: {model_name}")
    
    @error_handler
    def _validate_model(self) -> bool:
        """验证模型配置和依赖"""
        logger.info("开始验证模型配置...")
        
        # 检查模型是否支持
        if self.model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"模型 {self.model_name} 不在推荐列表中，但将继续尝试加载")
        
        # 检查bitsandbytes是否可用
        if not BNB_AVAILABLE:
            logger.warning("bitsandbytes库不可用，将尝试使用标准加载")
            if self.load_in_4bit or self.load_in_8bit:
                logger.error("4位/8位量化需要bitsandbytes库，请安装: pip install bitsandbytes")
                return False
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"检测到 {gpu_count} 个GPU设备")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name}, {gpu_memory:.1f}GB")
        else:
            logger.warning("未检测到CUDA设备，将使用CPU加载（性能可能较慢）")
        
        # 检查磁盘空间
        if self.cache_dir:
            disk_usage = psutil.disk_usage(self.cache_dir)
            free_space_gb = disk_usage.free / (1024**3)
            logger.info(f"缓存目录可用空间: {free_space_gb:.1f}GB")
            if free_space_gb < 10:
                logger.warning("可用空间不足10GB，可能影响模型加载")
        
        logger.info("模型配置验证完成")
        return True
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if not BNB_AVAILABLE:
            return None
        
        if self.load_in_4bit:
            logger.info("配置4位量化...")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            )
        elif self.load_in_8bit:
            logger.info("配置8位量化...")
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None
    
    def _get_device_map(self) -> Union[str, Dict[str, str]]:
        """获取设备映射"""
        if isinstance(self.device_map, dict):
            return self.device_map
        
        if self.device_map == "auto":
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    logger.info(f"检测到多GPU，启用自动设备映射")
                    return "auto"
                else:
                    logger.info("单GPU配置")
                    return {"": 0}
            else:
                logger.info("使用CPU")
                return "cpu"
        
        return self.device_map
    
    @contextmanager
    def _memory_optimization_context(self):
        """内存优化上下文"""
        logger.info("启用内存优化模式...")
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        # 开始监控
        self.memory_monitor.start_monitoring()
        
        try:
            yield
        finally:
            # 停止监控并清理
            self.memory_monitor.stop_monitoring()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
    
    @error_handler
    def load_model(
        self,
        progress_callback: Optional[callable] = None
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        加载CodeLlama-7B模型和分词器
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            tuple: (tokenizer, model)
            
        Raises:
            RuntimeError: 模型加载失败时抛出
        """
        start_time = time.time()
        self.load_status = ModelLoadStatus(
            model_name=self.model_name,
            status='loading',
            start_time=start_time
        )
        
        logger.info(f"开始加载模型: {self.model_name}")
        
        try:
            # 验证配置
            if not self._validate_model():
                raise RuntimeError("模型配置验证失败")
            
            with self._memory_optimization_context():
                # 加载配置
                logger.info("加载模型配置...")
                self.config = AutoConfig.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    use_auth_token=self.use_auth_token,
                    cache_dir=self.cache_dir
                )
                
                if progress_callback:
                    progress_callback("配置文件加载完成", 10)
                
                # 获取量化配置
                quantization_config = self._get_quantization_config()
                
                # 获取设备映射
                device_map = self._get_device_map()
                
                logger.info("开始加载分词器...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    use_auth_token=self.use_auth_token,
                    cache_dir=self.cache_dir
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                if progress_callback:
                    progress_callback("分词器加载完成", 20)
                
                logger.info("开始加载模型...")
                
                # 准备加载参数
                model_kwargs = {
                    'config': self.config,
                    'trust_remote_code': self.trust_remote_code,
                    'device_map': device_map,
                    'offload_folder': self.offload_folder,
                    'max_shard_size': self.max_shard_size,
                }
                
                # 添加量化配置
                if quantization_config:
                    model_kwargs['quantization_config'] = quantization_config
                elif not torch.cuda.is_available():
                    # CPU加载时的优化
                    model_kwargs['torch_dtype'] = torch.float32
                
                # 添加认证信息
                if self.use_auth_token:
                    model_kwargs['use_auth_token'] = self.use_auth_token
                
                if progress_callback:
                    progress_callback("模型加载中...", 30)
                
                # 执行模型加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                if progress_callback:
                    progress_callback("模型加载完成", 80)
                
                # 设置模型为评估模式
                self.model.eval()
                
                # 更新加载状态
                self.load_status.status = 'loaded'
                self.load_status.end_time = time.time()
                
                # 获取内存使用情况
                memory_report = self.memory_monitor.get_memory_report()
                self.load_status.memory_usage = memory_report['current']
                self.load_status.gpu_usage = {k: v for k, v in memory_report['current'].items() 
                                             if k.startswith('gpu_')}
                
                # 计算参数数量
                if hasattr(self.model, 'num_parameters'):
                    self.load_status.total_parameters = self.model.num_parameters()
                    logger.info(f"模型参数总数: {self.load_status.total_parameters:,}")
                
                if progress_callback:
                    progress_callback("模型加载完成", 100)
                
                logger.info(f"模型加载成功，耗时: {self.load_status.duration:.2f}秒")
                
                return self.tokenizer, self.model
                
        except Exception as e:
            self.load_status.status = 'failed'
            self.load_status.end_time = time.time()
            self.load_status.error_message = str(e)
            
            logger.error(f"模型加载失败: {str(e)}")
            
            # 尝试错误恢复
            self._attempt_recovery()
            
            raise RuntimeError(f"模型加载失败: {str(e)}") from e
    
    def _attempt_recovery(self):
        """尝试错误恢复"""
        logger.info("尝试错误恢复...")
        
        recovery_strategies = [
            "清理CUDA缓存",
            "释放GPU内存",
            "尝试更保守的加载配置"
        ]
        
        try:
            # 清理资源
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("资源清理完成")
            
            # 记录恢复尝试
            self.load_status.checkpoints_loaded.extend(recovery_strategies)
            
        except Exception as e:
            logger.error(f"错误恢复失败: {str(e)}")
    
    @error_handler
    def load_sharded_model(
        self,
        checkpoint_dir: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        加载分片模型
        
        Args:
            checkpoint_dir: 分片检查点目录
            progress_callback: 进度回调函数
            
        Returns:
            tuple: (tokenizer, model)
        """
        logger.info(f"开始加载分片模型: {checkpoint_dir}")
        
        try:
            # 验证检查点目录
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(f"检查点目录不存在: {checkpoint_dir}")
            
            # 查找分片文件
            shard_files = []
            for file in os.listdir(checkpoint_dir):
                if file.startswith('pytorch_model-') and file.endswith('.bin.index.json'):
                    index_file = os.path.join(checkpoint_dir, file)
                    shard_files.append(index_file)
                    break
            
            if not shard_files:
                # 查找普通分片文件
                for file in os.listdir(checkpoint_dir):
                    if file.startswith('pytorch_model-') and file.endswith('.bin'):
                        shard_files.append(os.path.join(checkpoint_dir, file))
            
            if not shard_files:
                raise FileNotFoundError("未找到分片文件")
            
            logger.info(f"找到 {len(shard_files)} 个分片文件")
            
            # 加载分片模型
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                device_map=self._get_device_map(),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=self.trust_remote_code,
                offload_folder=self.offload_folder
            )
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                trust_remote_code=self.trust_remote_code
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("分片模型加载完成")
            
            return self.tokenizer, self.model
            
        except Exception as e:
            logger.error(f"分片模型加载失败: {str(e)}")
            raise
    
    @error_handler
    def save_model(self, save_directory: str):
        """
        保存模型到指定目录
        
        Args:
            save_directory: 保存目录
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，无法保存")
        
        logger.info(f"开始保存模型到: {save_directory}")
        
        try:
            # 创建目录
            os.makedirs(save_directory, exist_ok=True)
            
            # 保存模型和分词器
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            
            # 保存加载配置
            config_info = {
                'model_name': self.model_name,
                'load_config': {
                    'load_in_4bit': self.load_in_4bit,
                    'load_in_8bit': self.load_in_8bit,
                    'device_map': str(self.device_map),
                    'bnb_4bit_compute_dtype': str(self.bnb_4bit_compute_dtype),
                    'bnb_4bit_use_double_quant': self.bnb_4bit_use_double_quant,
                    'bnb_4bit_quant_type': self.bnb_4bit_quant_type
                },
                'load_status': self.load_status.to_dict() if self.load_status else None
            }
            
            config_path = os.path.join(save_directory, 'loader_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模型保存完成: {save_directory}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取当前内存使用情况"""
        memory_report = self.memory_monitor.get_memory_report()
        
        return {
            'current_usage': memory_report['current'],
            'peak_usage': memory_report['peak'],
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def get_load_status(self) -> Optional[ModelLoadStatus]:
        """获取模型加载状态"""
        return self.load_status
    
    def unload_model(self):
        """卸载模型，释放内存"""
        logger.info("开始卸载模型...")
        
        try:
            # 停止监控
            self.memory_monitor.stop_monitoring()
            
            # 清理模型引用
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if hasattr(self, 'config') and self.config is not None:
                del self.config
                self.config = None
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 垃圾回收
            gc.collect()
            
            logger.info("模型卸载完成")
            
        except Exception as e:
            logger.error(f"模型卸载时出错: {str(e)}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.unload_model()


def create_model_loader(
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
    **kwargs
) -> ModelLoader:
    """
    创建模型加载器的便捷函数
    
    Args:
        model_name: 模型名称
        **kwargs: 其他加载参数
        
    Returns:
        ModelLoader实例
    """
    return ModelLoader(model_name=model_name, **kwargs)


# 示例使用函数
def example_usage():
    """模型加载器使用示例"""
    
    # 基本用法
    with create_model_loader(
        model_name="codellama/CodeLlama-7b-Instruct-hf",
        load_in_4bit=True,
        max_memory_per_gpu="6GB"
    ) as loader:
        
        def progress_callback(message: str, progress: int):
            print(f"[{progress}%] {message}")
        
        tokenizer, model = loader.load_model(progress_callback=progress_callback)
        
        # 获取加载状态
        status = loader.get_load_status()
        print(f"加载状态: {status.status}")
        print(f"加载耗时: {status.duration:.2f}秒")
        
        # 获取内存使用情况
        memory_usage = loader.get_memory_usage()
        print(f"当前内存使用: {memory_usage['current_usage']}")
        
        # 保存模型
        # loader.save_model("./saved_model")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行示例
    example_usage()