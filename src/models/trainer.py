"""
DocRED 训练器主类
完整的训练循环，支持早停、学习率调度、梯度累积、混合精度训练等
"""

import os
import json
import time
import logging
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Optional, Tuple, Any, Union

# 可选依赖 - tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not available. Visualization features will be disabled.")
import numpy as np
from collections import defaultdict, deque
import warnings
from pathlib import Path
import pickle

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停策略类"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """检查是否应该早停"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            logger.info(f"早停触发！最佳验证损失: {self.best_loss:.4f}")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("恢复最佳权重")
            return True
        return False


class LearningRateScheduler:
    """学习率调度器类"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_type: str = 'step', **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler = self._create_scheduler(**kwargs)
        
    def _create_scheduler(self, **kwargs):
        """创建学习率调度器"""
        if self.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        elif self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', 100)
            )
        elif self.scheduler_type == 'warmup':
            return WarmupScheduler(
                self.optimizer,
                warmup_steps=kwargs.get('warmup_steps', 1000),
                total_steps=kwargs.get('total_steps', 10000)
            )
        else:
            logger.warning(f"未知的学习率调度器类型: {self.scheduler_type}，使用StepLR")
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
    
    def step(self):
        """步进学习率调度器"""
        self.scheduler.step()
        
    def get_last_lr(self) -> float:
        """获取当前学习率"""
        return self.scheduler.get_last_lr()[0]


class WarmupScheduler:
    """预热学习率调度器"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
        else:
            # 预热结束后保持不变
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr']
                
    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class MetricMonitor:
    """指标监控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_steps = defaultdict(int)
        
    def update(self, metric_name: str, value: float, step: Optional[int] = None):
        """更新指标"""
        self.metrics[metric_name].append(value)
        if step is not None:
            self.global_steps[metric_name] = step
            
    def get_average(self, metric_name: str) -> float:
        """获取平均指标"""
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return np.mean(self.metrics[metric_name])
        return 0.0
        
    def get_global_step(self, metric_name: str) -> int:
        """获取全局步数"""
        return self.global_steps.get(metric_name, 0)
        
    def get_dict(self) -> Dict[str, float]:
        """获取所有指标的平均值"""
        return {name: self.get_average(name) for name in self.metrics.keys()}


class MemoryMonitor:
    """内存监控器"""
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """获取内存使用统计"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,   # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        return {'allocated': 0.0, 'cached': 0.0, 'max_allocated': 0.0}
    
    @staticmethod
    def clear_cache():
        """清理缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, save_dir: str, keep_n_checkpoints: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_checkpoints = keep_n_checkpoints
        
    def save(self, 
             model: nn.Module, 
             optimizer: torch.optim.Optimizer, 
             scheduler: Any,
             epoch: int, 
             step: int,
             metrics: Dict[str, float],
             is_best: bool = False,
             extra_state: Optional[Dict] = None) -> str:
        """保存检查点"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'extra_state': extra_state or {}
        }
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型到 {best_path}")
            
        # 保存当前模型
        current_path = self.save_dir / f'checkpoint_epoch_{epoch}_step_{step}.pth'
        torch.save(checkpoint, current_path)
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        return str(current_path)
        
    def load(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
             scheduler: Optional[Any] = None, path: Optional[str] = None) -> Dict[str, Any]:
        """加载检查点"""
        if path is None:
            path = self.save_dir / 'best_model.pth'
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")
            
        logger.info(f"从 {path} 加载检查点")
        checkpoint = torch.load(path, map_location='cpu')
        
        # 加载模型
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 加载调度器
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'extra_state': checkpoint.get('extra_state', {})
        }
        
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = list(self.save_dir.glob('checkpoint_*.pth'))
        if len(checkpoints) > self.keep_n_checkpoints:
            # 按修改时间排序，删除最旧的
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for checkpoint in checkpoints[:-self.keep_n_checkpoints]:
                checkpoint.unlink()
                logger.info(f"删除旧检查点: {checkpoint}")


class DocRedTrainer:
    """DocRED训练器主类"""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 scheduler_config: Optional[Dict] = None,
                 early_stopping_config: Optional[Dict] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 distributed: bool = False,
                 mixed_precision: bool = True,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 log_interval: int = 10,
                 eval_interval: int = 1000,
                 save_dir: str = 'checkpoints',
                 use_tensorboard: bool = True,
                 use_wandb: bool = False,
                 logging_level: str = 'INFO'):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            scheduler_config: 学习率调度器配置
            early_stopping_config: 早停配置
            device: 设备
            distributed: 是否使用分布式训练
            mixed_precision: 是否使用混合精度训练
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪最大范数
            log_interval: 日志记录间隔
            eval_interval: 验证间隔
            save_dir: 保存目录
            use_tensorboard: 是否使用tensorboard
            use_wandb: 是否使用wandb
            logging_level: 日志级别
        """
        # 设置设备
        self.device = self._setup_device(device)
        
        # 设置分布式训练
        self.distributed = distributed
        if distributed:
            self._setup_distributed()
            
        # 设置模型
        self.model = self._setup_model(model)
        
        # 设置数据加载器
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 设置优化器和损失函数
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), 
            lr=2e-5, 
            weight_decay=0.01
        )
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # 设置学习率调度器
        self.scheduler = self._setup_scheduler(scheduler_config)
        
        # 设置早停
        self.early_stopping = self._setup_early_stopping(early_stopping_config)
        
        # 设置训练配置
        self.mixed_precision = mixed_precision and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.mixed_precision else None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # 设置监控和日志
        self.metric_monitor = MetricMonitor()
        self.memory_monitor = MemoryMonitor()
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # 设置保存和日志记录
        self.checkpoint_manager = CheckpointManager(save_dir)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        else:
            self.writer = None
            if use_tensorboard and not TENSORBOARD_AVAILABLE:
                logging.warning("TensorBoard requested but not available. Disabling tensorboard logging.")
            
        if use_wandb:
            try:
                import wandb
                wandb.init(project="docred")
                self.wandb = wandb
            except ImportError:
                logger.warning("Wandb未安装，禁用wandb日志")
                self.use_wandb = False
                self.wandb = None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0
        self.is_training = True
        
        # 内存清理
        if self.device.type == 'cuda':
            self.memory_monitor.clear_cache()
            
        logger.info("训练器初始化完成")
        
    def _setup_device(self, device) -> torch.device:
        """设置设备"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
            
        logger.info(f"使用设备: {device}")
        return device
        
    def _setup_distributed(self):
        """设置分布式训练"""
        if not dist.is_available():
            raise RuntimeError("PyTorch分布式训练不可用")
            
        if not dist.is_initialized():
            init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
            
        logger.info("设置分布式训练")
        
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """设置模型"""
        model = model.to(self.device)
        
        if self.distributed:
            # 包装为分布式模型
            if hasattr(model, 'module'):
                model = model.module
            model = DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()] if self.device.type == 'cuda' else None,
                find_unused_parameters=True
            )
        elif torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            # 多GPU数据并行
            model = DataParallel(model)
            
        return model
        
    def _setup_scheduler(self, scheduler_config: Optional[Dict] = None):
        """设置学习率调度器"""
        if scheduler_config is None:
            scheduler_config = {'scheduler_type': 'step', 'step_size': 30, 'gamma': 0.1}
            
        return LearningRateScheduler(
            self.optimizer,
            **scheduler_config
        )
        
    def _setup_early_stopping(self, early_stopping_config: Optional[Dict] = None):
        """设置早停"""
        if early_stopping_config is None:
            early_stopping_config = {'patience': 7, 'min_delta': 0.001}
            
        return EarlyStopping(**early_stopping_config)
        
    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            resume_from: 从指定检查点恢复训练
            
        Returns:
            训练结果字典
        """
        # 恢复训练
        if resume_from:
            self._load_checkpoint(resume_from)
            
        logger.info(f"开始训练，总共 {num_epochs} 轮")
        logger.info(f"混合精度: {self.mixed_precision}, 梯度累积步数: {self.gradient_accumulation_steps}")
        
        training_history = defaultdict(list)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            epoch_metrics = self._train_epoch()
            
            # 记录训练历史
            for metric_name, value in epoch_metrics.items():
                training_history[f'train_{metric_name}'].append(value)
                
            # 验证
            if self.val_dataloader is not None and (epoch + 1) % max(1, self.eval_interval // len(self.train_dataloader)) == 0:
                val_metrics = self._validate()
                
                # 记录验证历史
                for metric_name, value in val_metrics.items():
                    training_history[f'val_{metric_name}'].append(value)
                    
                # 早停检查
                if self.early_stopping:
                    should_stop = self.early_stopping(val_metrics.get('loss', float('inf')), self.model)
                    if should_stop:
                        logger.info("触发早停，停止训练")
                        break
                        
            # 调度学习率
            if self.scheduler:
                self.scheduler.step()
                
            # 保存检查点
            current_metric = epoch_metrics.get('loss', 0.0)
            is_best = False
            if self.val_dataloader is not None:
                current_metric = val_metrics.get('loss', float('inf'))
                
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                is_best = True
                
            self._save_checkpoint(epoch_metrics, is_best=is_best)
            
            # 输出训练信息
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_results(epoch, epoch_metrics, val_metrics if self.val_dataloader else None, epoch_time)
            
        # 训练完成
        logger.info("训练完成")
        self._cleanup()
        
        return {
            'training_history': dict(training_history),
            'best_metric': self.best_metric,
            'final_epoch': self.epoch
        }
        
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.is_training = True
        
        # 重置指标
        epoch_metrics = defaultdict(float)
        num_batches = len(self.train_dataloader)
        
        # 遍历批次
        for batch_idx, batch in enumerate(self.train_dataloader):
            batch_metrics = self._train_step(batch)
            
            # 累积指标
            for metric_name, value in batch_metrics.items():
                epoch_metrics[metric_name] += value
                
            # 日志输出
            if (batch_idx + 1) % self.log_interval == 0:
                self._log_batch_results(batch_idx, num_batches, batch_metrics)
                
            # 内存监控
            if self.device.type == 'cuda' and (batch_idx + 1) % 100 == 0:
                memory_stats = self.memory_monitor.get_memory_stats()
                if memory_stats['allocated'] > 10:  # 如果显存使用超过10GB
                    self.memory_monitor.clear_cache()
                    
        # 计算平均指标
        for metric_name in epoch_metrics:
            epoch_metrics[metric_name] /= num_batches
            
        return dict(epoch_metrics)
        
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练一个步骤"""
        # 移动数据到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # 前向传播
        if self.mixed_precision:
            with autocast():
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)
                
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.model(batch)
            loss = self._compute_loss(outputs, batch)
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            # 梯度裁剪
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        self.global_step += 1
        
        # 计算指标
        metrics = self._compute_metrics(outputs, batch, loss.item() * self.gradient_accumulation_steps)
        
        # 记录日志
        if self.writer:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'train_{metric_name}', value, self.global_step)
                
        if self.use_wandb and self.wandb:
            for metric_name, value in metrics.items():
                self.wandb.log({f'train_{metric_name}': value}, step=self.global_step)
                
        return metrics
        
    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失"""
        if isinstance(outputs, dict):
            # 如果输出是字典，根据具体任务计算损失
            if 'logits' in outputs:
                targets = batch.get('labels')
                if targets is not None:
                    loss = self.criterion(outputs['logits'], targets)
                else:
                    loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
            else:
                # 复合任务损失
                total_loss = 0
                for task_name, task_output in outputs.items():
                    if isinstance(task_output, dict) and 'loss' in task_output:
                        total_loss += task_output['loss']
                loss = total_loss
        else:
            # 简单情况
            targets = batch.get('labels')
            if targets is not None:
                loss = self.criterion(outputs, targets)
            else:
                loss = torch.tensor(0.0, device=self.device)
                
        return loss
        
    def _compute_metrics(self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor], 
                        loss: float) -> Dict[str, float]:
        """计算指标"""
        metrics = {'loss': loss}
        
        # 根据任务类型计算具体指标
        if isinstance(outputs, dict):
            if 'logits' in outputs:
                logits = outputs['logits']
                targets = batch.get('labels')
                if targets is not None:
                    # 计算准确率
                    _, predicted = torch.max(logits, dim=-1)
                    correct = (predicted == targets).float()
                    metrics['accuracy'] = correct.mean().item()
                    
                    # 计算F1分数（如果是二分类或多分类）
                    if logits.shape[-1] == 2 or len(logits.shape) == 2:
                        metrics['f1'] = self._compute_f1(logits, targets)
        else:
            targets = batch.get('labels')
            if targets is not None:
                _, predicted = torch.max(outputs, dim=-1)
                correct = (predicted == targets).float()
                metrics['accuracy'] = correct.mean().item()
                metrics['f1'] = self._compute_f1(outputs, targets)
                
        return metrics
        
    def _compute_f1(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """计算F1分数"""
        if len(logits.shape) == 2:  # 多分类
            predicted = torch.argmax(logits, dim=-1)
        else:  # 二分类
            predicted = (torch.sigmoid(logits) > 0.5).long().squeeze()
            
        # 计算TP, FP, FN
        tp = ((predicted == 1) & (targets == 1)).sum().float()
        fp = ((predicted == 1) & (targets == 0)).sum().float()
        fn = ((predicted == 0) & (targets == 1)).sum().float()
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
            
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
            
        return f1.item()
        
    def _validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        self.is_training = False
        
        val_metrics = defaultdict(float)
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(batch)
                        loss = self._compute_loss(outputs, batch)
                else:
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)
                    
                # 计算指标
                batch_metrics = self._compute_metrics(outputs, batch, loss.item())
                
                # 累积指标
                for metric_name, value in batch_metrics.items():
                    val_metrics[metric_name] += value
                    
        # 计算平均指标
        for metric_name in val_metrics:
            val_metrics[metric_name] /= num_batches
            
        self.model.train()
        self.is_training = True
        
        # 记录验证日志
        if self.writer:
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f'val_{metric_name}', value, self.epoch)
                
        if self.use_wandb and self.wandb:
            for metric_name, value in val_metrics.items():
                self.wandb.log({f'val_{metric_name}': value}, step=self.global_step)
                
        return dict(val_metrics)
        
    def _save_checkpoint(self, epoch_metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint_path = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            metrics=epoch_metrics,
            is_best=is_best,
            extra_state={
                'best_metric': self.best_metric,
                'scheduler_state': self.scheduler.get_last_lr() if self.scheduler else 0.0
            }
        )
        
        if is_best:
            logger.info(f"新的最佳模型已保存到: {checkpoint_path}")
            
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        try:
            checkpoint_info = self.checkpoint_manager.load(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                path=checkpoint_path
            )
            
            self.epoch = checkpoint_info['epoch']
            self.global_step = checkpoint_info['step']
            self.best_metric = checkpoint_info['metrics'].get('loss', float('inf'))
            
            logger.info(f"从检查点恢复训练: epoch {self.epoch}, step {self.global_step}")
            
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            
    def _log_batch_results(self, batch_idx: int, num_batches: int, batch_metrics: Dict[str, float]):
        """输出批次训练结果"""
        memory_info = ""
        if self.device.type == 'cuda':
            memory_stats = self.memory_monitor.get_memory_stats()
            memory_info = f" | GPU内存: {memory_stats['allocated']:.2f}GB"
            
        lr_info = f" | LR: {self.scheduler.get_last_lr() if self.scheduler else 'N/A'}" if self.scheduler else ""
        
        log_str = f"Epoch {self.epoch+1} [{batch_idx+1}/{num_batches}] | "
        log_str += f"Loss: {batch_metrics['loss']:.4f} | "
        if 'accuracy' in batch_metrics:
            log_str += f"Acc: {batch_metrics['accuracy']:.4f} | "
        if 'f1' in batch_metrics:
            log_str += f"F1: {batch_metrics['f1']:.4f}"
        log_str += f"{lr_info}{memory_info}"
        
        logger.info(log_str)
        
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], 
                          val_metrics: Optional[Dict[str, float]], epoch_time: float):
        """输出epoch训练结果"""
        log_str = f"Epoch {epoch+1} 完成 | 耗时: {epoch_time:.2f}s | "
        
        # 训练指标
        log_str += "Train: "
        for metric_name, value in train_metrics.items():
            log_str += f"{metric_name}: {value:.4f} "
            
        # 验证指标
        if val_metrics:
            log_str += "| Val: "
            for metric_name, value in val_metrics.items():
                log_str += f"{metric_name}: {value:.4f} "
                
        logger.info(log_str)
        
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """评估模型"""
        if dataloader is None:
            dataloader = self.val_dataloader
            
        if dataloader is None:
            raise ValueError("没有提供评估数据加载器")
            
        logger.info("开始评估模型")
        self.model.eval()
        
        eval_metrics = defaultdict(float)
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(batch)
                loss = self._compute_loss(outputs, batch)
                
                # 计算指标
                batch_metrics = self._compute_metrics(outputs, batch, loss.item())
                
                # 累积指标
                for metric_name, value in batch_metrics.items():
                    eval_metrics[metric_name] += value
                    
        # 计算平均指标
        for metric_name in eval_metrics:
            eval_metrics[metric_name] /= num_batches
            
        logger.info("评估完成")
        for metric_name, value in eval_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        return dict(eval_metrics)
        
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """预测"""
        self.model.eval()
        
        with torch.no_grad():
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(batch)
            
            # 处理输出
            if isinstance(outputs, dict):
                # 如果输出是字典，提取logits和可能的概率
                predictions = {}
                if 'logits' in outputs:
                    logits = outputs['logits']
                    if len(logits.shape) > 1:
                        probabilities = torch.softmax(logits, dim=-1)
                        predictions['logits'] = logits.cpu().numpy()
                        predictions['probabilities'] = probabilities.cpu().numpy()
                        predictions['predicted'] = torch.argmax(logits, dim=-1).cpu().numpy()
                    else:
                        predictions['logits'] = logits.cpu().numpy()
                        predictions['probabilities'] = torch.sigmoid(logits).cpu().numpy()
                        predictions['predicted'] = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
                        
                # 添加其他输出
                for key, value in outputs.items():
                    if key != 'logits':
                        predictions[key] = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
            else:
                # 简单输出
                if len(outputs.shape) > 1:
                    probabilities = torch.softmax(outputs, dim=-1)
                    predictions = {
                        'logits': outputs.cpu().numpy(),
                        'probabilities': probabilities.cpu().numpy(),
                        'predicted': torch.argmax(outputs, dim=-1).cpu().numpy()
                    }
                else:
                    probabilities = torch.sigmoid(outputs)
                    predictions = {
                        'logits': outputs.cpu().numpy(),
                        'probabilities': probabilities.cpu().numpy(),
                        'predicted': (probabilities > 0.5).long().cpu().numpy()
                    }
                    
        return predictions
        
    def _cleanup(self):
        """清理资源"""
        if self.writer:
            self.writer.close()
            
        if self.distributed and dist.is_initialized():
            destroy_process_group()
            
        if self.device.type == 'cuda':
            self.memory_monitor.clear_cache()
            
        logger.info("训练器清理完成")
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'distributed': self.distributed,
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps
        }
        
        if self.device.type == 'cuda':
            gpu_count = torch.cuda.device_count()
            gpu_memory = [torch.cuda.get_device_properties(i).total_memory 
                         for i in range(gpu_count)]
            model_info['gpu_count'] = gpu_count
            model_info['gpu_memory'] = [mem / 1024**3 for mem in gpu_memory]  # GB
            
        return model_info


def create_trainer(model: nn.Module, 
                  train_dataloader: DataLoader,
                  val_dataloader: Optional[DataLoader] = None,
                  config: Optional[Dict] = None) -> DocRedTrainer:
    """
    创建训练器的工厂函数
    
    Args:
        model: 模型
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        config: 配置字典
        
    Returns:
        训练器实例
    """
    if config is None:
        config = {}
        
    # 默认配置
    default_config = {
        'optimizer': None,
        'criterion': None,
        'scheduler_config': {
            'scheduler_type': 'step',
            'step_size': 30,
            'gamma': 0.1
        },
        'early_stopping_config': {
            'patience': 7,
            'min_delta': 0.001
        },
        'device': None,
        'distributed': False,
        'mixed_precision': True,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'log_interval': 10,
        'eval_interval': 1000,
        'save_dir': 'checkpoints',
        'use_tensorboard': True,
        'use_wandb': False
    }
    
    # 合并配置
    trainer_config = {**default_config, **config}
    
    # 创建训练器
    trainer = DocRedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **trainer_config
    )
    
    return trainer


if __name__ == "__main__":
    # 示例用法
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=64, num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # 创建示例数据
    batch_size = 32
    input_size = 10
    num_classes = 2
    num_batches = 100
    
    # 生成随机数据
    train_data = torch.randn(batch_size * num_batches, input_size)
    train_labels = torch.randint(0, num_classes, (batch_size * num_batches,))
    val_data = torch.randn(batch_size * 10, input_size)
    val_labels = torch.randint(0, num_classes, (batch_size * 10,))
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = SimpleModel(input_size=input_size, num_classes=num_classes)
    
    # 创建训练器
    trainer = create_trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config={
            'save_dir': './experiments/checkpoints',
            'mixed_precision': True,
            'gradient_accumulation_steps': 1,
            'log_interval': 5,
            'eval_interval': 20
        }
    )
    
    # 获取模型信息
    model_info = trainer.get_model_info()
    print("模型信息:", model_info)
    
    # 开始训练
    print("开始训练...")
    results = trainer.train(num_epochs=3, resume_from=None)
    print("训练结果:", results)
    
    # 评估模型
    eval_results = trainer.evaluate()
    print("评估结果:", eval_results)
    
    # 预测示例
    sample_batch = {'input': val_data[:5]}
    predictions = trainer.predict(sample_batch)
    print("预测结果:", predictions)