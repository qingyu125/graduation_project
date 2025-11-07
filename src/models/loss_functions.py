"""
DocRED任务损失函数定义
实现要素分类、逻辑一致性、混合损失及多任务学习支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math


class ElementClassificationLoss(nn.Module):
    """
    要素分类交叉熵损失函数
    用于实体分类和关系分类
    """
    
    def __init__(self, 
                 element_type: str = "entity",
                 label_smoothing: float = 0.0,
                 reduction: str = "mean",
                 ignore_index: int = -100):
        """
        Args:
            element_type: 要素类型 ("entity" 或 "relation")
            label_smoothing: 标签平滑参数
            reduction: 损失聚合方式
            ignore_index: 忽略的标签索引
        """
        super().__init__()
        self.element_type = element_type
        self.ignore_index = ignore_index
        
        if label_smoothing > 0:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction=reduction,
                ignore_index=ignore_index
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                reduction=reduction,
                ignore_index=ignore_index
            )
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                reduction_dim: str = "batch") -> torch.Tensor:
        """
        Args:
            predictions: 模型预测，支持多种形状
                - [batch_size, num_classes]: 普通分类
                - [batch_size, seq_len, num_classes]: 序列标注
                - [batch_size, N, M, num_classes]: 关系分类等2D任务
            targets: 真实标签，与predictions的前n-1个维度匹配
            mask: 有效位置掩码，与targets形状相同
            reduction_dim: 聚合维度选择
        
        Returns:
            分类损失值
        """
        # 根据predictions的维度进行不同的处理
        if predictions.dim() == 2:  # [batch, num_classes]
            # 普通分类任务
            if mask is not None:
                valid_mask = mask > 0
                if valid_mask.sum() == 0:
                    return torch.tensor(0.0, device=predictions.device)
                valid_pred = predictions[valid_mask]
                valid_targets = targets[valid_mask]
                return self.loss_fn(valid_pred, valid_targets)
            else:
                return self.loss_fn(predictions, targets)
        
        elif predictions.dim() == 3:  # [batch, seq_len, num_classes]
            # 序列标注任务
            # 重塑为2D进行计算
            pred_2d = predictions.view(-1, predictions.size(-1))
            targets_2d = targets.view(-1)
            
            if mask is not None:
                mask_2d = mask.view(-1)
                valid_indices = mask_2d > 0
                if valid_indices.sum() == 0:
                    return torch.tensor(0.0, device=predictions.device)
                valid_pred = pred_2d[valid_indices]
                valid_targets = targets_2d[valid_indices]
                return self.loss_fn(valid_pred, valid_targets)
            else:
                return self.loss_fn(pred_2d, targets_2d)
        
        elif predictions.dim() == 4:  # [batch, N, M, num_classes]
            # 2D任务（如关系分类）
            batch_size, n, m, num_classes = predictions.shape
            
            if targets.shape == (batch_size,):  # 单值标签（如关系分类结果）
                # 直接分类：predictions [batch, n, m, num_classes] -> 选择某个位置的预测
                # 这里假设我们对所有位置都预测，选择主要位置进行分类
                if mask is not None and mask.dim() == 2:  # [batch, num_positions]
                    # 选择有效位置
                    valid_mask = mask > 0
                    if valid_mask.sum() == 0:
                        return torch.tensor(0.0, device=predictions.device)
                    # 对所有位置取平均或加权平均
                    pred_pooled = predictions.mean(dim=(1, 2))  # [batch, num_classes]
                    return self.loss_fn(pred_pooled, targets)
                else:
                    # 对所有位置取平均
                    pred_pooled = predictions.mean(dim=(1, 2))  # [batch, num_classes]
                    return self.loss_fn(pred_pooled, targets)
            
            elif targets.dim() == 3:  # [batch, n, m] 位置级标签
                # 对每个位置进行分类，然后求平均
                total_loss = 0.0
                count = 0
                
                for i in range(n):
                    for j in range(m):
                        pred_ij = predictions[:, i, j, :]  # [batch, num_classes]
                        target_ij = targets[:, i, j]  # [batch]
                        
                        if mask is not None:
                            if mask.dim() == 3:  # [batch, n, m]
                                mask_ij = mask[:, i, j]  # [batch]
                            elif mask.dim() == 2 and mask.shape[1] == n * m:  # [batch, n*m]
                                mask_ij = mask[:, i * m + j]  # [batch]
                            else:
                                mask_ij = torch.ones(batch_size, device=predictions.device)
                            
                            if mask_ij.sum() > 0:
                                loss_ij = self.loss_fn(pred_ij[mask_ij > 0], target_ij[mask_ij > 0])
                                total_loss += loss_ij
                                count += 1
                        else:
                            loss_ij = self.loss_fn(pred_ij, target_ij)
                            total_loss += loss_ij
                            count += 1
                
                return total_loss / max(count, 1)
            
            else:
                # 默认处理
                pred_2d = predictions.view(-1, num_classes)
                targets_2d = targets.view(-1)
                
                if mask is not None:
                    mask_2d = mask.view(-1)
                    valid_indices = mask_2d > 0
                    if valid_indices.sum() == 0:
                        return torch.tensor(0.0, device=predictions.device)
                    valid_pred = pred_2d[valid_indices]
                    valid_targets = targets_2d[valid_indices]
                    return self.loss_fn(valid_pred, valid_targets)
                else:
                    return self.loss_fn(pred_2d, targets_2d)
        else:
            # 默认处理：重塑为2D
            pred_2d = predictions.view(-1, predictions.size(-1))
            targets_2d = targets.view(-1)
            
            if mask is not None:
                mask_2d = mask.view(-1)
                valid_indices = mask_2d > 0
                if valid_indices.sum() == 0:
                    return torch.tensor(0.0, device=predictions.device)
                valid_pred = pred_2d[valid_indices]
                valid_targets = targets_2d[valid_indices]
                return self.loss_fn(valid_pred, valid_targets)
            else:
                return self.loss_fn(pred_2d, targets_2d)


class LogicalConsistencyLoss(nn.Module):
    """
    逻辑一致性正则化项
    确保关系预测的逻辑一致性，如：
    - 如果A是B的子类型，则存在传递性关系
    - 如果A是B的父亲，则B不应该是A的父亲
    """
    
    def __init__(self, 
                 consistency_type: str = "antisymmetry",
                 temperature: float = 1.0,
                 weight: float = 1.0):
        """
        Args:
            consistency_type: 一致性类型
                - "antisymmetry": 反对称性 (如果r(x,y)则¬r(y,x))
                - "transitivity": 传递性 (如果r(x,y)和r(y,z)则r(x,z))
                - "hierarchy": 层次性约束
            temperature: 温度参数，用于sigmoid激活
            weight: 正则化权重
        """
        super().__init__()
        self.consistency_type = consistency_type
        self.temperature = temperature
        self.weight = weight
        
    def forward(self, 
                relation_logits: torch.Tensor,
                entity_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            relation_logits: 关系预测 logits [batch_size, num_entities, num_entities, num_relations]
            entity_masks: 实体有效掩码 [batch_size, num_entities]
        
        Returns:
            逻辑一致性损失
        """
        batch_size, num_entities, _, num_relations = relation_logits.shape
        
        # 计算关系概率
        relation_probs = torch.sigmoid(relation_logits / self.temperature)
        
        consistency_loss = 0.0
        count = 0
        
        for b in range(batch_size):
            # 获取当前batch的关系矩阵
            rel_matrix = relation_probs[b]  # [num_entities, num_entities, num_relations]
            
            if entity_masks is not None:
                # 应用实体掩码
                entity_mask = entity_masks[b] > 0  # [num_entities]
                
                # 确保entity_mask与rel_matrix的维度匹配
                # rel_matrix形状: [num_entities, num_entities, num_relations]
                # entity_mask形状: [num_entities]
                
                # 扩展维度进行广播
                entity_mask_row = entity_mask.unsqueeze(-1).unsqueeze(-1)  # [num_entities, 1, 1]
                entity_mask_col = entity_mask.unsqueeze(-2).unsqueeze(-1)  # [1, num_entities, 1]
                
                # 应用掩码：只保留两个实体都有效的位置
                rel_matrix = rel_matrix * entity_mask_row * entity_mask_col
            
            if self.consistency_type == "antisymmetry":
                # 反对称性：r(i,j) = 1 意味着 r(j,i) = 0
                loss = self._antisymmetry_loss(rel_matrix)
            elif self.consistency_type == "transitivity":
                # 传递性：(A→B & B→C) → A→C
                loss = self._transitivity_loss(rel_matrix)
            elif self.consistency_type == "hierarchy":
                # 层次性：同一实体的关系概率约束
                loss = self._hierarchy_loss(rel_matrix)
            else:
                raise ValueError(f"Unknown consistency type: {self.consistency_type}")
            
            consistency_loss += loss
            count += 1
        
        if count > 0:
            return self.weight * consistency_loss / count
        else:
            return torch.tensor(0.0, device=relation_logits.device)
    
    def _antisymmetry_loss(self, rel_matrix: torch.Tensor) -> torch.Tensor:
        """反对称性损失"""
        # 对于每个关系类型，计算反对称约束
        num_relations = rel_matrix.size(-1)
        loss = 0.0
        
        for r in range(num_relations):
            # rel[:, :, r] 表示关系r的邻接矩阵
            r_matrix = rel_matrix[:, :, r]
            # 反对称性：r(i,j) * r(j,i) 应该接近0
            antisym_loss = torch.sum(r_matrix * r_matrix.transpose(-2, -1))
            loss += antisym_loss
        
        return loss / num_relations
    
    def _transitivity_loss(self, rel_matrix: torch.Tensor) -> torch.Tensor:
        """传递性损失"""
        num_relations = rel_matrix.size(-1)
        loss = 0.0
        
        for r in range(num_relations):
            r_matrix = rel_matrix[:, :, r]  # [num_entities, num_entities]
            # 传递性：r(i,k) >= r(i,j) * r(j,k) 的最小值
            # 使用乘法近似：r(i,k) - r(i,j) * r(j,k) 应该 >= 0
            
            # 计算传递性违反：r(i,k) - sum_j r(i,j) * r(j,k)
            # 这里我们用简化版本：检查是否存在不满足传递性的三元组
            # 使用广播计算所有(i,k,j)组合
            try:
                # r_matrix: [n, n]
                # 计算 r(i,k) 和 r(i,j) * r(j,k) 的差异
                n = r_matrix.size(0)
                
                # 扩展维度用于广播
                r_i_k = r_matrix.unsqueeze(2).expand(n, n, n)  # [n, n, n]，r_i_k[i,k,j] = r(i,k)
                r_i_j = r_matrix.unsqueeze(1).expand(n, n, n)  # [n, n, n]，r_i_j[i,j,k] = r(i,j)
                r_j_k = r_matrix.unsqueeze(0).expand(n, n, n)  # [n, n, n]，r_j_k[j,k,i] = r(j,k)
                
                # 计算违反传递性的情况
                predicted_transitive = r_i_j * r_j_k  # [n, n, n]，预测的传递关系
                actual_vs_predicted = r_i_k - predicted_transitive  # [n, n, n]
                
                # 使用ReLU获取正值（违反）
                transitivity_violation = F.relu(actual_vs_predicted)
                loss += torch.sum(transitivity_violation)
                
            except Exception as e:
                # 如果矩阵乘法失败，使用简化的损失计算
                # 计算矩阵的幂来检查传递性（简化版本）
                r_squared = torch.matmul(r_matrix, r_matrix)
                transitivity_violation = F.relu(r_squared - r_matrix)
                loss += torch.sum(transitivity_violation)
        
        return loss / num_relations
    
    def _hierarchy_loss(self, rel_matrix: torch.Tensor) -> torch.Tensor:
        """层次性损失"""
        # 约束：实体与自身的关系概率应该很小
        self_relations = torch.diagonal(rel_matrix, dim1=-3, dim2=-2)  # [num_entities, num_relations]
        loss = torch.sum(self_relations)
        return loss


class AdaptiveWeightLoss(nn.Module):
    """
    自适应权重调整机制
    动态调整各损失组件的权重
    """
    
    def __init__(self, 
                 initial_weights: Dict[str, float],
                 adaptation_method: str = "gradient_based",
                 learning_rate: float = 0.01,
                 temperature: float = 1.0):
        """
        Args:
            initial_weights: 初始权重字典
            adaptation_method: 权重调整方法
                - "gradient_based": 基于梯度大小
                - "loss_based": 基于损失值
                - "uncertainty_based": 基于不确定性
            learning_rate: 权重学习率
            temperature: 温度参数
        """
        super().__init__()
        self.adaptation_method = adaptation_method
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # 初始化权重
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_weights.get(name, 1.0)))
            for name in initial_weights
        })
        
    def forward(self, 
                loss_dict: Dict[str, torch.Tensor],
                loss_gradients: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            loss_dict: 各损失组件字典
            loss_gradients: 对应的梯度（如果可用）
        
        Returns:
            调整后的权重字典
        """
        if self.adaptation_method == "gradient_based":
            return self._gradient_based_adaptation(loss_dict, loss_gradients)
        elif self.adaptation_method == "loss_based":
            return self._loss_based_adaptation(loss_dict)
        elif self.adaptation_method == "uncertainty_based":
            return self._uncertainty_based_adaptation(loss_dict)
        else:
            return {name: torch.sigmoid(param) for name, param in self.weights.items()}
    
    def _gradient_based_adaptation(self, 
                                  loss_dict: Dict[str, torch.Tensor],
                                  loss_gradients: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """基于梯度的权重调整"""
        if loss_gradients is None:
            # 如果没有提供梯度，尝试计算
            loss_gradients = {}
            for name, loss in loss_dict.items():
                if loss.requires_grad:
                    gradients = torch.autograd.grad(
                        loss, 
                        list(self.weights.parameters())[0], 
                        retain_graph=True, 
                        create_graph=True
                    )
                    if gradients:
                        loss_gradients[name] = gradients[0].abs()
        
        # 基于梯度大小调整权重
        adapted_weights = {}
        for name, param in self.weights.items():
            if name in loss_gradients:
                # 梯度越小，权重越大（鼓励学习）
                grad_norm = loss_gradients[name].mean()
                weight_update = -self.learning_rate * grad_norm
                new_weight = param + weight_update
            else:
                new_weight = param
            
            # 应用sigmoid约束到正数
            adapted_weights[name] = torch.sigmoid(new_weight / self.temperature)
        
        return adapted_weights
    
    def _loss_based_adaptation(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """基于损失值的权重调整"""
        # 损失越小，权重越大
        total_loss = sum(loss_dict.values())
        
        adapted_weights = {}
        for name, param in self.weights.items():
            if name in loss_dict:
                loss_ratio = loss_dict[name] / (total_loss + 1e-8)
                # 损失比例越小，权重越大
                weight_update = -self.learning_rate * loss_ratio
                new_weight = param + weight_update
            else:
                new_weight = param
            
            adapted_weights[name] = torch.sigmoid(new_weight / self.temperature)
        
        return adapted_weights
    
    def _uncertainty_based_adaptation(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """基于不确定性的权重调整"""
        # 假设损失值的不确定性与其大小相关
        loss_variance = {}
        for name, loss in loss_dict.items():
            # 简单的不确定性估计：损失的平方作为方差
            loss_variance[name] = loss ** 2
        
        # 方差越大，权重越小
        total_variance = sum(loss_variance.values())
        
        adapted_weights = {}
        for name, param in self.weights.items():
            if name in loss_variance:
                uncertainty_factor = 1.0 / (loss_variance[name] + 1e-8)
                weight_update = self.learning_rate * uncertainty_factor
                new_weight = param + weight_update
            else:
                new_weight = param
            
            adapted_weights[name] = torch.sigmoid(new_weight / self.temperature)
        
        return adapted_weights


class GradientClippingAndScaling(nn.Module):
    """
    梯度裁剪和损失缩放
    用于稳定训练过程
    """
    
    def __init__(self, 
                 max_norm: float = 1.0,
                 scale_method: str = "gradient_norm",
                 initial_scale: float = 2**15,
                 min_scale: float = 1.0):
        """
        Args:
            max_norm: 最大梯度范数
            scale_method: 缩放方法
                - "gradient_norm": 基于梯度范数
                - "loss_value": 基于损失值
                - "dynamic": 动态缩放
            initial_scale: 初始缩放因子
            min_scale: 最小缩放因子
        """
        super().__init__()
        self.max_norm = max_norm
        self.scale_method = scale_method
        self.current_scale = initial_scale
        self.min_scale = min_scale
        self.scale_factor = 1.0
        
    def forward(self, 
                loss: torch.Tensor,
                model_parameters: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            loss: 原始损失
            model_parameters: 模型参数（用于梯度裁剪）
        
        Returns:
            缩放后的损失和统计信息
        """
        stats = {"original_loss": loss.item(), "scale_factor": 1.0}
        
        if self.scale_method == "gradient_norm":
            # 基于梯度范数的动态缩放
            if model_parameters is not None and loss.requires_grad:
                # 计算梯度
                gradients = torch.autograd.grad(
                    loss, 
                    model_parameters, 
                    retain_graph=True, 
                    create_graph=True
                )
                
                if gradients:
                    total_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]))
                    
                    # 根据梯度范数调整缩放因子
                    if total_norm > self.max_norm:
                        self.scale_factor = self.max_norm / (total_norm + 1e-8)
                    else:
                        self.scale_factor = 1.0
                    
                    stats["gradient_norm"] = total_norm.item()
                    stats["clip_factor"] = self.scale_factor
        
        elif self.scale_method == "loss_value":
            # 基于损失值的缩放
            loss_value = abs(loss.item())
            if loss_value > 1.0:
                self.scale_factor = 1.0 / loss_value
            else:
                self.scale_factor = 1.0
            stats["loss_value"] = loss_value
        
        elif self.scale_method == "dynamic":
            # 动态混合缩放
            if model_parameters is not None and loss.requires_grad:
                gradients = torch.autograd.grad(
                    loss, 
                    model_parameters, 
                    retain_graph=True, 
                    create_graph=True
                )
                
                if gradients:
                    total_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]))
                    
                    # 结合梯度范数和损失值
                    grad_factor = min(self.max_norm / (total_norm + 1e-8), 1.0)
                    loss_factor = 1.0 / (abs(loss.item()) + 1e-8)
                    
                    # 综合因子
                    self.scale_factor = grad_factor * min(loss_factor, 2.0)
                    
                    stats["gradient_norm"] = total_norm.item()
                    stats["grad_factor"] = grad_factor
                    stats["loss_factor"] = loss_factor
        
        # 应用缩放
        scaled_loss = loss * self.scale_factor
        
        # 更新缩放因子（自适应调整）
        if self.scale_factor < 0.5:
            self.current_scale = min(self.current_scale * 2, 2**20)
        elif self.scale_factor > 2.0:
            self.current_scale = max(self.current_scale / 2, self.min_scale)
        
        stats["scaled_loss"] = scaled_loss.item()
        stats["current_scale"] = self.current_scale
        
        return scaled_loss, stats


class MultiTaskLoss(nn.Module):
    """
    支持多任务学习的损失计算
    整合实体识别、关系抽取等多个子任务
    """
    
    def __init__(self, 
                 task_weights: Dict[str, float],
                 task_loss_functions: Dict[str, nn.Module],
                 gradient_accumulation_steps: int = 1,
                 uncertainty_weighting: bool = True):
        """
        Args:
            task_weights: 任务权重字典
            task_loss_functions: 任务损失函数字典
            gradient_accumulation_steps: 梯度累积步数
            uncertainty_weighting: 是否使用不确定性加权
        """
        super().__init__()
        self.task_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight))
            for name, weight in task_weights.items()
        })
        
        self.task_losses = nn.ModuleDict(task_loss_functions)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.uncertainty_weighting = uncertainty_weighting
        self.task_uncertainties = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(1.0)) for name in task_weights
        })
        
        self.step_count = 0
        
    def forward(self, 
                task_outputs: Dict[str, torch.Tensor],
                task_targets: Dict[str, torch.Tensor],
                task_masks: Optional[Dict[str, torch.Tensor]] = None,
                return_per_task: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Args:
            task_outputs: 各任务输出字典
            task_targets: 各任务目标字典
            task_masks: 各任务掩码字典
            return_per_task: 是否返回每个任务的损失
        
        Returns:
            总损失，或(总损失, 各任务损失字典)
        """
        per_task_losses = {}
        per_task_weights = {}
        
        # 计算各任务损失
        for task_name, loss_fn in self.task_losses.items():
            if task_name in task_outputs and task_name in task_targets:
                output = task_outputs[task_name]
                target = task_targets[task_name]
                mask = task_masks.get(task_name) if task_masks else None
                
                task_loss = loss_fn(output, target, mask)
                per_task_losses[task_name] = task_loss
                
                # 获取任务权重
                if self.uncertainty_weighting:
                    # 基于不确定性的权重
                    uncertainty = torch.exp(-self.task_uncertainties[task_name])
                    weight = uncertainty * torch.sigmoid(self.task_weights[task_name])
                    per_task_weights[task_name] = weight
                else:
                    per_task_weights[task_name] = torch.sigmoid(self.task_weights[task_name])
        
        # 计算总损失
        total_loss = 0.0
        for task_name in per_task_losses:
            loss = per_task_losses[task_name]
            weight = per_task_weights[task_name]
            total_loss += weight * loss
        
        # 梯度累积
        if self.training and self.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.gradient_accumulation_steps
        
        self.step_count += 1
        
        if return_per_task:
            return total_loss, per_task_losses
        else:
            return total_loss


class DocREDLoss(nn.Module):
    """
    DocRED任务的混合损失函数
    整合要素分类、逻辑一致性、正则化等组件
    """
    
    def __init__(self,
                 config: Optional[Dict] = None):
        """
        Args:
            config: 损失函数配置
        """
        super().__init__()
        
        # 默认配置
        default_config = {
            "element_classification": {
                "entity_loss_weight": 1.0,
                "relation_loss_weight": 1.0,
                "label_smoothing": 0.0
            },
            "logical_consistency": {
                "weight": 0.1,
                "consistency_types": ["antisymmetry"],
                "temperature": 1.0
            },
            "adaptive_weighting": {
                "enabled": True,
                "method": "gradient_based",
                "learning_rate": 0.01
            },
            "gradient_clipping": {
                "enabled": True,
                "max_norm": 1.0,
                "method": "dynamic"
            },
            "multi_task": {
                "enabled": True,
                "gradient_accumulation": 1,
                "uncertainty_weighting": True
            }
        }
        
        if config is None:
            config = default_config
        
        # 初始化各个组件
        self.element_classifier = ElementClassificationLoss(
            label_smoothing=config["element_classification"].get("label_smoothing", 0.0)
        )
        
        self.logical_consistency = LogicalConsistencyLoss(
            weight=config["logical_consistency"].get("weight", 0.1),
            temperature=config["logical_consistency"].get("temperature", 1.0)
        )
        
        # 初始化任务权重
        task_weights = {
            "entity": config["element_classification"].get("entity_loss_weight", 1.0),
            "relation": config["element_classification"].get("relation_loss_weight", 1.0)
        }
        
        # 初始化任务损失函数
        task_loss_fns = {
            "entity": self.element_classifier,
            "relation": self.element_classifier
        }
        
        self.multi_task_loss = MultiTaskLoss(
            task_weights=task_weights,
            task_loss_functions=task_loss_fns,
            gradient_accumulation_steps=config["multi_task"].get("gradient_accumulation", 1),
            uncertainty_weighting=config["multi_task"].get("uncertainty_weighting", True)
        )
        
        if config["adaptive_weighting"].get("enabled", True):
            self.adaptive_weight = AdaptiveWeightLoss(
                initial_weights=task_weights,
                adaptation_method=config["adaptive_weighting"].get("method", "gradient_based"),
                learning_rate=config["adaptive_weighting"].get("learning_rate", 0.01)
            )
        else:
            self.adaptive_weight = None
        
        if config["gradient_clipping"].get("enabled", True):
            self.gradient_scaling = GradientClippingAndScaling(
                max_norm=config["gradient_clipping"].get("max_norm", 1.0),
                scale_method=config["gradient_clipping"].get("method", "dynamic")
            )
        else:
            self.gradient_scaling = None
        
        self.config = config
        self.training_stats = []
        
    def forward(self, 
                model_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None,
                model_parameters: Optional[List[torch.Tensor]] = None,
                return_stats: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Args:
            model_outputs: 模型输出字典
            targets: 目标字典
            masks: 掩码字典
            model_parameters: 模型参数（用于梯度裁剪）
            return_stats: 是否返回统计信息
        
        Returns:
            总损失和统计信息（如果需要）
        """
        stats = {}
        
        # 1. 要素分类损失
        element_losses = {}
        
        # 实体分类损失
        if "entity_logits" in model_outputs and "entity_labels" in targets:
            entity_loss = self.element_classifier(
                model_outputs["entity_logits"],
                targets["entity_labels"],
                masks.get("entity_mask") if masks else None
            )
            element_losses["entity"] = entity_loss
        
        # 关系分类损失
        if "relation_logits" in model_outputs and "relation_labels" in targets:
            relation_loss = self.element_classifier(
                model_outputs["relation_logits"],
                targets["relation_labels"],
                masks.get("relation_mask") if masks else None
            )
            element_losses["relation"] = relation_loss
        
        stats["element_losses"] = {k: v.item() for k, v in element_losses.items()}
        
        # 2. 逻辑一致性损失
        consistency_loss = 0.0
        if "relation_logits" in model_outputs:
            entity_masks = masks.get("entity_mask") if masks else None
            consistency_loss = self.logical_consistency(
                model_outputs["relation_logits"],
                entity_masks
            )
        
        stats["consistency_loss"] = consistency_loss.item()
        
        # 3. 多任务整合
        task_outputs = {
            k: v for k, v in model_outputs.items() 
            if k in ["entity_logits", "relation_logits"]
        }
        task_targets = {
            k: v for k, v in targets.items()
            if k in ["entity_labels", "relation_labels"]
        }
        
        multi_task_loss, per_task_losses = self.multi_task_loss(
            task_outputs, task_targets, masks, return_per_task=True
        )
        
        # 4. 自适应权重调整
        adaptive_weights = None
        if self.adaptive_weight is not None:
            loss_dict = {**element_losses, "consistency": consistency_loss}
            adaptive_weights = self.adaptive_weight(loss_dict)
            stats["adaptive_weights"] = {k: v.item() for k, v in adaptive_weights.items()}
        
        # 5. 计算总损失
        total_loss = multi_task_loss + consistency_loss
        
        # 应用自适应权重
        if adaptive_weights is not None:
            element_loss_sum = sum(element_losses.values())
            total_loss = (adaptive_weights.get("entity", 1.0) * element_losses.get("entity", 0) +
                         adaptive_weights.get("relation", 1.0) * element_losses.get("relation", 0) +
                         adaptive_weights.get("consistency", 1.0) * consistency_loss)
        
        # 6. 梯度裁剪和缩放
        scale_stats = {}
        if self.gradient_scaling is not None:
            total_loss, scale_stats = self.gradient_scaling(total_loss, model_parameters)
            stats["gradient_scaling"] = scale_stats
        
        # 更新统计信息
        stats["total_loss"] = total_loss.item()
        stats["per_task_losses"] = {k: v.item() for k, v in per_task_losses.items()}
        
        self.training_stats.append(stats)
        if len(self.training_stats) > 1000:  # 保持最近1000步的统计
            self.training_stats = self.training_stats[-1000:]
        
        if return_stats:
            return total_loss, stats
        else:
            return total_loss
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        if not self.training_stats:
            return {}
        
        # 计算最近统计信息的平均值
        recent_stats = self.training_stats[-100:]  # 最近100步
        
        avg_stats = {}
        for key in recent_stats[0].keys():
            if isinstance(recent_stats[0][key], dict):
                avg_stats[key] = {}
                for subkey in recent_stats[0][key].keys():
                    values = [stats[key][subkey] for stats in recent_stats 
                             if key in stats and subkey in stats[key]]
                    if values:
                        avg_stats[key][subkey] = np.mean(values)
            else:
                values = [stats[key] for stats in recent_stats if key in stats]
                if values:
                    avg_stats[key] = np.mean(values)
        
        return avg_stats


def create_docred_loss(config: Optional[Dict] = None) -> DocREDLoss:
    """
    创建DocRED损失函数实例的工厂函数
    
    Args:
        config: 损失函数配置
    
    Returns:
        DocREDLoss实例
    """
    return DocREDLoss(config)


# 示例配置
EXAMPLE_CONFIG = {
    "element_classification": {
        "entity_loss_weight": 1.0,
        "relation_loss_weight": 1.2,  # 关系预测通常更重要
        "label_smoothing": 0.1
    },
    "logical_consistency": {
        "weight": 0.05,  # 较低的一致性权重
        "consistency_types": ["antisymmetry", "transitivity"],
        "temperature": 1.0
    },
    "adaptive_weighting": {
        "enabled": True,
        "method": "gradient_based",
        "learning_rate": 0.001
    },
    "gradient_clipping": {
        "enabled": True,
        "max_norm": 1.0,
        "method": "dynamic"
    },
    "multi_task": {
        "enabled": True,
        "gradient_accumulation": 1,
        "uncertainty_weighting": True
    }
}


if __name__ == "__main__":
    # 简单的测试用例
    print("DocRED损失函数模块测试")
    
    # 创建配置
    config = EXAMPLE_CONFIG
    
    # 创建损失函数
    loss_fn = create_docred_loss(config)
    
    # 模拟模型输出
    batch_size, num_entities, num_relations = 2, 50, 96
    
    model_outputs = {
        "entity_logits": torch.randn(batch_size, num_entities, 2),  # 实体分类（2类）
        "relation_logits": torch.randn(batch_size, num_entities, num_entities, num_relations)
    }
    
    targets = {
        "entity_labels": torch.randint(0, 2, (batch_size, num_entities)),
        "relation_labels": torch.randint(0, num_relations, (batch_size, 10, 10))  # 假设10个有效关系对
    }
    
    masks = {
        "entity_mask": torch.ones(batch_size, num_entities),
        "relation_mask": torch.ones(batch_size, 10, 10)
    }
    
    # 计算损失
    with torch.no_grad():
        total_loss, stats = loss_fn(
            model_outputs, targets, masks, return_stats=True
        )
    
    print(f"总损失: {total_loss.item():.4f}")
    print(f"统计信息: {stats}")
    print("损失函数创建成功！")