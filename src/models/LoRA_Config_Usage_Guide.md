# LoRA配置管理系统使用指南

## 概述

本项目的LoRA配置管理系统位于 `/workspace/docred_paper/src/models/lora_config.py`，提供了完整的LoRA（Low-Rank Adaptation）配置管理功能。

## 主要功能

### 1. LoRA配置类 (LoraConfig)
- **核心参数**: rank=8, lora_alpha=32, learning_rate=2e-4
- **动态参数调整**: 支持运行时参数修改
- **参数验证**: 自动验证参数合理性
- **内存估算**: 计算LoRA额外内存消耗

### 2. 预设配置模板
- **轻量级** (lightweight): rank=4, alpha=16, lr=3e-4
- **平衡** (balanced): rank=8, alpha=32, lr=2e-4
- **高性能** (high_performance): rank=16, alpha=64, lr=1e-4
- **实体抽取** (entity_extraction): 专用配置
- **关系分类** (relation_classification): 专用配置
- **联合学习** (joint_learning): 专用配置
- **文档问答** (document_qa): 专用配置
- **超轻量** (ultra_light): rank=2, alpha=8
- **质量优先** (quality_focused): rank=32, alpha=128, lr=5e-5

### 3. 配置管理器 (LoraConfigManager)
- **预设配置管理**: 9种不同场景的预设配置
- **动态参数调整**: 实时修改配置参数
- **参数验证**: 确保配置有效性
- **自动调优**: 基于训练历史自动优化参数
- **文件操作**: 支持JSON和YAML格式的保存/加载
- **配置比较**: 多配置性能对比分析

## 快速使用

### 基础用法

```python
from src.models.lora_config import LoraConfig, LoraConfigManager, create_balanced_config

# 方法1: 使用便利函数
config = create_balanced_config()
print(f"Rank: {config.rank}, Alpha: {config.lora_alpha}")

# 方法2: 创建配置管理器
manager = LoraConfigManager()
config = manager.get_preset('balanced')

# 方法3: 手动创建配置
config = LoraConfig(
    rank=8,
    lora_alpha=32,
    learning_rate=2e-4,
    task_type="entity_extraction"
)
```

### 动态参数调整

```python
# 更新参数
config.update_params(rank=16, learning_rate=1e-4)
print(f"调整后: Rank={config.rank}, LR={config.learning_rate}")

# 参数验证
try:
    config.update_params(rank=-1)  # 无效参数
except ValueError as e:
    print(f"参数验证失败: {e}")
```

### 配置保存和加载

```python
from src.models.lora_config import LoraConfigManager

manager = LoraConfigManager('./configs')

# 保存配置
manager.save_config(config, 'my_config', 'json')  # 或 'yaml'

# 加载配置
loaded_config = manager.load_config('my_config')
```

### 自动调优

```python
# 模拟训练历史
metric_history = [
    (1, 0.5), (2, 0.6), (3, 0.65), (4, 0.7), (5, 0.75)
]

# 自动调优
tuned_config = manager.auto_tune(config, metric_history, "adaptive")
print(f"调优后: Rank={tuned_config.rank}, Alpha={tuned_config.lora_alpha}")
```

### 配置比较

```python
configs = [
    manager.get_preset('lightweight'),
    manager.get_preset('balanced'),
    manager.get_preset('high_performance')
]

comparison = manager.compare_configs(configs, model_params=1e9)
print(f"内存范围: {comparison['summary']['min_memory']:.2f} - {comparison['summary']['max_memory']:.2f} MB")
```

## 任务专用配置

### 实体抽取
```python
entity_config = manager.get_preset('entity_extraction')
# 优化参数: rank=12, alpha=48, dropout=0.15
```

### 关系分类
```python
relation_config = manager.get_preset('relation_classification')
# 优化参数: rank=10, alpha=40, batch_size=24
```

### 联合学习
```python
joint_config = manager.get_preset('joint_learning')
# 优化参数: rank=16, alpha=64, more epochs
```

### 文档问答
```python
qa_config = manager.get_preset('document_qa')
# 优化参数: rank=20, alpha=80, lower lr
```

## 配置报告生成

```python
# 生成详细报告
report = manager.export_config_report(
    config, 
    'config_report.md', 
    model_params=1e9
)
print("报告已生成: config_report.md")
```

## 性能监控

```python
# 获取参数重要性
importance = manager.get_parameter_importance(config)
print(f"参数重要性: {importance}")

# 内存估算
memory_mb = config.get_memory_estimate(model_params=1e9)
print(f"预估额外内存: {memory_mb:.2f} MB")
```

## 配置参数说明

### 核心LoRA参数
- **rank**: LoRA秩，控制低秩矩阵维度 (1-128)
- **lora_alpha**: LoRA缩放参数 (8-256)
- **lora_dropout**: dropout率 (0.0-0.5)
- **learning_rate**: 学习率 (1e-6 - 1e-2)

### 模块配置
- **target_modules**: 目标模块列表
- **exclude_modules**: 排除模块列表

### 训练参数
- **max_epochs**: 最大训练轮数
- **batch_size**: 批次大小
- **warmup_steps**: 预热步数
- **gradient_accumulation_steps**: 梯度累积步数

### 高级配置
- **use_rslora**: 是否使用Rank-Stabilized LoRA
- **lora_init_type**: 初始化类型 (standard/gaussian/scaled)

## 最佳实践

1. **选择合适的预设配置**: 根据任务类型选择专用配置
2. **资源约束考虑**: 小模型使用轻量级配置，大模型使用高性能配置
3. **参数调优**: 使用自动调优功能基于训练历史优化参数
4. **配置管理**: 使用配置管理器统一管理多个配置
5. **性能监控**: 定期生成配置报告监控性能特征

## 故障排除

### 常见问题

1. **参数验证失败**
   - 检查参数是否在有效范围内
   - 确认任务类型和模型规模是否有效

2. **内存估算不准确**
   - 实际内存消耗可能因实现细节而异
   - 建议预留额外的内存空间

3. **自动调优效果不佳**
   - 确保有足够的训练历史数据
   - 尝试不同的调优策略

## 更新日志

- **v1.0.0** (2025-11-06): 初始版本
  - 基础LoRA配置管理
  - 9种预设配置模板
  - 自动调优功能
  - 配置报告生成