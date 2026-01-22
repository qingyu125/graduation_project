# 要素抽取模块

## 面向高效推理的要素抽取与应用算法设计与实现

本模块实现论文核心功能：**要素抽取**，从两个维度对推理过程进行要素提取和分析。

## 模块概述

### 核心功能

1. **局部置信度要素抽取** - 从DeepConf推理过程中提取置信度相关信息
2. **路径压缩要素抽取** - 从RPC压缩过程中提取压缩效果信息
3. **综合分析与可视化** - 对抽取的要素进行深度分析和可视化

### 文件结构

```
element_extraction/
├── __init__.py              # 模块初始化
├── element_extractor.py     # 核心要素抽取实现
├── element_analyzer.py      # 要素深度分析
├── demo_element_extraction.py  # 演示脚本
└── README.md               # 本文档
```

## 快速开始

### 1. 基本用法

```python
from element_extraction import load_and_extract

# 从文件加载结果并抽取要素
result = load_and_extract(
    result_path="online_outputs/result.pkl",
    question_text="Find the sum of all integer bases...",
    ground_truth="70"
)

print(f"效率评分: {result.overall_efficiency_score}")
print(f"质量评分: {result.reasoning_quality_score}")
```

### 2. 使用抽取管道

```python
from element_extraction import ElementExtractionPipeline

# 创建抽取管道
pipeline = ElementExtractionPipeline(
    confidence_window_size=50,   # 置信度滑动窗口大小
    critical_percentile=0.1,     # 关键置信点百分位
    rpc_P=1024,                  # RPC压缩间隔
    rpc_R=32,                    # RPC保留窗口
    rpc_c=4                      # RPC压缩比
)

# 执行抽取
extraction_result = pipeline.extract_from_result(
    result_dict,
    question_text="问题文本",
    ground_truth="正确答案"
)

# 保存结果
pipeline.save_results("element_extraction_results")

# 生成可视化
pipeline.generate_visualization("element_extraction_results")
```

### 3. 要素分析

```python
from element_extraction import ElementAnalyzer

# 创建分析器
analyzer = ElementAnalyzer()

# 生成综合报告
reports = analyzer.generate_comprehensive_report([extraction_result])

# 访问不同类型的报告
confidence_report = reports['confidence_pattern']
compression_report = reports['compression_effect']
tradeoff_report = reports['efficiency_quality_tradeoff']

# 打印报告摘要
for name, report in reports.items():
    print(f"\n{name.upper()} 报告摘要:")
    for key, value in report.summary.items():
        print(f"  {key}: {value}")
```

### 4. 命令行使用

```bash
# 单个结果分析
python demo_element_extraction.py --result_file result.pkl

# RPC vs 非RPC对比
python demo_element_extraction.py --compare \
    --result_no_rpc no_rpc_result.pkl \
    --result_rpc rpc_result.pkl

# 批量处理
python demo_element_extraction.py --batch --result_dir ./results/

# 带深度分析
python demo_element_extraction.py --result_file result.pkl --analyze
```

## 核心数据结构

### LocalConfidenceElement（局部置信度要素）

```python
@dataclass
class LocalConfidenceElement:
    trace_id: int                    # Trace ID
    token_position: int              # Token位置
    token_confidence: float          # 单Token置信度
    sliding_window_confidence: float # 滑动窗口置信度
    local_mean_confidence: float     # 局部均值置信度
    relative_position: float         # 相对位置（0-1）
    is_critical_point: bool          # 是否为关键点
    preceding_confidences: List[float]  # 前置置信度
    following_confidences: List[float]  # 后置置信度
```

### CompressionPathElement（路径压缩要素）

```python
@dataclass
class CompressionPathElement:
    compression_id: int              # 压缩事件ID
    layer_id: int                    # 压缩发生的层
    position: int                    # 压缩位置
    P: int                           # 压缩间隔
    R: int                           # 保留窗口
    c: int                           # 压缩比
    original_token_count: int        # 原始Token数
    compressed_token_count: int      # 压缩后Token数
    compression_ratio: float         # 压缩比
    tokens_compressed: int           # 被压缩的Token数
    tokens_kept: int                 # 保留的Token数
```

### ElementExtractionResult（抽取结果）

```python
@dataclass
class ElementExtractionResult:
    question_id: int                 # 问题ID
    question_text: str               # 问题文本
    ground_truth: Optional[str]      # 正确答案
    
    local_confidence_elements: List[LocalConfidenceElement]
    confidence_statistics: Dict[str, Any]
    
    compression_path_elements: List[CompressionPathElement]
    compression_statistics: Dict[str, Any]
    
    overall_efficiency_score: float  # 效率评分
    reasoning_quality_score: float   # 质量评分
    
    rpc_enabled: bool                # 是否启用RPC
```

## 分析报告类型

### 1. 置信度模式分析（confidence_pattern）

分析推理过程中的置信度分布和变化模式，包括：
- 置信度统计（均值、方差、范围）
- 关键置信点分布
- 置信度趋势分析

### 2. 压缩效果分析（compression_effect）

分析RPC压缩的效果和稳定性，包括：
- 压缩比统计
- 各层压缩效果对比
- 压缩效率随位置变化

### 3. 效率-质量权衡分析（efficiency_quality_tradeoff）

分析效率与质量的权衡关系，包括：
- 帕累托最优解分析
- 相关性分析
- 综合权衡评分

## 输出示例

### 控制台输出

```
要素抽取结果摘要:
  局部置信度要素数量: 3564
  路径压缩要素数量: 24
  效率评分: 2.45
  质量评分: 0.72

置信度统计:
  平均置信度: 0.2847
  置信度标准差: 0.1532
  关键点数量: 356

压缩统计:
  压缩次数: 24
  平均压缩比: 4.26x
  压缩Token数: 18942
```

### 生成的文件

```
element_extraction_results/
├── extraction_results_20251202_120000.json    # JSON格式结果
├── extraction_results_20251202_120000.pkl     # Pickle格式结果
├── confidence_distribution_20251202_120000.png # 置信度分布图
├── compression_effect_20251202_120000.png     # 压缩效果图
└── comprehensive_scores_20251202_120000.png   # 综合评分图
```

## 与现有代码的集成

本模块设计为**非侵入式**，不修改现有代码：

### 1. 从现有结果文件导入

```python
# 现有代码生成的结果
# result = deep_llm.generate(...)

# 要素抽取（不修改原代码）
from element_extraction import ElementExtractionPipeline

pipeline = ElementExtractionPipeline()
extraction = pipeline.extract_from_result(result.__dict__)
```

### 2. 在推理流程中集成

```python
# 在推理完成后调用要素抽取
result = deep_llm.generate(prompt)

# 要素抽取（可选，不影响原推理流程）
if enable_element_extraction:
    pipeline = ElementExtractionPipeline()
    extraction = pipeline.extract_from_result(result.__dict__)
    pipeline.save_results()
```

### 3. 批量分析历史结果

```python
# 分析历史推理结果
from element_extraction import demo_batch_results

results = demo_batch_results(
    result_dir="./online_outputs/",
    output_dir="./element_analysis/"
)
```

## 论文对应关系

本模块实现论文《面向高效推理的要素抽取与应用算法设计与实现》中的核心贡献：

| 论文章节 | 模块实现 |
|---------|---------|
| 第三章 局部置信度要素抽取 | `LocalConfidenceExtractor` 类 |
| 第四章 路径压缩要素抽取 | `CompressionPathExtractor` 类 |
| 第五章 综合分析 | `ElementAnalyzer` 类 |
| 第六章 实验与分析 | `ElementExtractionPipeline` 类 |

## 依赖

- numpy
- matplotlib
- seaborn
- 现有DeepConf/RPC代码

## 扩展使用

### 自定义置信度计算

```python
class CustomConfidenceExtractor(LocalConfidenceExtractor):
    def _compute_sliding_window(self, confs, window_size):
        # 自定义滑动窗口计算
        # ...
        return custom_sliding_confs
```

### 自定义分析报告

```python
class CustomAnalyzer(ElementAnalyzer):
    def analyze_custom_metric(self, extraction_results):
        # 自定义分析逻辑
        # ...
        return custom_report
```

