"""
要素抽取模块 - 面向高效推理的要素抽取与应用算法设计与实现

本模块提供两个维度的要素抽取功能：
1. 局部置信度要素抽取 - 从DeepConf推理过程中抽取
2. 路径压缩要素抽取 - 从RPC压缩过程中抽取

主要类:
- LocalConfidenceExtractor: 局部置信度要素抽取器
- CompressionPathExtractor: 路径压缩要素抽取器
- ElementExtractionPipeline: 要素抽取流程控制器
- ElementExtractionResult: 要素抽取结果容器

主要数据结构:
- LocalConfidenceElement: 局部置信度要素
- CompressionPathElement: 路径压缩要素
"""

from .element_extractor import (
    LocalConfidenceElement,
    CompressionPathElement,
    ElementExtractionResult,
    LocalConfidenceExtractor,
    CompressionPathExtractor,
    ElementExtractionPipeline,
    load_and_extract
)

__all__ = [
    'LocalConfidenceElement',
    'CompressionPathElement',
    'ElementExtractionResult',
    'LocalConfidenceExtractor',
    'CompressionPathExtractor',
    'ElementExtractionPipeline',
    'load_and_extract'
]

__version__ = '1.0.0'
__author__ = 'MiniMax Agent'
