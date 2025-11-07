#!/usr/bin/env python3
"""
数据处理模块
处理DocRED数据集的加载、预处理和格式化
"""

from .data_loader import (
    DocRedDataLoader,
    Document,
    Entity,
    Relation,
    TrainingSample,
    RelationExtractionDataset,
    create_data_loaders
)

from .preprocessor import (
    DocRedPreprocessor,
    DataPreprocessorConfig
)

from .manager import (
    DataManager
)

__all__ = [
    'DocRedDataLoader',
    'Document', 
    'Entity',
    'Relation',
    'TrainingSample',
    'RelationExtractionDataset',
    'create_data_loaders',
    'DocRedPreprocessor',
    'DataPreprocessorConfig',
    'DataManager'
]