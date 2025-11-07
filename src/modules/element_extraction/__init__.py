"""
要素抽取模块

该模块负责从伪代码中提取结构化要素，包括：
1. AST解析 - 解析伪代码语法结构
2. 要素提取 - 提取实体、关系、事件等要素
3. 要素分类 - 对提取的要素进行分类和验证
4. 质量验证 - 验证要素质量和一致性
5. 格式转换 - 转换为DocRED数据集格式

主要类：
- ElementExtractor: 要素提取器主类
- ElementClassifier: 要素分类器
- ElementValidator: 要素验证器
- DocREDFormatter: 格式转换器
"""

from .ast_parser import PseudoCodeASTParser, ASTNode, NodeType
from .extractor import (
    ElementExtractor, 
    ExtractedElement, 
    ExtractedEntity, 
    ExtractedRelation, 
    ElementType,
    EntityCategory,
    RelationCategory
)
from .classifier import (
    ElementClassifier,
    ValidationResult,
    ClassificationResult,
    QualityLevel
)
from .validator import (
    ElementValidator,
    ValidationIssue,
    ValidationReport,
    ValidationError
)
from .formatter import (
    DocREDFormatter,
    DocREDElement
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"

__all__ = [
    # AST解析器
    'PseudoCodeASTParser',
    'ASTNode',
    'NodeType',
    
    # 要素提取器
    'ElementExtractor',
    'ExtractedElement',
    'ExtractedEntity', 
    'ExtractedRelation',
    'ElementType',
    'EntityCategory',
    'RelationCategory',
    
    # 要素分类器
    'ElementClassifier',
    'ValidationResult',
    'ClassificationResult',
    'QualityLevel',
    
    # 要素验证器
    'ElementValidator',
    'ValidationIssue',
    'ValidationReport',
    'ValidationError',
    
    # 格式转换器
    'DocREDFormatter',
    'DocREDElement'
]