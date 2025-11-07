"""
知识融合与推理模块
提供完整的关系抽取知识融合和推理验证功能

主要组件:
- KnowledgeGraph: 知识图谱加载和查询
- LMulator: 基于伪代码的双轨推理引擎
- KnowledgeFusionEngine: 综合知识融合策略
- ReasoningValidator: 推理结果验证和一致性检查

作者: MiniMax Agent
日期: 2025-11-06
版本: 1.0.0
"""

import logging
from typing import Optional

from .knowledge_graph import (
    KnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    create_knowledge_graph
)

from .lmulator import (
    LMulator,
    PseudocodeBlock,
    NaturalLanguageComment,
    ReasoningStep,
    FusionResult,
    ReasoningType,
    ReasoningConfidence,
    PseudocodeAnalyzer,
    NaturalLanguageGenerator,
    create_lmulator
)

from .fusion_engine import (
    KnowledgeFusionEngine,
    EntityFusionEngine,
    RelationFusionEngine,
    EntityRecord,
    RelationRecord,
    FusionDecision,
    FusionStrategy,
    ConflictResolution,
    create_fusion_engine
)

from .reasoning_validator import (
    ReasoningValidator,
    ValidationRule,
    ValidationIssue,
    ValidationResult,
    ValidationLevel,
    IssueSeverity,
    ConsistencyType,
    LogicalConsistencyChecker,
    FactualConsistencyChecker,
    ReasoningQualityEvaluator,
    create_reasoning_validator
)

# 模块版本信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__description__ = "知识融合与推理模块"

# 导出的公共接口
__all__ = [
    # 知识图谱相关
    "KnowledgeGraph",
    "KnowledgeNode", 
    "KnowledgeEdge",
    "create_knowledge_graph",
    
    # LMulator推理引擎相关
    "LMulator",
    "PseudocodeBlock",
    "NaturalLanguageComment",
    "ReasoningStep",
    "FusionResult",
    "ReasoningType",
    "ReasoningConfidence",
    "PseudocodeAnalyzer",
    "NaturalLanguageGenerator",
    "create_lmulator",
    
    # 知识融合引擎相关
    "KnowledgeFusionEngine",
    "EntityFusionEngine",
    "RelationFusionEngine",
    "EntityRecord",
    "RelationRecord",
    "FusionDecision",
    "FusionStrategy",
    "ConflictResolution",
    "create_fusion_engine",
    
    # 推理验证器相关
    "ReasoningValidator",
    "ValidationRule",
    "ValidationIssue",
    "ValidationResult",
    "ValidationLevel",
    "IssueSeverity",
    "ConsistencyType",
    "LogicalConsistencyChecker",
    "FactualConsistencyChecker",
    "ReasoningQualityEvaluator",
    "create_reasoning_validator"
]

# 模块级配置
DEFAULT_CONFIG = {
    'knowledge_graph': {
        'confidence_threshold': 0.7,
        'max_cache_size': 1000,
        'enable_temporal_reasoning': True
    },
    'lmulator': {
        'max_reasoning_steps': 10,
        'confidence_threshold': 0.7,
        'enable_temporal_reasoning': True,
        'enable_analogical_reasoning': True
    },
    'fusion_engine': {
        'similarity_threshold': 0.8,
        'confidence_threshold': 0.7,
        'merge_threshold': 0.6,
        'fusion_strategy': 'weighted_fusion',
        'conflict_resolution': 'majority_vote'
    },
    'reasoning_validator': {
        'validation_level': 'comprehensive',
        'max_issues': 100,
        'enable_quality_evaluation': True
    }
}

def get_default_config() -> dict:
    """获取模块默认配置"""
    return DEFAULT_CONFIG.copy()

def create_knowledge_fusion_system(config: Optional[dict] = None) -> dict:
    """
    创建完整的知识融合与推理系统
    
    Args:
        config: 可选的配置字典，如果为None则使用默认配置
        
    Returns:
        包含所有组件实例的字典
    """
    import logging
    from typing import Optional
    
    # 合并配置
    merged_config = get_default_config()
    if config:
        for key, value in config.items():
            if key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
    
    logger = logging.getLogger(__name__)
    logger.info("创建知识融合与推理系统")
    
    # 创建各个组件
    knowledge_graph = create_knowledge_graph(merged_config.get('knowledge_graph', {}))
    lmulator = create_lmulator(merged_config.get('lmulator', {}))
    fusion_engine = create_fusion_engine(merged_config.get('fusion_engine', {}))
    reasoning_validator = create_reasoning_validator(merged_config.get('reasoning_validator', {}))
    
    system = {
        'knowledge_graph': knowledge_graph,
        'lmulator': lmulator,
        'fusion_engine': fusion_engine,
        'reasoning_validator': reasoning_validator,
        'config': merged_config,
        'version': __version__
    }
    
    logger.info("知识融合与推理系统创建完成")
    return system

def process_knowledge_fusion_pipeline(input_data: dict, system: dict) -> dict:
    """
    执行完整的知识融合与推理流程
    
    Args:
        input_data: 输入数据，包含entities和relations
        system: 知识融合系统实例
        
    Returns:
        处理结果字典
    """
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("开始执行知识融合与推理流程")
    
    try:
        # 第一步：双轨推理
        logger.info("步骤1: 执行双轨推理")
        fusion_result = system['lmulator'].dual_track_reasoning(input_data)
        
        # 第二步：知识融合
        logger.info("步骤2: 执行知识融合")
        knowledge_graph = system['knowledge_graph']
        full_fusion_result = system['fusion_engine'].perform_fusion(
            {
                'entities': fusion_result.entity_fusion,
                'relations': fusion_result.relation_fusion
            }, 
            knowledge_graph
        )
        
        # 第三步：推理验证
        logger.info("步骤3: 执行推理验证")
        validation_result = system['reasoning_validator'].validate_reasoning_result(
            input_data, 
            full_fusion_result, 
            knowledge_graph
        )
        
        # 整合结果
        pipeline_result = {
            'success': True,
            'fusion_result': fusion_result,
            'full_fusion_result': full_fusion_result,
            'validation_result': validation_result,
            'pipeline_metrics': {
                'processing_time': None,  # 可以在实际实现中添加时间统计
                'entities_processed': len(input_data.get('entities', [])),
                'relations_processed': len(input_data.get('relations', [])),
                'validation_score': validation_result.overall_score,
                'is_valid': validation_result.is_valid
            },
            'system_info': {
                'version': system['version'],
                'config': system['config']
            }
        }
        
        logger.info("知识融合与推理流程完成")
        return pipeline_result
        
    except Exception as e:
        logger.error(f"知识融合与推理流程失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'fusion_result': None,
            'full_fusion_result': None,
            'validation_result': None
        }

# 便捷函数
def quick_knowledge_fusion(input_data: dict, config: Optional[dict] = None) -> dict:
    """
    快速执行知识融合流程
    
    Args:
        input_data: 输入数据
        config: 可选的配置
        
    Returns:
        处理结果
    """
    system = create_knowledge_fusion_system(config)
    return process_knowledge_fusion_pipeline(input_data, system)

def validate_input_data(data: dict) -> tuple[bool, list]:
    """
    验证输入数据的格式和内容
    
    Args:
        data: 要验证的数据
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查基本结构
    if not isinstance(data, dict):
        errors.append("输入数据必须是字典类型")
        return False, errors
    
    if 'entities' not in data:
        errors.append("输入数据必须包含'entities'字段")
    
    if 'relations' not in data:
        errors.append("输入数据必须包含'relations'字段")
    
    # 验证实体格式
    entities = data.get('entities', [])
    if not isinstance(entities, list):
        errors.append("'entities'字段必须是列表类型")
    else:
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                errors.append(f"实体{i}必须是字典类型")
                continue
            
            if 'name' not in entity:
                errors.append(f"实体{i}必须包含'name'字段")
            
            if 'type' not in entity:
                errors.append(f"实体{i}必须包含'type'字段")
    
    # 验证关系格式
    relations = data.get('relations', [])
    if not isinstance(relations, list):
        errors.append("'relations'字段必须是列表类型")
    else:
        for i, relation in enumerate(relations):
            if not isinstance(relation, dict):
                errors.append(f"关系{i}必须是字典类型")
                continue
            
            if 'source_name' not in relation and 'source_id' not in relation:
                errors.append(f"关系{i}必须包含'source_name'或'source_id'字段")
            
            if 'target_name' not in relation and 'target_id' not in relation:
                errors.append(f"关系{i}必须包含'target_name'或'target_id'字段")
    
    is_valid = len(errors) == 0
    return is_valid, errors

# 模块初始化日志
logger = logging.getLogger(__name__)
logger.info(f"知识融合与推理模块加载完成 (v{__version__})")