"""
基于伪代码的推理引擎（LMulator核心）
实现双轨推理输出：伪代码 + 自然语言注释
支持要素验证、关系推理和知识融合
"""

import json
import logging
import ast
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """推理类型枚举"""
    DEDUCTIVE = "deductive"  # 演绎推理
    INDUCTIVE = "inductive"  # 归纳推理
    ABDUCTIVE = "abductive"  # 溯因推理
    TEMPORAL = "temporal"   # 时间推理
    CAUSAL = "causal"       # 因果推理
    ANALOGICAL = "analogical"  # 类比推理

class ReasoningConfidence(Enum):
    """推理置信度等级"""
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.2-0.5
    UNCERTAIN = "uncertain"  # 0.0-0.2

@dataclass
class PseudocodeBlock:
    """伪代码块结构"""
    function_name: str
    parameters: List[str]
    local_variables: List[str]
    control_structures: List[Dict]
    statements: List[str]
    return_value: Optional[str] = None
    
@dataclass
class NaturalLanguageComment:
    """自然语言注释结构"""
    statement_id: str
    comment_text: str
    reasoning_type: ReasoningType
    confidence: ReasoningConfidence
    evidence: List[str]
    explanation: str

@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    input_facts: List[str]
    reasoning_type: ReasoningType
    pseudocode: PseudocodeBlock
    natural_comments: List[NaturalLanguageComment]
    output_facts: List[str]
    confidence: float
    explanation: str

@dataclass
class FusionResult:
    """知识融合结果"""
    entity_fusion: Dict[str, Any]
    relation_fusion: Dict[str, Any]
    reasoning_steps: List[ReasoningStep]
    validation_results: Dict[str, Any]
    confidence_score: float

class PseudocodeAnalyzer:
    """伪代码分析器"""
    
    def __init__(self):
        self.function_patterns = [
            r'def\s+(\w+)\s*\([^)]*\):',
            r'function\s+(\w+)\s*\([^)]*\):',
            r'(\w+)\s*=\s*function\s*\([^)]*\):'
        ]
        
        self.control_patterns = {
            'if': r'if\s+(.+?):',
            'elif': r'elif\s+(.+?):',
            'else': r'else:',
            'for': r'for\s+(.+?)\s+in\s+(.+?):',
            'while': r'while\s+(.+?):',
            'try': r'try:',
            'except': r'except\s+(.+?):'
        }
        
        self.statement_patterns = [
            r'(\w+)\s*=\s*(.+)',
            r'(\w+)\.(\w+)\((.*?)\)',
            r'return\s+(.+)',
            r'print\s*\((.*?)\)',
            r'assert\s+(.+)'
        ]
    
    def parse_pseudocode(self, pseudocode: str) -> PseudocodeBlock:
        """解析伪代码为结构化块"""
        try:
            lines = [line.strip() for line in pseudocode.split('\n') if line.strip()]
            
            function_name = self._extract_function_name(lines)
            parameters = self._extract_parameters(lines)
            local_variables = self._extract_local_variables(lines)
            control_structures = self._extract_control_structures(lines)
            statements = self._extract_statements(lines)
            return_value = self._extract_return_value(lines)
            
            return PseudocodeBlock(
                function_name=function_name,
                parameters=parameters,
                local_variables=local_variables,
                control_structures=control_structures,
                statements=statements,
                return_value=return_value
            )
            
        except Exception as e:
            logger.error(f"伪代码解析失败: {e}")
            raise
    
    def _extract_function_name(self, lines: List[str]) -> str:
        """提取函数名"""
        for line in lines:
            for pattern in self.function_patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1)
        return "unknown_function"
    
    def _extract_parameters(self, lines: List[str]) -> List[str]:
        """提取参数"""
        for line in lines:
            for pattern in self.function_patterns:
                match = re.search(pattern, line)
                if match:
                    params_str = line.split('(')[1].split(')')[0]
                    return [param.strip() for param in params_str.split(',') if param.strip()]
        return []
    
    def _extract_local_variables(self, lines: List[str]) -> List[str]:
        """提取局部变量"""
        variables = set()
        for line in lines:
            # 查找赋值语句
            for pattern in self.statement_patterns:
                match = re.search(pattern, line)
                if match and '=' in line and 'def ' not in line and 'return' not in line:
                    var_name = match.group(1).strip()
                    if var_name and var_name.isidentifier():
                        variables.add(var_name)
        return list(variables)
    
    def _extract_control_structures(self, lines: List[str]) -> List[Dict]:
        """提取控制结构"""
        structures = []
        for i, line in enumerate(lines):
            for control_type, pattern in self.control_patterns.items():
                match = re.search(pattern, line)
                if match:
                    structure = {
                        'type': control_type,
                        'line_number': i + 1,
                        'content': line,
                        'condition': match.group(1) if match.groups() else None,
                        'indentation': len(line) - len(line.lstrip())
                    }
                    structures.append(structure)
        return structures
    
    def _extract_statements(self, lines: List[str]) -> List[str]:
        """提取执行语句"""
        statements = []
        for line in lines:
            if line and not line.startswith('#'):
                # 过滤函数定义和注释
                if not any(re.search(pattern, line) for pattern in self.function_patterns):
                    statements.append(line)
        return statements
    
    def _extract_return_value(self, lines: List[str]) -> Optional[str]:
        """提取返回值"""
        for line in lines:
            if line.strip().startswith('return '):
                return line.strip()[7:].strip()
        return None

class NaturalLanguageGenerator:
    """自然语言注释生成器"""
    
    def __init__(self):
        self.reasoning_templates = {
            ReasoningType.DEDUCTIVE: [
                "根据已知事实 {}，可以推导出 {}",
                "从前提 {} 出发，逻辑上得出结论 {}",
                "基于规则 {}，推导出 {}"
            ],
            ReasoningType.INDUCTIVE: [
                "通过观察 {} 的模式，推断 {}",
                "从多个实例 {} 中总结出一般规律 {}",
                "归纳 {} 的共同特征，形成 {}"
            ],
            ReasoningType.ABDUCTIVE: [
                "为了解释 {}，推测可能的原因为 {}",
                "观察结果 {}，最合理的假设是 {}",
                "基于现象 {}，推断可能的原因 {}"
            ],
            ReasoningType.TEMPORAL: [
                "在时间 {} 之后，发生了 {}",
                "按照时间顺序 {}，接下来应该是 {}",
                "基于时间关系 {}，可以预测 {}"
            ],
            ReasoningType.CAUSAL: [
                "由于 {}，导致了 {}",
                "因果关系显示 {} 是 {} 的原因",
                "基于因果推理 {} 导致了 {}"
            ],
            ReasoningType.ANALOGICAL: [
                "类似于 {} 的情况，推断 {}",
                "通过类比推理 {}，得出 {}",
                "借鉴 {} 的模式，推断 {}"
            ]
        }
    
    def generate_comments(self, pseudocode_block: PseudocodeBlock, 
                         context: Dict[str, Any]) -> List[NaturalLanguageComment]:
        """生成自然语言注释"""
        try:
            comments = []
            
            # 为每个函数参数生成注释
            for param in pseudocode_block.parameters:
                comment = self._generate_parameter_comment(param, context)
                if comment:
                    comments.append(comment)
            
            # 为每个局部变量生成注释
            for var in pseudocode_block.local_variables:
                comment = self._generate_variable_comment(var, context)
                if comment:
                    comments.append(comment)
            
            # 为控制结构生成注释
            for control in pseudocode_block.control_structures:
                comment = self._generate_control_comment(control, context)
                if comment:
                    comments.append(comment)
            
            # 为主要语句生成注释
            for statement in pseudocode_block.statements[:5]:  # 限制注释数量
                comment = self._generate_statement_comment(statement, context)
                if comment:
                    comments.append(comment)
            
            logger.info(f"生成自然语言注释: {len(comments)}条")
            return comments
            
        except Exception as e:
            logger.error(f"生成自然语言注释失败: {e}")
            return []
    
    def _generate_parameter_comment(self, param: str, context: Dict[str, Any]) -> Optional[NaturalLanguageComment]:
        """生成参数注释"""
        param_lower = param.lower()
        reasoning_type = ReasoningType.DEDUCTIVE
        
        if 'entity' in param_lower:
            comment_text = f"参数 {param} 表示需要进行推理的实体对象"
            explanation = "实体参数提供了推理的基础对象，包含实体的基本信息和属性"
        elif 'relation' in param_lower:
            comment_text = f"参数 {param} 定义了实体间的关系类型和约束"
            explanation = "关系参数指定了实体间需要验证或推断的具体关系"
        elif 'context' in param_lower:
            comment_text = f"参数 {param} 包含了推理所需的上下文信息"
            explanation = "上下文参数提供推理的环境信息，如时间、地点等背景"
        else:
            comment_text = f"参数 {param} 包含推理过程需要的具体数据"
            explanation = "该参数为推理算法提供了必要的数据输入"
        
        return NaturalLanguageComment(
            statement_id=f"param_{param}",
            comment_text=comment_text,
            reasoning_type=reasoning_type,
            confidence=ReasoningConfidence.HIGH,
            evidence=[f"函数参数定义: {param}"],
            explanation=explanation
        )
    
    def _generate_variable_comment(self, var: str, context: Dict[str, Any]) -> Optional[NaturalLanguageComment]:
        """生成变量注释"""
        var_lower = var.lower()
        reasoning_type = ReasoningType.DEDUCTIVE
        
        if 'result' in var_lower:
            comment_text = f"变量 {var} 存储推理计算的中间结果"
            explanation = "结果变量用于暂存推理过程中的计算值，为后续步骤提供输入"
        elif 'flag' in var_lower or 'status' in var_lower:
            comment_text = f"变量 {var} 记录推理过程的执行状态"
            explanation = "状态变量跟踪推理是否成功完成，包含验证标志和错误信息"
        elif 'count' in var_lower or 'index' in var_lower:
            comment_text = f"变量 {var} 用于控制循环和索引操作"
            explanation = "计数变量管理循环执行次数，确保推理过程按预期进行"
        else:
            comment_text = f"变量 {var} 暂存推理过程的临时数据"
            explanation = "该变量作为推理算法的辅助存储，提高计算效率"
        
        return NaturalLanguageComment(
            statement_id=f"var_{var}",
            comment_text=comment_text,
            reasoning_type=reasoning_type,
            confidence=ReasoningConfidence.MEDIUM,
            evidence=[f"变量定义: {var}"],
            explanation=explanation
        )
    
    def _generate_control_comment(self, control: Dict, context: Dict[str, Any]) -> Optional[NaturalLanguageComment]:
        """生成控制结构注释"""
        control_type = control.get('type', '')
        condition = control.get('condition', '')
        
        if control_type == 'if':
            comment_text = f"条件判断: 如果 {condition}，则执行推理分支"
            explanation = "条件控制根据预设条件决定是否执行特定的推理逻辑"
        elif control_type == 'for':
            comment_text = f"循环结构: 对 {condition} 中的每个元素执行推理操作"
            explanation = "循环控制对数据集合中的每个元素应用相同的推理规则"
        elif control_type == 'while':
            comment_text = f"循环控制: 当 {condition} 满足时，持续执行推理"
            explanation = "循环控制确保推理过程持续进行，直到满足终止条件"
        elif control_type == 'try':
            comment_text = "异常处理: 尝试执行推理操作"
            explanation = "异常处理捕获推理过程中可能出现的错误，确保程序稳定性"
        else:
            comment_text = f"控制结构: {control_type} 控制推理流程"
            explanation = "该控制结构管理推理代码的执行流程和逻辑"
        
        return NaturalLanguageComment(
            statement_id=f"control_{control_type}_{hashlib.md5(condition.encode()).hexdigest()[:8]}",
            comment_text=comment_text,
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=ReasoningConfidence.MEDIUM,
            evidence=[f"控制结构: {control.get('content', '')}"],
            explanation=explanation
        )
    
    def _generate_statement_comment(self, statement: str, context: Dict[str, Any]) -> Optional[NaturalLanguageComment]:
        """生成语句注释"""
        # 简化版语句注释生成
        if 'assert' in statement:
            comment_text = f"验证断言: {statement}"
            explanation = "断言语句验证推理结果的正确性，确保逻辑一致性"
            confidence = ReasoningConfidence.HIGH
        elif 'return' in statement:
            comment_text = f"返回结果: {statement}"
            explanation = "返回语句输出推理过程的最终结果"
            confidence = ReasoningConfidence.HIGH
        elif '=' in statement:
            comment_text = f"赋值操作: {statement}"
            explanation = "赋值语句计算和存储推理过程的中间值"
            confidence = ReasoningConfidence.MEDIUM
        else:
            comment_text = f"执行语句: {statement}"
            explanation = "该语句执行具体的推理操作"
            confidence = ReasoningConfidence.LOW
        
        return NaturalLanguageComment(
            statement_id=f"stmt_{hashlib.md5(statement.encode()).hexdigest()[:8]}",
            comment_text=comment_text,
            reasoning_type=ReasoningType.DEDUCTIVE,
            confidence=confidence,
            evidence=[f"源码语句: {statement}"],
            explanation=explanation
        )

class LMulator:
    """基于伪代码的推理引擎（LMulator核心）"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化LMulator推理引擎"""
        self.config = config or {}
        self.pseudocode_analyzer = PseudocodeAnalyzer()
        self.nl_generator = NaturalLanguageGenerator()
        
        # 配置参数
        self.max_reasoning_steps = self.config.get('max_reasoning_steps', 10)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.enable_temporal_reasoning = self.config.get('enable_temporal_reasoning', True)
        self.enable_analogical_reasoning = self.config.get('enable_analogical_reasoning', True)
        
        # 推理规则库
        self.reasoning_rules = self._initialize_reasoning_rules()
        
        logger.info("LMulator推理引擎初始化完成")
    
    def dual_track_reasoning(self, input_data: Dict[str, Any]) -> FusionResult:
        """双轨推理：伪代码 + 自然语言注释"""
        try:
            logger.info("开始双轨推理过程")
            
            # 第一步：解析输入数据
            entities = input_data.get('entities', [])
            relations = input_data.get('relations', [])
            context = input_data.get('context', {})
            
            # 第二步：生成伪代码推理过程
            reasoning_steps = self._generate_reasoning_steps(entities, relations, context)
            
            # 第三步：生成自然语言注释
            for step in reasoning_steps:
                step.natural_comments = self.nl_generator.generate_comments(
                    step.pseudocode, context)
            
            # 第四步：知识融合
            fusion_result = self._perform_knowledge_fusion(entities, relations, reasoning_steps)
            
            # 第五步：推理验证
            fusion_result.validation_results = self._validate_reasoning(fusion_result)
            
            logger.info("双轨推理完成")
            return fusion_result
            
        except Exception as e:
            logger.error(f"双轨推理失败: {e}")
            return self._create_error_result(str(e))
    
    def _generate_reasoning_steps(self, entities: List[Dict], 
                                relations: List[Dict], 
                                context: Dict[str, Any]) -> List[ReasoningStep]:
        """生成推理步骤序列"""
        reasoning_steps = []
        
        step_id = 0
        
        # 步骤1: 实体预处理
        pseudocode1 = self._generate_entity_preprocessing_pseudocode(entities)
        step1 = ReasoningStep(
            step_id=f"step_{step_id}",
            input_facts=[f"实体: {entity.get('name', '')}" for entity in entities],
            reasoning_type=ReasoningType.DEDUCTIVE,
            pseudocode=pseudocode1,
            natural_comments=[],
            output_facts=[f"预处理实体: {entity.get('name', '')}" for entity in entities],
            confidence=0.9,
            explanation="对输入实体进行标准化和验证"
        )
        reasoning_steps.append(step1)
        step_id += 1
        
        # 步骤2: 关系验证
        if relations:
            pseudocode2 = self._generate_relation_validation_pseudocode(relations)
            step2 = ReasoningStep(
                step_id=f"step_{step_id}",
                input_facts=[f"关系: {relation.get('source_name', '')} -> {relation.get('target_name', '')}" for relation in relations],
                reasoning_type=ReasoningType.DEDUCTIVE,
                pseudocode=pseudocode2,
                natural_comments=[],
                output_facts=[f"验证关系: {relation.get('source_name', '')} -> {relation.get('target_name', '')}" for relation in relations],
                confidence=0.8,
                explanation="验证实体间关系的有效性"
            )
            reasoning_steps.append(step2)
            step_id += 1
        
        # 步骤3: 知识推理（如果启用了高级推理）
        if context.get('enable_advanced_reasoning', False):
            pseudocode3 = self._generate_advanced_reasoning_pseudocode(entities, relations, context)
            step3 = ReasoningStep(
                step_id=f"step_{step_id}",
                input_facts=[f"实体: {entity.get('name', '')}" for entity in entities] + 
                           [f"关系: {relation.get('source_name', '')} -> {relation.get('target_name', '')}" for relation in relations],
                reasoning_type=ReasoningType.CAUSAL,
                pseudocode=pseudocode3,
                natural_comments=[],
                output_facts=[f"推理新事实: {entity.get('name', '')}" for entity in entities],
                confidence=0.7,
                explanation="基于现有事实进行高级推理"
            )
            reasoning_steps.append(step3)
        
        return reasoning_steps
    
    def _generate_entity_preprocessing_pseudocode(self, entities: List[Dict]) -> PseudocodeBlock:
        """生成实体预处理伪代码"""
        pseudocode_lines = [
            "def preprocess_entities(entities):",
            "    # 初始化预处理结果",
            "    processed_entities = []",
            "    validation_errors = []",
            "    ",
            "    for entity in entities:",
            "        # 验证实体信息",
            "        if not validate_entity(entity):",
            "            validation_errors.append(f'Invalid entity: {entity[\"name\"]}')",
            "            continue",
            "        ",
            "        # 标准化实体名称",
            "        standardized_name = standardize_entity_name(entity['name'])",
            "        ",
            "        # 添加到处理结果",
            "        processed_entities.append({",
            "            'name': standardized_name,",
            "            'type': entity['type'],",
            "            'confidence': entity.get('confidence', 0.0),",
            "            'original': entity",
            "        })",
            "    ",
            "    return processed_entities, validation_errors"
        ]
        
        pseudocode = '\n'.join(pseudocode_lines)
        return self.pseudocode_analyzer.parse_pseudocode(pseudocode)
    
    def _generate_relation_validation_pseudocode(self, relations: List[Dict]) -> PseudocodeBlock:
        """生成关系验证伪代码"""
        pseudocode_lines = [
            "def validate_relations(relations, entities):",
            "    # 构建实体名称到ID的映射",
            "    entity_map = {entity['name']: entity for entity in entities}",
            "    ",
            "    validated_relations = []",
            "    validation_results = []",
            "    ",
            "    for relation in relations:",
            "        source_name = relation.get('source_name')",
            "        target_name = relation.get('target_name')",
            "        ",
            "        # 检查实体存在性",
            "        if source_name not in entity_map or target_name not in entity_map:",
            "            validation_results.append(f'Missing entities for relation: {source_name} -> {target_name}')",
            "            continue",
            "        ",
            "        # 验证关系语义",
            "        if validate_relation_semantics(relation):",
            "            validated_relations.append(relation)",
            "            validation_results.append(f'Valid relation: {source_name} -> {target_name}')",
            "        else:",
            "            validation_results.append(f'Invalid relation semantics: {source_name} -> {target_name}')",
            "    ",
            "    return validated_relations, validation_results"
        ]
        
        pseudocode = '\n'.join(pseudocode_lines)
        return self.pseudocode_analyzer.parse_pseudocode(pseudocode)
    
    def _generate_advanced_reasoning_pseudocode(self, entities: List[Dict], 
                                              relations: List[Dict], 
                                              context: Dict[str, Any]) -> PseudocodeBlock:
        """生成高级推理伪代码"""
        reasoning_methods = []
        if self.enable_temporal_reasoning:
            reasoning_methods.append("temporal")
        if self.enable_analogical_reasoning:
            reasoning_methods.append("analogical")
        
        pseudocode_lines = [
            f"def advanced_reasoning(entities, relations, methods={reasoning_methods}):",
            "    inferred_facts = []",
            "    reasoning_log = []",
            "    ",
            "    for method in methods:",
            "        if method == 'temporal':",
            "            # 时间推理",
            "            temporal_facts = temporal_reasoning(entities, relations)",
            "            inferred_facts.extend(temporal_facts)",
            "            reasoning_log.append(f'Temporal reasoning found {len(temporal_facts)} facts')",
            "        ",
            "        elif method == 'analogical':",
            "            # 类比推理",
            "            analogical_facts = analogical_reasoning(entities, relations)",
            "            inferred_facts.extend(analogical_facts)",
            "            reasoning_log.append(f'Analogical reasoning found {len(analogical_facts)} facts')",
            "        ",
            "        elif method == 'causal':",
            "            # 因果推理",
            "            causal_facts = causal_reasoning(entities, relations)",
            "            inferred_facts.extend(causal_facts)",
            "            reasoning_log.append(f'Causal reasoning found {len(causal_facts)} facts')",
            "    ",
            "    return inferred_facts, reasoning_log"
        ]
        
        pseudocode = '\n'.join(pseudocode_lines)
        return self.pseudocode_analyzer.parse_pseudocode(pseudocode)
    
    def _perform_knowledge_fusion(self, entities: List[Dict], 
                                relations: List[Dict], 
                                reasoning_steps: List[ReasoningStep]) -> FusionResult:
        """执行知识融合"""
        try:
            # 实体融合
            entity_fusion = self._fuse_entities(entities, reasoning_steps)
            
            # 关系融合
            relation_fusion = self._fuse_relations(relations, reasoning_steps)
            
            # 计算总体置信度
            confidence_score = self._calculate_overall_confidence(entity_fusion, relation_fusion, reasoning_steps)
            
            fusion_result = FusionResult(
                entity_fusion=entity_fusion,
                relation_fusion=relation_fusion,
                reasoning_steps=reasoning_steps,
                validation_results={},
                confidence_score=confidence_score
            )
            
            logger.info(f"知识融合完成: 置信度={confidence_score:.3f}")
            return fusion_result
            
        except Exception as e:
            logger.error(f"知识融合失败: {e}")
            raise
    
    def _fuse_entities(self, entities: List[Dict], reasoning_steps: List[ReasoningStep]) -> Dict[str, Any]:
        """融合实体信息"""
        fused_entities = {}
        
        for entity in entities:
            entity_name = entity.get('name', '')
            entity_type = entity.get('type', '')
            
            if entity_name not in fused_entities:
                fused_entities[entity_name] = {
                    'name': entity_name,
                    'type': entity_type,
                    'sources': [],
                    'properties': {},
                    'confidence_scores': [],
                    'fused_properties': {}
                }
            
            # 记录来源
            fused_entities[entity_name]['sources'].append('input')
            
            # 合并属性
            for key, value in entity.items():
                if key not in ['name', 'type']:
                    if key not in fused_entities[entity_name]['properties']:
                        fused_entities[entity_name]['properties'][key] = []
                    fused_entities[entity_name]['properties'][key].append(value)
            
            # 记录置信度
            confidence = entity.get('confidence', 0.0)
            fused_entities[entity_name]['confidence_scores'].append(confidence)
        
        # 执行属性融合
        for entity_name, entity_data in fused_entities.items():
            entity_data['fused_properties'] = self._merge_entity_properties(entity_data['properties'])
            entity_data['average_confidence'] = sum(entity_data['confidence_scores']) / len(entity_data['confidence_scores'])
        
        return fused_entities
    
    def _fuse_relations(self, relations: List[Dict], reasoning_steps: List[ReasoningStep]) -> Dict[str, Any]:
        """融合关系信息"""
        fused_relations = {}
        
        for relation in relations:
            source_name = relation.get('source_name', '')
            target_name = relation.get('target_name', '')
            relation_key = f"{source_name} -> {target_name}"
            
            if relation_key not in fused_relations:
                fused_relations[relation_key] = {
                    'source': source_name,
                    'target': target_name,
                    'relation_types': [],
                    'confidence_scores': [],
                    'evidence': [],
                    'fused_confidence': 0.0
                }
            
            # 记录关系类型
            relation_type = relation.get('relation_type', '')
            fused_relations[relation_key]['relation_types'].append(relation_type)
            
            # 记录置信度
            confidence = relation.get('confidence', 0.0)
            fused_relations[relation_key]['confidence_scores'].append(confidence)
            
            # 记录证据
            evidence = relation.get('evidence', [])
            fused_relations[relation_key]['evidence'].extend(evidence)
        
        # 计算融合置信度
        for relation_key, relation_data in fused_relations.items():
            relation_data['fused_confidence'] = sum(relation_data['confidence_scores']) / len(relation_data['confidence_scores'])
            relation_data['unique_relation_types'] = list(set(relation_data['relation_types']))
        
        return fused_relations
    
    def _merge_entity_properties(self, properties: Dict[str, List]) -> Dict[str, Any]:
        """合并实体属性"""
        merged = {}
        
        for prop_name, prop_values in properties.items():
            if len(prop_values) == 1:
                merged[prop_name] = prop_values[0]
            else:
                # 对于多个值，选择最常见的或计算平均值
                if all(isinstance(v, (int, float)) for v in prop_values):
                    merged[prop_name] = sum(prop_values) / len(prop_values)
                else:
                    # 选择最频繁的值
                    from collections import Counter
                    most_common = Counter(prop_values).most_common(1)
                    merged[prop_name] = most_common[0][0] if most_common else prop_values[0]
        
        return merged
    
    def _calculate_overall_confidence(self, entity_fusion: Dict, 
                                    relation_fusion: Dict, 
                                    reasoning_steps: List[ReasoningStep]) -> float:
        """计算总体置信度"""
        entity_confidences = []
        relation_confidences = []
        reasoning_confidences = []
        
        # 实体置信度
        for entity_data in entity_fusion.values():
            entity_confidences.append(entity_data.get('average_confidence', 0.0))
        
        # 关系置信度
        for relation_data in relation_fusion.values():
            relation_confidences.append(relation_data.get('fused_confidence', 0.0))
        
        # 推理步骤置信度
        for step in reasoning_steps:
            reasoning_confidences.append(step.confidence)
        
        # 计算加权平均
        all_confidences = entity_confidences + relation_confidences + reasoning_confidences
        
        if all_confidences:
            return sum(all_confidences) / len(all_confidences)
        else:
            return 0.0
    
    def _validate_reasoning(self, fusion_result: FusionResult) -> Dict[str, Any]:
        """验证推理结果"""
        validation_results = {
            'consistency_checks': [],
            'contradiction_detection': [],
            'confidence_analysis': {},
            'overall_valid': False
        }
        
        # 一致性检查
        for entity_name, entity_data in fusion_result.entity_fusion.items():
            if len(entity_data.get('sources', [])) > 1:
                validation_results['consistency_checks'].append({
                    'entity': entity_name,
                    'status': 'consistent',
                    'note': f'Multiple sources agree on {entity_name}'
                })
        
        # 矛盾检测
        for relation_key, relation_data in fusion_result.relation_fusion.items():
            relation_types = relation_data.get('unique_relation_types', [])
            if len(relation_types) > 1:
                validation_results['contradiction_detection'].append({
                    'relation': relation_key,
                    'conflicting_types': relation_types,
                    'confidence': relation_data.get('fused_confidence', 0.0)
                })
        
        # 置信度分析
        validation_results['confidence_analysis'] = {
            'mean_confidence': fusion_result.confidence_score,
            'confidence_threshold': self.confidence_threshold,
            'meets_threshold': fusion_result.confidence_score >= self.confidence_threshold
        }
        
        # 总体有效性
        validation_results['overall_valid'] = (
            fusion_result.confidence_score >= self.confidence_threshold and
            len(validation_results['contradiction_detection']) == 0
        )
        
        return validation_results
    
    def _initialize_reasoning_rules(self) -> Dict[str, Any]:
        """初始化推理规则库"""
        return {
            'temporal_rules': {
                'before': ['先于', '之前', 'prior to'],
                'after': ['后于', '之后', 'after'],
                'during': ['在...期间', 'during'],
                'simultaneous': ['同时', 'simultaneous']
            },
            'causal_rules': {
                'cause_effect': ['导致', '造成', 'resulted in', 'caused'],
                'if_then': ['如果...那么', 'if...then'],
                'because': ['因为', '由于', 'because']
            },
            'semantic_rules': {
                'synonym': ['同义词', 'synonym'],
                'hypernym': ['上位词', 'hypernym'],
                'hyponym': ['下位词', 'hyponym']
            }
        }
    
    def _create_error_result(self, error_message: str) -> FusionResult:
        """创建错误结果"""
        return FusionResult(
            entity_fusion={},
            relation_fusion={},
            reasoning_steps=[],
            validation_results={'error': error_message},
            confidence_score=0.0
        )

def create_lmulator(config: Optional[Dict] = None) -> LMulator:
    """创建LMulator实例的工厂函数"""
    return LMulator(config)

if __name__ == "__main__":
    # 测试代码
    config = {
        'max_reasoning_steps': 10,
        'confidence_threshold': 0.7,
        'enable_temporal_reasoning': True,
        'enable_analogical_reasoning': True
    }
    
    lmulator = create_lmulator(config)
    
    # 测试数据
    test_input = {
        'entities': [
            {'name': '苹果公司', 'type': 'ORG', 'confidence': 0.8},
            {'name': '中国', 'type': 'LOC', 'confidence': 0.9}
        ],
        'relations': [
            {'source_name': '苹果公司', 'target_name': '中国', 'relation_type': 'located_in', 'confidence': 0.7}
        ],
        'context': {
            'enable_advanced_reasoning': True
        }
    }
    
    # 执行双轨推理
    result = lmulator.dual_track_reasoning(test_input)
    
    print("双轨推理结果:")
    print(f"实体融合: {len(result.entity_fusion)} 个实体")
    print(f"关系融合: {len(result.relation_fusion)} 个关系")
    print(f"推理步骤: {len(result.reasoning_steps)} 步")
    print(f"总体置信度: {result.confidence_score:.3f}")
    print(f"验证结果: {result.validation_results}")
