"""
AST解析器模块
解析伪代码并构建抽象语法树结构
"""

import ast
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """AST节点类型枚举"""
    FUNCTION = "function"
    VARIABLE = "variable"
    RELATION = "relation"
    ENTITY = "entity"
    CONDITION = "condition"
    ACTION = "action"
    COMMENT = "comment"
    ASSIGNMENT = "assignment"
    COMPARISON = "comparison"
    LOGICAL_OP = "logical_op"
    STRING_LITERAL = "string_literal"
    IDENTIFIER = "identifier"


@dataclass
class ASTNode:
    """AST节点数据结构"""
    node_type: NodeType
    content: str
    line_number: int
    children: List['ASTNode']
    metadata: Dict[str, Any]
    parent: Optional['ASTNode'] = None


class PseudoCodeASTParser:
    """伪代码AST解析器"""
    
    def __init__(self):
        self.pseudo_keywords = {
            'if', 'else', 'elif', 'for', 'while', 'return', 'function',
            'end', 'then', 'do', 'while', 'repeat', 'until', 'case',
            'when', 'otherwise', 'begin', 'end', 'var', 'let', 'set',
            'relate', 'entity', 'event', 'attribute', 'type', 'find',
            'extract', 'analyze', 'process'
        }
        
        self.relation_keywords = [
            'relate', 'relationship', 'is_related_to', 'has_relation',
            'connected_to', 'interacts_with', 'works_for', 'located_in',
            'member_of', 'part_of', 'affects', 'causes', 'influences'
        ]
        
        self.entity_patterns = {
            'PERSON': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'ORG': r'\b[A-Z][A-Z][A-Za-z]*\b',
            'LOC': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(City|State|Country|Street|Road|University|Company)\b',
            'TIME': r'\b\d{4}|\b\d{1,2}/\d{1,2}/\d{4}|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
        }
    
    def parse_pseudocode(self, pseudocode: str) -> ASTNode:
        """
        解析伪代码并返回AST根节点
        
        Args:
            pseudocode: 伪代码字符串
            
        Returns:
            ASTNode: 解析后的AST根节点
        """
        try:
            logger.info(f"开始解析伪代码，长度: {len(pseudocode)} 字符")
            
            # 预处理伪代码
            processed_code = self._preprocess_pseudocode(pseudocode)
            
            # 分解为语句
            statements = self._split_statements(processed_code)
            
            # 构建AST
            root_node = ASTNode(
                node_type=NodeType.FUNCTION,
                content="root",
                line_number=0,
                children=[],
                metadata={}
            )
            
            for statement in statements:
                try:
                    stmt_node = self._parse_statement(statement)
                    if stmt_node:
                        stmt_node.parent = root_node
                        root_node.children.append(stmt_node)
                except Exception as e:
                    logger.warning(f"解析语句失败: {statement}, 错误: {str(e)}")
                    # 创建错误节点
                    error_node = ASTNode(
                        node_type=NodeType.COMMENT,
                        content=f"解析错误: {statement}",
                        line_number=0,
                        children=[],
                        metadata={"error": str(e)}
                    )
                    error_node.parent = root_node
                    root_node.children.append(error_node)
            
            logger.info(f"AST解析完成，根节点有 {len(root_node.children)} 个子节点")
            return root_node
            
        except Exception as e:
            logger.error(f"AST解析失败: {str(e)}")
            raise
    
    def _preprocess_pseudocode(self, code: str) -> str:
        """预处理伪代码"""
        # 移除多余空白字符
        code = re.sub(r'\s+', ' ', code.strip())
        
        # 标准化伪代码关键词
        code = code.replace('=>', '=')
        code = code.replace('->', '->')
        code = code.replace(':=', '=')
        
        return code
    
    def _split_statements(self, code: str) -> List[str]:
        """将伪代码分解为独立语句"""
        # 识别分隔符
        separators = ['\n', ';', 'then', 'end', 'do']
        
        statements = []
        current_statement = ""
        
        for char in code:
            current_statement += char
            
            # 检查是否到达语句结束
            if any(sep in current_statement[-10:] for sep in ['\n', ';']) or \
               any(sep in current_statement.lower()[-15:] for sep in [' then', ' end', ' do']):
                
                if current_statement.strip():
                    statements.append(current_statement.strip())
                    current_statement = ""
        
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements
    
    def _parse_statement(self, statement: str) -> Optional[ASTNode]:
        """解析单个语句"""
        statement = statement.strip()
        
        if not statement:
            return None
        
        # 识别语句类型
        if self._is_relation_statement(statement):
            return self._parse_relation_statement(statement)
        elif self._is_assignment_statement(statement):
            return self._parse_assignment_statement(statement)
        elif self._is_conditional_statement(statement):
            return self._parse_conditional_statement(statement)
        elif self._is_comment(statement):
            return self._parse_comment(statement)
        else:
            return self._parse_generic_statement(statement)
    
    def _is_relation_statement(self, statement: str) -> bool:
        """判断是否为关系语句"""
        statement_lower = statement.lower()
        return any(keyword in statement_lower for keyword in self.relation_keywords)
    
    def _is_assignment_statement(self, statement: str) -> bool:
        """判断是否为赋值语句"""
        return '=' in statement and not statement.lower().startswith('if')
    
    def _is_conditional_statement(self, statement: str) -> bool:
        """判断是否为条件语句"""
        return statement.lower().startswith(('if', 'elif', 'else'))
    
    def _is_comment(self, statement: str) -> bool:
        """判断是否为注释"""
        return statement.strip().startswith(('#', '//', '/*', 'rem', 'comment'))
    
    def _parse_relation_statement(self, statement: str) -> ASTNode:
        """解析关系语句"""
        # 提取关系类型和参与者
        participants = self._extract_participants(statement)
        relation_type = self._extract_relation_type(statement)
        
        metadata = {
            'relation_type': relation_type,
            'participants': participants,
            'original_statement': statement
        }
        
        return ASTNode(
            node_type=NodeType.RELATION,
            content=f"{relation_type}({', '.join(participants)})",
            line_number=1,
            children=[],
            metadata=metadata
        )
    
    def _parse_assignment_statement(self, statement: str) -> ASTNode:
        """解析赋值语句"""
        parts = statement.split('=', 1)
        if len(parts) == 2:
            var_name = parts[0].strip()
            value = parts[1].strip()
            
            # 检查是否为实体赋值
            entity_type = self._detect_entity_type(value)
            
            metadata = {
                'variable': var_name,
                'value': value,
                'entity_type': entity_type,
                'is_entity': entity_type is not None
            }
            
            return ASTNode(
                node_type=NodeType.VARIABLE,
                content=statement,
                line_number=1,
                children=[],
                metadata=metadata
            )
        
        return self._parse_generic_statement(statement)
    
    def _parse_conditional_statement(self, statement: str) -> ASTNode:
        """解析条件语句"""
        # 提取条件表达式
        condition = self._extract_condition(statement)
        
        metadata = {
            'condition': condition,
            'statement_type': 'conditional'
        }
        
        return ASTNode(
            node_type=NodeType.CONDITION,
            content=statement,
            line_number=1,
            children=[],
            metadata=metadata
        )
    
    def _parse_comment(self, statement: str) -> ASTNode:
        """解析注释语句"""
        return ASTNode(
            node_type=NodeType.COMMENT,
            content=statement,
            line_number=1,
            children=[],
            metadata={'original': statement}
        )
    
    def _parse_generic_statement(self, statement: str) -> ASTNode:
        """解析通用语句"""
        return ASTNode(
            node_type=NodeType.IDENTIFIER,
            content=statement,
            line_number=1,
            children=[],
            metadata={}
        )
    
    def _extract_participants(self, statement: str) -> List[str]:
        """提取参与者实体"""
        participants = []
        
        # 查找命名实体
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, statement, re.IGNORECASE)
            participants.extend(matches)
        
        # 提取括号中的实体
        bracket_content = re.search(r'\(([^)]+)\)', statement)
        if bracket_content:
            entities = [e.strip() for e in bracket_content.group(1).split(',')]
            participants.extend(entities)
        
        return list(set(participants))  # 去重
    
    def _extract_relation_type(self, statement: str) -> str:
        """提取关系类型"""
        statement_lower = statement.lower()
        
        for keyword in self.relation_keywords:
            if keyword in statement_lower:
                return keyword
        
        # 尝试提取通用的关系词汇
        relation_match = re.search(r'(is_related_to|has_relation|connected_to|works_for|located_in)', 
                                  statement, re.IGNORECASE)
        if relation_match:
            return relation_match.group(1)
        
        return "related_to"
    
    def _detect_entity_type(self, text: str) -> Optional[str]:
        """检测实体类型"""
        for entity_type, pattern in self.entity_patterns.items():
            if re.match(pattern, text, re.IGNORECASE):
                return entity_type
        
        # 简单的基于内容的启发式规则
        if any(word in text.lower() for word in ['person', 'people', 'individual', 'person']):
            return 'PERSON'
        elif any(word in text.lower() for word in ['company', 'organization', 'corp', 'inc']):
            return 'ORG'
        elif any(word in text.lower() for word in ['city', 'state', 'country', 'location']):
            return 'LOC'
        
        return None
    
    def _extract_condition(self, statement: str) -> str:
        """提取条件表达式"""
        # 简单提取if后面的条件
        if_statement = re.search(r'if\s+(.+?)(?:\s+then|\s+do|\s*:|\s+{)', statement, re.IGNORECASE)
        if if_statement:
            return if_statement.group(1).strip()
        
        return statement
    
    def print_ast(self, node: ASTNode, indent: str = "", is_last: bool = True):
        """打印AST结构（调试用）"""
        connector = "└── " if is_last else "├── "
        print(f"{indent}{connector}[{node.node_type.value}] {node.content}")
        
        if node.children:
            new_indent = indent + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self.print_ast(child, new_indent, is_last_child)