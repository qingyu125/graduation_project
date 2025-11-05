import ast
from typing import Dict

class SafeCodeEvaluator:
    """使用AST安全解析伪代码，避免exec的安全风险"""
    def evaluate(self, code: str) -> Dict:
        """解析伪代码中的entities/events/relations变量"""
        # 解析抽象语法树
        tree = ast.parse(code)
        variables = {}
        
        # 遍历AST节点，提取目标变量
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # 只处理简单变量赋值（entities/events/relations）
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ["entities", "events", "relations"]:
                        # 安全执行赋值表达式
                        try:
                            # 使用ast.literal_eval解析值（仅支持字面量）
                            value = ast.literal_eval(node.value)
                            variables[target.id] = value
                        except:
                            # 解析失败时赋空值
                            variables[target.id] = {} if target.id == "entities" else []
        
        return variables