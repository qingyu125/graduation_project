import ast
from utils.code_safety import SafeCodeEvaluator

class CodeParser:
    def __init__(self):
        self.safe_evaluator = SafeCodeEvaluator()
    
    def parse(self, pseudo_code):
        """解析伪代码为结构化要素"""
        # 安全执行伪代码，提取变量
        try:
            variables = self.safe_evaluator.evaluate(pseudo_code)
            return {
                "entities": variables.get("entities", {}),
                "events": variables.get("events", []),
                "relations": variables.get("relations", [])
            }
        except Exception as e:
            print(f"解析错误: {e}")
            return {"entities": {}, "events": [], "relations": []}