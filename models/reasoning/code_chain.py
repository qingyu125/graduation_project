from models.base.code_llama_wrapper import CodeLlamaWrapper
import json

class CodeChainReasoner:
    def __init__(self, model_wrapper: CodeLlamaWrapper):
        self.model = model_wrapper
        self.reason_template = """# 任务：基于要素代码链推理关系是否成立
# 已知要素：
{elements}

# 推理目标：判断{head}与{tail}的{rel_type}是否成立

# 推理步骤：
# 1. 检查实体是否存在
# 2. 检查关系是否在标注中
# 3. 检查知识图谱验证结果
# 4. 检查是否有事件支撑

# 代码推理：
def check_relation(elements, head, tail, rel_type):
    # 步骤1：实体存在性
    if head not in elements["entities"] or tail not in elements["entities"]:
        return False, "实体不存在"
    
    # 步骤2：关系标注检查
    target_rels = [r for r in elements["relations"] 
                  if r["head"]==head and r["tail"]==tail and r["type"]==rel_type]
    if not target_rels:
        return False, "未标注该关系"
    
    # 步骤3：知识图谱验证
    if not target_rels[0]["kg_verify"]:
        return False, "知识图谱中无此关系"
    
    # 步骤4：事件支撑检查
    if not target_rels[0]["event_support"]:
        return False, "无足够事件支撑"
    
    return True, "关系成立（实体存在+标注正确+知识验证+事件支撑）"

# 执行推理
result, reason = check_relation(
    elements={elements_json},
    head="{head}",
    tail="{tail}",
    rel_type="{rel_type}"
)

# 输出结论
print(f"结论：{result}")
print(f"推理依据：{reason}")
"""
    
    def reason(self, elements, head, tail, rel_type):
        """执行代码链推理"""
        elements_str = json.dumps(elements, ensure_ascii=False, indent=2)
        elements_json = json.dumps(elements, ensure_ascii=False)
        
        prompt = self.reason_template.format(
            elements=elements_str,
            elements_json=elements_json,
            head=head,
            tail=tail,
            rel_type=rel_type
        )
        
        # 生成推理结果
        reasoning_output = self.model.generate(prompt)
        
        # 解析结论
        conclusion = "结论：True" in reasoning_output
        reason = reasoning_output.split("推理依据：")[-1].strip()
        
        return {
            "conclusion": conclusion,
            "reason": reason,
            "code_chain": prompt
        }