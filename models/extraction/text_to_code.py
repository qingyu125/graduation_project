from models.base.code_llama_wrapper import CodeLlamaWrapper
import yaml

class TextToCodeConverter:
    def __init__(self, model_wrapper: CodeLlamaWrapper, config_path="./config/data_config.yaml"):
        self.model = model_wrapper
        self.data_config = yaml.safe_load(open(config_path))
        self.prompt_template = self._build_prompt_template()
    
    def _build_prompt_template(self):
        """构建少样本提示模板"""
        entities = ", ".join(self.data_config["要素类型定义"]["entities"])
        relations = ", ".join(self.data_config["要素类型定义"]["relations"])
        events = ", ".join(self.data_config["要素类型定义"]["events"])
        
        return f"""# 任务：将法律文本转换为含实体、事件、关系的伪代码
# 实体类型：{entities}
# 关系类型：{relations}
# 事件触发词：{events}

# 示例1：
# 文本：张三于2023年加入丙公司，担任CEO
# 伪代码：
entities = {{"张三": "PER", "丙公司": "ORG", "2023年": "TIME"}}
events = [{{"trigger": "加入", "time": "2023年", "participants": ["张三", "丙公司"], "role": "CEO"}}]
relations = [{{"head": "张三", "tail": "丙公司", "type": "雇佣关系"}}]

# 示例2：
# 文本：A公司与B公司因合同纠纷于2024年解约
# 伪代码：
entities = {{"A公司": "ORG", "B公司": "ORG", "2024年": "TIME"}}
events = [{{"trigger": "解约", "time": "2024年", "participants": ["A公司", "B公司"]}}]
relations = [{{"head": "A公司", "tail": "B公司", "type": "合同纠纷"}}]

# 待转换文本：{{text}}
# 伪代码：
"""
    
    def convert(self, text):
        """将文本转换为伪代码"""
        prompt = self.prompt_template.format(text=text)
        generated_code = self.model.generate(prompt)
        # 提取伪代码部分（去除提示模板）
        code = generated_code.split("# 伪代码：")[-1].strip()
        return code