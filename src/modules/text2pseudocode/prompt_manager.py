#!/usr/bin/env python3
"""
提示模板管理器
管理不同类型的少样本提示模板
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# 尝试不同的导入方式
try:
    from ...utils.config import get_config
except ImportError:
    try:
        from src.utils.config import get_config
    except ImportError:
        # 如果无法导入，使用默认配置
        def get_config(key=None, default=None):
            return default

logger = logging.getLogger(__name__)

@dataclass
class PromptExample:
    """提示模板示例"""
    input_text: str
    pseudo_code: str
    explanation: str = ""
    category: str = "general"

@dataclass
class PromptTemplate:
    """提示模板"""
    name: str
    category: str
    system_prompt: str
    examples: List[PromptExample]
    input_format: str
    output_format: str
    instructions: str

class PromptTemplateManager:
    """提示模板管理器"""
    
    def __init__(self):
        self.config = get_config()
        self.prompt_config = self.config.prompt_templates
        
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """初始化所有提示模板"""
        # 人物关系模板
        self.templates['person_relationship'] = PromptTemplate(
            name="人物关系模板",
            category="relationship",
            system_prompt=self._get_system_prompt(),
            examples=self._get_person_relationship_examples(),
            input_format="文本: {text}",
            output_format="伪代码: {pseudo_code}",
            instructions="请将给定的人物关系文本转换为结构化伪代码，明确表达人物实体及其关系。"
        )
        
        # 事件描述模板
        self.templates['event_description'] = PromptTemplate(
            name="事件描述模板", 
            category="event",
            system_prompt=self._get_system_prompt(),
            examples=self._get_event_description_examples(),
            input_format="文本: {text}",
            output_format="伪代码: {pseudo_code}",
            instructions="请将给定的事件描述文本转换为结构化伪代码，明确表达事件主体、动作、时间和地点等要素。"
        )
        
        # 组织关系模板
        self.templates['organization_relationship'] = PromptTemplate(
            name="组织关系模板",
            category="organization", 
            system_prompt=self._get_system_prompt(),
            examples=self._get_organization_examples(),
            input_format="文本: {text}",
            output_format="伪代码: {pseudo_code}",
            instructions="请将给定的组织关系文本转换为结构化伪代码，明确表达组织实体及其各种关系。"
        )
        
        # 通用模板
        self.templates['general'] = PromptTemplate(
            name="通用模板",
            category="general",
            system_prompt=self._get_system_prompt(),
            examples=self._get_general_examples(),
            input_format="文本: {text}",
            output_format="伪代码: {pseudo_code}",
            instructions="请将给定的文本转换为结构化伪代码，提取所有实体、事件和关系。"
        )
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个专业的文本到伪代码转换助手。你的任务是将自然语言文本转换为结构化的伪代码，以便后续进行要素抽取和关系推理。

转换原则：
1. 明确识别和定义所有实体（人物、组织、地点、事件等）
2. 清晰表达实体间的关系（职位关系、家庭关系、商业关系等）
3. 保持逻辑结构和语义完整性
4. 使用Python风格的变量命名和数据结构

输出格式要求：
- 使用变量定义实体：entity_name = "实体名称"
- 使用元组表达关系：relation = (head_entity, relationship_type, tail_entity)
- 添加必要的注释说明
- 保持代码的可读性和结构化

请严格按照以上要求进行转换。"""
    
    def _get_person_relationship_examples(self) -> List[PromptExample]:
        """获取人物关系示例"""
        return [
            PromptExample(
                input_text="张三和李四是同事，张三是李四的上级。",
                pseudo_code='''# 定义人物实体
zhang_san = "张三"
li_si = "李四"

# 定义关系
zhang_san_position = "上级"
li_si_position = "下级"

# 表达同事关系
colleague_relationship = (zhang_san, "同事", li_si)
zhang_san_to_li_si = (zhang_san, zhang_san_position, li_si)
li_si_to_zhang_san = (li_si, li_si_position, zhang_san)

# 推理结论
if colleague_relationship[1] == "同事":
    print(f"{zhang_san}和{li_si}是同事关系")''',
                explanation="提取人物实体张三和李四，以及他们的同事关系和职位关系",
                category="person_relationship"
            ),
            PromptExample(
                input_text="王五是赵六的丈夫，他们有两个孩子王小明和王小红。",
                pseudo_code='''# 定义人物实体
wang_wu = "王五"
zhao_liu = "赵六"
wang_xiao_ming = "王小明"
wang_xiao_hong = "王小红"

# 定义关系
spouse_relationship = (wang_wu, "丈夫", zhao_liu)
father_to_ming = (wang_wu, "父亲", wang_xiao_ming)
father_to_hong = (wang_wu, "父亲", wang_xiao_hong)
mother_to_ming = (zhao_liu, "母亲", wang_xiao_ming)
mother_to_hong = (zhao_liu, "母亲", wang_xiao_hong)

# 家庭关系推理
family_relationship = (wang_wu, "丈夫", zhao_liu)
children = [wang_xiao_ming, wang_xiao_hong]

for child in children:
    print(f"{wang_wu}和{zhao_liu}是{child}的父母")''',
                explanation="提取完整的家庭关系，包括配偶关系和亲子关系",
                category="person_relationship"
            ),
            PromptExample(
                input_text="马云是阿里巴巴集团的创始人兼前董事长。",
                pseudo_code='''# 定义人物和实体
ma_yun = "马云"
alibaba_group = "阿里巴巴集团"

# 定义职位关系
founder_position = "创始人"
chairman_position = "前董事长"

# 表达职位关系
ma_yun_founded_alibaba = (ma_yun, founder_position, alibaba_group)
ma_yun_chairman_alibaba = (ma_yun, chairman_position, alibaba_group)

# 推理结论
if founder_position in [pos[1] for pos in [ma_yun_founded_alibaba, ma_yun_chairman_alibaba]]:
    print(f"{ma_yun}是{alibaba_group}的重要创始人")''',
                explanation="提取人物马云及其在阿里巴巴集团的职位关系",
                category="person_relationship"
            )
        ]
    
    def _get_event_description_examples(self) -> List[PromptExample]:
        """获取事件描述示例"""
        return [
            PromptExample(
                input_text="苹果公司于2024年9月1日发布了新款iPhone手机。",
                pseudo_code='''# 定义实体
apple_inc = "苹果公司"
new_iphone = "新款iPhone手机"
release_date = "2024年9月1日"

# 定义事件
release_event = "发布"
event_time = release_date

# 表达发布事件
product_release = (apple_inc, release_event, new_iphone)
time_relationship = (release_event, "发生在", event_time)

# 产品属性
iphone_feature = "智能手机功能"

# 事件推理
if product_release[1] == "发布":
    print(f"{product_release[0]}在{event_time}发布了{product_release[2]}")''',
                explanation="提取发布事件的主体、产品、时间和属性",
                category="event_description"
            ),
            PromptExample(
                input_text="特斯拉CEO马斯克宣布将在上海建设第二座超级工厂。",
                pseudo_code='''# 定义实体
tesla_inc = "特斯拉公司"
elon_musk = "马斯克"
tesla_ceo = "CEO"
shanghai = "上海"
tesla_gigafactory_2 = "第二座超级工厂"

# 定义事件
announcement_event = "宣布"
construction_plan = "建设计划"

# 表达职位和事件
musk_as_ceo = (elon_musk, tesla_ceo, tesla_inc)
construction_announcement = (elon_musk, announcement_event, construction_plan)
factory_location = (construction_plan, "位于", shanghai)

# 推理结论
if construction_announcement[1] == "宣布":
    print(f"{construction_announcement[0]}宣布了{construction_announcement[2]}")
    print(f"工厂将建在{shanghai}")''',
                explanation="提取CEO身份、宣布事件、地点和建设计划",
                category="event_description"
            )
        ]
    
    def _get_organization_examples(self) -> List[PromptExample]:
        """获取组织关系示例"""
        return [
            PromptExample(
                input_text="微软公司与谷歌公司在人工智能领域达成战略合作协议。",
                pseudo_code='''# 定义组织实体
microsoft_corp = "微软公司"
google_corp = "谷歌公司"
ai_field = "人工智能领域"

# 定义合作关系
strategic_cooperation = "战略合作"
agreement_event = "达成协议"

# 表达合作关系
cooperation_agreement = (microsoft_corp, strategic_cooperation, google_corp)
ai_cooperation_field = (cooperation_agreement, "涉及领域", ai_field)
agreement_made = (cooperation_agreement, agreement_event, "已达成")

# 推理结论
if cooperation_agreement[1] == "战略合作":
    print(f"{microsoft_corp}和{google_corp}达成战略合作")
    print(f"合作领域：{ai_field}")''',
                explanation="提取组织实体、合作关系类型和合作领域",
                category="organization_relationship"
            ),
            PromptExample(
                input_text="华为技术有限公司是中国最大的通信设备制造商。",
                pseudo_code='''# 定义组织实体
huawei_tech = "华为技术有限公司"
china = "中国"
communication_equipment = "通信设备"

# 定义分类关系
largest_manufacturer = "最大制造商"
manufacturer_type = "通信设备制造商"

# 表达分类关系
huawei_as_leader = (huawei_tech, largest_manufacturer, communication_equipment)
huawei_business_type = (huawei_tech, "属于", manufacturer_type)
china_location = (manufacturer_type, "总部位于", china)

# 推理结论
if manufacturer_type in [rel[1] for rel in [huawei_business_type]]:
    print(f"{huawei_tech}是{manufacturer_type}")''',
                explanation="提取组织分类关系、规模地位和地理位置",
                category="organization_relationship"
            )
        ]
    
    def _get_general_examples(self) -> List[PromptExample]:
        """获取通用示例"""
        return [
            PromptExample(
                input_text="北京是中华人民共和国的首都。",
                pseudo_code='''# 定义地理实体
beijing = "北京"
china = "中华人民共和国"
capital_relationship = "首都"

# 表达地理关系
beijing_capital_of_china = (beijing, capital_relationship, china)
geographical_location = (beijing, "位于", "华北地区")

# 推理结论
if beijing_capital_of_china[1] == "首都":
    print(f"{beijing_capital_of_china[0]}是{beijing_capital_of_china[2]}的首都")''',
                explanation="提取地理实体和首都关系",
                category="general"
            ),
            PromptExample(
                input_text="爱因斯坦提出了相对论理论。",
                pseudo_code='''# 定义人物和理论实体
albert_einstein = "爱因斯坦"
relativity_theory = "相对论理论"
scientific_contribution = "提出"

# 表达科学贡献关系
einstein_contribution = (albert_einstein, scientific_contribution, relativity_theory)
theory_type = (relativity_theory, "属于", "物理学理论")

# 推理结论
if einstein_contribution[1] == "提出":
    print(f"{einstein_contribution[0]}提出了{einstein_contribution[2]}")''',
                explanation="提取科学家和科学理论的关系",
                category="general"
            )
        ]
    
    def get_template(self, category: str = "general") -> Optional[PromptTemplate]:
        """获取指定类别的模板"""
        return self.templates.get(category)
    
    def get_all_categories(self) -> List[str]:
        """获取所有可用的模板类别"""
        return list(self.templates.keys())
    
    def create_prompt(self, 
                     text: str, 
                     category: str = "general",
                     include_examples: bool = True) -> str:
        """创建完整的提示"""
        template = self.get_template(category)
        if not template:
            logger.warning(f"未找到模板类别: {category}")
            template = self.get_template("general")
        
        # 构建提示
        prompt_parts = [template.system_prompt]
        
        if include_examples and template.examples:
            prompt_parts.append("\n## 示例:")
            for i, example in enumerate(template.examples[:3], 1):  # 最多使用3个示例
                prompt_parts.append(f"\n### 示例 {i}:")
                prompt_parts.append(f"输入: {example.input_text}")
                prompt_parts.append(f"输出: {example.pseudo_code}")
                if example.explanation:
                    prompt_parts.append(f"说明: {example.explanation}")
        
        prompt_parts.append(f"\n## 任务:")
        prompt_parts.append(template.instructions)
        prompt_parts.append(f"\n现在请处理以下文本:")
        prompt_parts.append(f"文本: {text}")
        prompt_parts.append("\n请转换为伪代码:")
        
        return "\n".join(prompt_parts)
    
    def classify_text_type(self, text: str) -> str:
        """分类文本类型"""
        text_lower = text.lower()
        
        # 人物关系关键词
        person_keywords = ["丈夫", "妻子", "父亲", "母亲", "儿子", "女儿", "兄弟", "姐妹", "同事", "上级", "下属", "CEO", "董事长", "总裁"]
        # 事件关键词
        event_keywords = ["发布", "宣布", "召开", "举行", "建设", "收购", "投资", "合作", "签约"]
        # 组织关键词
        org_keywords = ["公司", "集团", "企业", "组织", "机构", "部门", "大学", "医院", "银行"]
        
        # 统计关键词出现次数
        person_count = sum(1 for keyword in person_keywords if keyword in text_lower)
        event_count = sum(1 for keyword in event_keywords if keyword in text_lower)
        org_count = sum(1 for keyword in org_keywords if keyword in text_lower)
        
        # 根据统计结果分类
        if person_count > event_count and person_count > org_count:
            return "person_relationship"
        elif event_count > org_count:
            return "event_description" 
        elif org_count > 0:
            return "organization_relationship"
        else:
            return "general"
    
    def save_templates(self, output_dir: str = "./data/processed"):
        """保存模板到文件"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化格式
            template_data = {}
            for category, template in self.templates.items():
                template_data[category] = {
                    'name': template.name,
                    'category': template.category,
                    'system_prompt': template.system_prompt,
                    'examples': [
                        {
                            'input_text': ex.input_text,
                            'pseudo_code': ex.pseudo_code,
                            'explanation': ex.explanation,
                            'category': ex.category
                        } for ex in template.examples
                    ],
                    'input_format': template.input_format,
                    'output_format': template.output_format,
                    'instructions': template.instructions
                }
            
            # 保存到JSON文件
            templates_path = output_path / "prompt_templates.json"
            with open(templates_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"提示模板已保存到: {templates_path}")
            return str(templates_path)
            
        except Exception as e:
            logger.error(f"保存模板失败: {e}")
            return ""