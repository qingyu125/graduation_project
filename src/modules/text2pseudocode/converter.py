#!/usr/bin/env python3
"""
文本到伪代码转换器
基于CodeLlama-7B模型实现文本到伪代码的转换
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel, PeftConfig

from .prompt_manager import PromptTemplateManager
from ...utils.config import get_config

logger = logging.getLogger(__name__)

class CodeLlamaWrapper:
    """CodeLlama模型包装器"""
    
    def __init__(self, model_path: str = None, load_4bit: bool = True):
        self.config = get_config()
        self.model_config = self.config.models
        
        if model_path is None:
            model_path = self.model_config.base.model_path
        
        self.model_path = model_path
        self.load_4bit = load_4bit
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load_model(self) -> bool:
        """加载模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            
            # 配置4位量化
            if self.load_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.load_4bit else torch.float32
            )
            
            # 尝试加载LoRA权重
            lora_path = self.config.models.fine_tuned.checkpoint_path
            if Path(lora_path).exists():
                try:
                    logger.info(f"加载LoRA权重: {lora_path}")
                    self.model = PeftModel.from_pretrained(self.model, lora_path)
                    logger.info("LoRA权重加载成功")
                except Exception as e:
                    logger.warning(f"LoRA权重加载失败: {e}")
            
            # 创建推理pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.3) -> str:
        """生成文本"""
        try:
            if self.pipeline is None:
                logger.error("模型未加载")
                return ""
            
            # 设置生成参数
            self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
            
            # 生成文本
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generated_text = outputs[0]['generated_text']
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return ""

class Text2PseudoCodeConverter:
    """文本到伪代码转换器"""
    
    def __init__(self, 
                 model_path: str = None,
                 load_4bit: bool = True,
                 auto_classify: bool = True):
        self.config = get_config()
        self.prompt_config = self.config.prompt_templates
        
        # 初始化组件
        self.prompt_manager = PromptTemplateManager()
        self.model_wrapper = CodeLlamaWrapper(model_path, load_4bit)
        self.auto_classify = auto_classify
        
        # 转换历史
        self.conversion_history = []
        
    def load_model(self) -> bool:
        """加载模型"""
        return self.model_wrapper.load_model()
    
    def classify_input_text(self, text: str) -> str:
        """分类输入文本"""
        if not self.auto_classify:
            return "general"
        
        return self.prompt_manager.classify_text_type(text)
    
    def create_conversion_prompt(self, text: str, category: str = None) -> str:
        """创建转换提示"""
        if category is None:
            category = self.classify_input_text(text)
        
        return self.prompt_manager.create_prompt(text, category, include_examples=True)
    
    def convert_text_to_pseudocode(self, 
                                  text: str, 
                                  category: str = None,
                                  temperature: float = 0.3,
                                  max_length: int = 512) -> Dict[str, Any]:
        """转换文本到伪代码"""
        try:
            # 创建提示
            prompt = self.create_conversion_prompt(text, category)
            
            logger.info(f"开始转换文本到伪代码（类别: {category}）")
            
            # 生成伪代码
            generated_pseudocode = self.model_wrapper.generate(prompt, max_length, temperature)
            
            if not generated_pseudocode:
                logger.error("伪代码生成失败")
                return {
                    'success': False,
                    'error': '模型生成失败',
                    'original_text': text,
                    'category': category
                }
            
            # 后处理生成的伪代码
            processed_pseudocode = self._post_process_pseudocode(generated_pseudocode)
            
            # 评估生成质量
            quality_score = self._evaluate_quality(processed_pseudocode)
            
            # 保存转换记录
            from datetime import datetime
            conversion_record = {
                'timestamp': datetime.now().isoformat(),
                'original_text': text,
                'category': category,
                'prompt': prompt,
                'generated_pseudocode': generated_pseudocode,
                'processed_pseudocode': processed_pseudocode,
                'quality_score': quality_score,
                'temperature': temperature,
                'max_length': max_length
            }
            self.conversion_history.append(conversion_record)
            
            return {
                'success': True,
                'original_text': text,
                'category': category,
                'generated_pseudocode': generated_pseudocode,
                'processed_pseudocode': processed_pseudocode,
                'quality_score': quality_score,
                'extraction_ready': quality_score > 0.7
            }
            
        except Exception as e:
            logger.error(f"转换失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_text': text,
                'category': category
            }
    
    def _post_process_pseudocode(self, pseudocode: str) -> str:
        """后处理生成的伪代码"""
        try:
            # 清理格式
            lines = pseudocode.split('\n')
            processed_lines = []
            
            for line in lines:
                # 移除多余的空行
                if line.strip():
                    # 标准化缩进
                    line = line.expandtabs(4)
                    processed_lines.append(line)
            
            processed_pseudocode = '\n'.join(processed_lines)
            
            # 添加标准头部
            if not processed_pseudocode.startswith('#'):
                header = "# 自动生成的伪代码\n"
                processed_pseudocode = header + processed_pseudocode
            
            return processed_pseudocode
            
        except Exception as e:
            logger.warning(f"后处理失败: {e}")
            return pseudocode
    
    def _evaluate_quality(self, pseudocode: str) -> float:
        """评估伪代码质量"""
        try:
            score = 0.0
            max_score = 1.0
            
            # 检查基本结构
            if '#' in pseudocode:  # 有注释
                score += 0.2
            
            if '=' in pseudocode:  # 有变量定义
                score += 0.2
            
            if '(' in pseudocode and ')' in pseudocode:  # 有元组定义
                score += 0.2
            
            if 'relationship' in pseudocode.lower() or 'relation' in pseudocode.lower():
                score += 0.2
            
            # 检查语法正确性
            try:
                # 简单的语法检查
                lines = [line.strip() for line in pseudocode.split('\n') if line.strip()]
                valid_lines = sum(1 for line in lines if self._is_valid_line(line))
                if lines:
                    syntax_score = valid_lines / len(lines)
                    score += syntax_score * 0.2
            except:
                pass
            
            return min(score, max_score)
            
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")
            return 0.5
    
    def _is_valid_line(self, line: str) -> bool:
        """检查单行语法是否有效"""
        line = line.strip()
        if not line or line.startswith('#'):
            return True
        
        # 基本的Python语法检查
        valid_patterns = [
            r'^#.*',  # 注释
            r'^\w+\s*=\s*".*"',  # 字符串赋值
            r'^\w+\s*=\s*\w+',  # 变量赋值
            r'^\w+\s*=\s*\([^)]+\)',  # 元组赋值
            r'^if\s+.*:',  # 条件语句
            r'^for\s+.*:',  # 循环语句
            r'^print\(.+\)',  # 打印语句
        ]
        
        import re
        for pattern in valid_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def batch_convert(self, 
                     texts: List[str], 
                     categories: List[str] = None,
                     **kwargs) -> List[Dict[str, Any]]:
        """批量转换文本"""
        results = []
        
        if categories is None:
            categories = [None] * len(texts)
        
        for i, text in enumerate(texts):
            category = categories[i] if i < len(categories) else None
            result = self.convert_text_to_pseudocode(text, category, **kwargs)
            results.append(result)
        
        logger.info(f"批量转换完成，处理了 {len(texts)} 个文本")
        return results
    
    def save_conversion_history(self, output_path: str = "./results/conversion_history.json"):
        """保存转换历史"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversion_history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"转换历史已保存到: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"保存转换历史失败: {e}")
            return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取转换统计信息"""
        if not self.conversion_history:
            return {}
        
        total_conversions = len(self.conversion_history)
        successful_conversions = sum(1 for record in self.conversion_history if record.get('success'))
        avg_quality_score = sum(record.get('quality_score', 0) for record in self.conversion_history) / total_conversions
        
        # 分类统计
        category_stats = {}
        for record in self.conversion_history:
            category = record.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'count': 0, 'total_quality': 0}
            category_stats[category]['count'] += 1
            category_stats[category]['total_quality'] += record.get('quality_score', 0)
        
        for category in category_stats:
            category_stats[category]['avg_quality'] = category_stats[category]['total_quality'] / category_stats[category]['count']
        
        return {
            'total_conversions': total_conversions,
            'successful_conversions': successful_conversions,
            'success_rate': successful_conversions / total_conversions,
            'average_quality_score': avg_quality_score,
            'category_statistics': category_stats
        }

class PseudoCodeQualityAssessor:
    """伪代码质量评估器"""
    
    def __init__(self):
        self.config = get_config()
        self.extraction_config = self.config.extraction
    
    def assess_quality(self, pseudocode: str) -> Dict[str, Any]:
        """全面评估伪代码质量"""
        try:
            assessments = {
                'syntax_validity': self._assess_syntax(pseudocode),
                'structure_quality': self._assess_structure(pseudocode),
                'semantic_consistency': self._assess_semantics(pseudocode),
                'extraction_readiness': self._assess_extraction_readiness(pseudocode)
            }
            
            # 计算综合评分
            overall_score = sum(assessments.values()) / len(assessments)
            
            assessments['overall_score'] = overall_score
            assessments['quality_level'] = self._get_quality_level(overall_score)
            
            return assessments
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return {'error': str(e)}
    
    def _assess_syntax(self, pseudocode: str) -> float:
        """评估语法正确性"""
        try:
            lines = [line.strip() for line in pseudocode.split('\n') if line.strip()]
            valid_lines = 0
            
            for line in lines:
                if self._is_valid_line(line):
                    valid_lines += 1
            
            return valid_lines / len(lines) if lines else 0.0
            
        except:
            return 0.5
    
    def _assess_structure(self, pseudocode: str) -> float:
        """评估结构质量"""
        score = 0.0
        max_score = 1.0
        
        # 检查是否有实体定义
        if re.search(r'\w+\s*=\s*["\'].*["\']', pseudocode):
            score += 0.3
        
        # 检查是否有关系定义
        if re.search(r'\w+\s*=\s*\([^)]+\)', pseudocode):
            score += 0.3
        
        # 检查是否有推理逻辑
        if 'if' in pseudocode or 'for' in pseudocode or 'print' in pseudocode:
            score += 0.2
        
        # 检查注释
        if '#' in pseudocode:
            score += 0.2
        
        return min(score, max_score)
    
    def _assess_semantics(self, pseudocode: str) -> float:
        """评估语义一致性"""
        # 简化的语义评估
        score = 0.5  # 默认分数
        
        # 检查实体和关系的一致性
        entities = re.findall(r'(\w+)\s*=\s*["\']([^"\']+)["\']', pseudocode)
        relations = re.findall(r'(\w+)\s*=\s*\(([^)]+)\)', pseudocode)
        
        if entities and relations:
            # 检查关系中引用的实体是否在实体定义中存在
            entity_names = [name for name, value in entities]
            
            for rel_name, rel_content in relations:
                # 提取关系中的实体引用
                referenced_entities = re.findall(r'["\']([^"\']+)["\']', rel_content)
                
                # 检查引用是否合理
                if len(referenced_entities) >= 2:
                    # 至少包含两个实体引用
                    score += 0.3
                
                # 检查实体名称一致性
                for ref_entity in referenced_entities:
                    if ref_entity in entity_names or any(ref_entity.lower() in name.lower() for name in entity_names):
                        score += 0.1
        
        return min(score, 1.0)
    
    def _assess_extraction_readiness(self, pseudocode: str) -> float:
        """评估是否准备好进行要素抽取"""
        score = 0.0
        max_score = 1.0
        
        # 检查基本要素
        has_entities = bool(re.search(r'\w+\s*=\s*["\'][^"\']+["\']', pseudocode))
        has_relations = bool(re.search(r'\w+\s*=\s*\([^)]+\)', pseudocode))
        
        if has_entities:
            score += 0.4
        
        if has_relations:
            score += 0.4
        
        # 检查结构完整性
        if pseudocode.count('\n') >= 3:  # 至少有几行代码
            score += 0.2
        
        return min(score, max_score)
    
    def _is_valid_line(self, line: str) -> bool:
        """检查单行是否有效"""
        import re
        line = line.strip()
        
        if not line or line.startswith('#'):
            return True
        
        valid_patterns = [
            r'^#.*',  # 注释
            r'^\w+\s*=\s*".*"',  # 字符串赋值
            r'^\w+\s*=\s*\w+',  # 变量赋值
            r'^\w+\s*=\s*\([^)]+\)',  # 元组赋值
            r'^if\s+.*:',  # 条件语句
            r'^for\s+.*:',  # 循环语句
            r'^print\(.+\)',  # 打印语句
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _get_quality_level(self, score: float) -> str:
        """获取质量等级"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.6:
            return "良好"
        elif score >= 0.4:
            return "一般"
        else:
            return "较差"


# 便捷函数
def create_converter(model_path: str = None, load_4bit: bool = True) -> Text2PseudoCodeConverter:
    """创建转换器实例"""
    converter = Text2PseudoCodeConverter(model_path, load_4bit)
    return converter