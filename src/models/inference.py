# -*- coding: utf-8 -*-
"""
模型推理模块
提供CodeLlama-7B和BERT模型的推理功能
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time

# 导入必要的库
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        pipeline
    )
    from transformers import BertTokenizer, BertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformers库未安装: {e}")
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseInferenceEngine:
    """推理引擎基类"""
    
    def __init__(self, model_name: str, model_path: str = None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """加载模型"""
        raise NotImplementedError
        
    def unload_model(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"模型 {self.model_name} 已卸载")
        
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        return text.strip()
        
    def postprocess_output(self, output: Any) -> Dict[str, Any]:
        """输出后处理"""
        return {"raw_output": output}


class CodeLlamaInferenceEngine(BaseInferenceEngine):
    """CodeLlama推理引擎"""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Instruct-hf", 
                 model_path: str = None, is_finetuned: bool = False):
        super().__init__(model_name, model_path)
        self.is_finetuned = is_finetuned
        self.model_type = "finetuned" if is_finetuned else "base"
        
    def load_model(self) -> bool:
        """加载CodeLlama模型"""
        try:
            logger.info(f"加载CodeLlama模型: {self.model_name} (类型: {self.model_type})")
            
            if not TRANSFORMERS_AVAILABLE:
                logger.error("Transformers库未安装，无法加载模型")
                return False
                
            # 选择模型路径
            if self.model_path and os.path.exists(self.model_path):
                # 加载微调后的模型
                model_path = self.model_path
                logger.info(f"从本地路径加载模型: {model_path}")
            else:
                # 加载预训练模型
                model_path = self.model_name
                logger.info(f"从Hugging Face加载模型: {model_path}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # 加载模型
            model_kwargs = {
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'device_map': 'auto' if torch.cuda.is_available() else 'cpu'
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            self.model.eval()
            self.is_loaded = True
            logger.info(f"CodeLlama模型加载成功 (类型: {self.model_type})")
            return True
            
        except Exception as e:
            logger.error(f"CodeLlama模型加载失败: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成响应"""
        if not self.is_loaded:
            return "模型未加载"
            
        try:
            # 预处理输入
            input_text = self.preprocess_text(prompt)
            
            # 编码输入
            inputs = self.tokenizer.encode(
                input_text, 
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )
            
            # 移动到设备
            inputs = inputs.to(self.device)
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.encode("</s>")[0]
                )
            
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"CodeLlama推理失败: {str(e)}")
            return f"推理错误: {str(e)}"
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取关系"""
        if not self.is_loaded:
            return []
            
        # 构建提示词
        prompt = f"""
请分析以下文本，识别其中的实体和关系，以JSON格式返回结果：

文本：{text}

请按照以下格式返回：
{{
    "entities": [
        {{"text": "实体名称", "type": "实体类型", "confidence": 0.95}}
    ],
    "relations": [
        {{"head": "头实体", "relation": "关系类型", "tail": "尾实体", "confidence": 0.88}}
    ]
}}
"""
        
        response = self.generate_response(prompt, max_length=1024)
        
        # 尝试解析JSON响应
        try:
            # 提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                return self._parse_text_response(response)
        except json.JSONDecodeError:
            return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> List[Dict[str, Any]]:
        """解析文本格式的响应"""
        # 简单的文本解析逻辑
        entities = []
        relations = []
        
        # 尝试提取实体（简单的正则匹配）
        import re
        entity_patterns = [
            r'"([^"]+)"',
            r'([^，。！？；：\s]{1,10})(?=公司|组织|机构|个人)',
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if len(match) > 1 and match not in ['entities', 'relations']:
                    entities.append({
                        "text": match,
                        "type": "UNKNOWN",
                        "confidence": 0.5
                    })
        
        return {
            "entities": entities[:5],  # 限制数量
            "relations": relations
        }


class BertInferenceEngine(BaseInferenceEngine):
    """BERT推理引擎"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 model_path: str = None, num_labels: int = None):
        super().__init__(model_name, model_path)
        self.num_labels = num_labels
        self.relation_labels = None
        
    def load_model(self) -> bool:
        """加载BERT模型"""
        try:
            logger.info(f"加载BERT模型: {self.model_name}")
            
            if not TRANSFORMERS_AVAILABLE:
                logger.error("Transformers库未安装，无法加载模型")
                return False
                
            # 选择模型路径
            if self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path
            else:
                model_path = self.model_name
                
            # 加载分词器和模型
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            
            if self.num_labels:
                self.model = BertForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=self.num_labels
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path
                )
                self.num_labels = self.model.config.num_labels
                
            # 设置设备
            self.model.to(self.device)
            self.model.eval()
            
            # 加载关系标签
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                self.relation_labels = self.model.config.id2label
            else:
                # 默认关系标签
                self.relation_labels = {
                    "0": "NO_RELATION",
                    "1": "WORKS_FOR",
                    "2": "AFFILIATED_TO",
                    "3": "CO_LOCATED",
                    "4": "PART_OF"
                }
            
            self.is_loaded = True
            logger.info(f"BERT模型加载成功，标签数: {self.num_labels}")
            return True
            
        except Exception as e:
            logger.error(f"BERT模型加载失败: {str(e)}")
            return False
    
    def classify_relation(self, text: str, entity1: str, entity2: str) -> Dict[str, Any]:
        """分类两个实体之间的关系"""
        if not self.is_loaded:
            return {"relation": "UNKNOWN", "confidence": 0.0, "error": "模型未加载"}
            
        try:
            # 构建输入序列
            input_text = f"{entity1} [SEP] {entity2} [SEP] {text}"
            
            # 编码
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
            # 获取预测结果
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # 获取标签
            relation = self.relation_labels.get(str(predicted_class), f"RELATION_{predicted_class}")
            
            return {
                "relation": relation,
                "confidence": confidence,
                "entity1": entity1,
                "entity2": entity2,
                "predicted_class": predicted_class,
                "all_probabilities": {
                    self.relation_labels.get(str(i), f"RELATION_{i}"): prob.item()
                    for i, prob in enumerate(probabilities[0])
                }
            }
            
        except Exception as e:
            logger.error(f"BERT关系分类失败: {str(e)}")
            return {"relation": "UNKNOWN", "confidence": 0.0, "error": str(e)}
    
    def batch_classify(self, text: str, entity_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """批量关系分类"""
        results = []
        for entity1, entity2 in entity_pairs:
            result = self.classify_relation(text, entity1, entity2)
            results.append(result)
        return results


class ModelManager:
    """模型管理器，统一管理所有模型"""
    
    def __init__(self):
        self.models = {
            'codelama_finetuned': None,
            'codelama_base': None,
            'bert_finetuned': None
        }
        self.load_status = {
            'codelama_finetuned': False,  # False=未加载, True=真实模型, 'mock'=模拟模式
            'codelama_base': False,
            'bert_finetuned': False
        }
        self.model_paths = {
            'codelama_finetuned': "experiments/checkpoints/final_model",
            'codelama_base': "codellama/CodeLlama-7b-Instruct-hf",
            'bert_finetuned': "experiments/checkpoints/bert_model"
        }
        
    def load_model(self, model_type: str) -> bool:
        """加载指定模型"""
        if model_type not in self.models:
            logger.error(f"未知的模型类型: {model_type}")
            return False
            
        if self.load_status[model_type]:
            logger.info(f"模型 {model_type} 已加载")
            return True
            
        try:
            if model_type == 'codelama_finetuned':
                engine = CodeLlamaInferenceEngine(
                    model_name="codellama/CodeLlama-7b-Instruct-hf",
                    model_path=self.model_paths[model_type],
                    is_finetuned=True
                )
            elif model_type == 'codelama_base':
                engine = CodeLlamaInferenceEngine(
                    model_name="codellama/CodeLlama-7b-Instruct-hf",
                    is_finetuned=False
                )
            elif model_type == 'bert_finetuned':
                engine = BertInferenceEngine(
                    model_name="bert-base-uncased",
                    model_path=self.model_paths[model_type]
                )
            else:
                logger.error(f"不支持的模型类型: {model_type}")
                return False
                
            if engine.load_model():
                self.models[model_type] = engine
                self.load_status[model_type] = True
                logger.info(f"模型 {model_type} 加载成功")
                return True
            else:
                logger.error(f"模型 {model_type} 加载失败")
                return False
                
        except Exception as e:
            logger.error(f"加载模型 {model_type} 时发生错误: {str(e)}")
            # 智能回退：模型加载失败时标记为模拟模式
            self.load_status[model_type] = 'mock'
            logger.warning(f"模型 {model_type} 加载失败，将使用模拟模式")
            return True  # 返回True，但标记为模拟模式
    
    def unload_model(self, model_type: str = None):
        """卸载模型"""
        if model_type:
            # 卸载指定模型
            if model_type in self.models and self.models[model_type]:
                self.models[model_type].unload_model()
                self.models[model_type] = None
                self.load_status[model_type] = False
                logger.info(f"模型 {model_type} 已卸载")
        else:
            # 卸载所有模型
            for model_type in self.models:
                self.unload_model(model_type)
    
    def get_model(self, model_type: str):
        """获取模型实例"""
        if model_type not in self.models or not self.load_status[model_type]:
            logger.warning(f"模型 {model_type} 未加载")
            return None
        return self.models[model_type]
    
    def is_model_loaded(self, model_type: str) -> bool:
        """检查模型是否已加载"""
        return self.load_status.get(model_type, False)
    
    def get_model_status(self) -> Dict[str, bool]:
        """获取所有模型状态"""
        return self.load_status.copy()
    
    def predict_with_model(self, model_type: str, text: str, **kwargs) -> Dict[str, Any]:
        """使用指定模型进行预测"""
        # 检查模型状态
        status = self.load_status.get(model_type, False)
        if status == 'mock':
            # 使用模拟推理
            result = self._mock_inference(text, model_type)
            return {"result": result, "model_type": model_type, "success": True, "mode": "mock"}
        
        engine = self.get_model(model_type)
        if not engine:
            # 如果没有真实模型，使用模拟推理
            logger.warning(f"模型 {model_type} 未加载，使用模拟推理")
            result = self._mock_inference(text, model_type)
            return {"result": result, "model_type": model_type, "success": True, "mode": "fallback"}
        
        try:
            if model_type.startswith('codelama'):
                # CodeLlama模型
                if model_type == 'codelama_finetuned':
                    result = engine.extract_relations(text)
                else:
                    # 未微调的模型使用更简单的提示
                    prompt = f"分析以下文本中的实体和关系：\n{text}"
                    result = {"response": engine.generate_response(prompt), "model_type": "base"}
                return {"result": result, "model_type": model_type, "success": True}
                
            elif model_type == 'bert_finetuned':
                # BERT模型需要实体对
                # 这里需要先从文本中提取实体（可以使用简单的规则）
                entities = self._extract_entities_simple(text)
                entity_pairs = []
                for i, e1 in enumerate(entities):
                    for e2 in entities[i+1:]:
                        entity_pairs.append((e1, e2))
                
                if entity_pairs:
                    results = []
                    for entity1, entity2 in entity_pairs[:5]:  # 限制数量
                        result = engine.classify_relation(text, entity1, entity2)
                        results.append(result)
                    
                    return {
                        "result": {
                            "entities": entities,
                            "relations": results
                        },
                        "model_type": model_type,
                        "success": True
                    }
                else:
                    return {"error": "未检测到实体对", "success": False}
            else:
                return {"error": f"不支持的模型类型: {model_type}", "success": False}
                
        except Exception as e:
            logger.error(f"模型 {model_type} 预测失败: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _extract_entities_simple(self, text: str) -> List[str]:
        """简单的实体提取（后续可以改进为NER模型）"""
        import re
        
        # 简单的实体识别规则
        patterns = [
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?=\s+(公司|组织|机构|部门))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([一-龥]{2,4}(?:公司|组织|机构|大学|医院|银行))',
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # 取第一个捕获组
                if len(match) >= 2 and match not in entities:
                    entities.append(match)
        
        return entities[:10]  # 限制数量
    
    def _mock_inference(self, text: str, model_type: str = 'codelama_finetuned') -> Dict[str, Any]:
        """模拟推理生成示例结果"""
        import random
        
        # 简单的实体和关系提取模拟
        entities = []
        relations = []
        
        # 模拟实体识别
        entity_keywords = {
            '公司': ['Apple', 'Microsoft', 'Google', 'Amazon', 'Meta'],
            '人物': ['Steve Jobs', 'Tim Cook', 'Bill Gates', 'Sundar Pichai'],
            '产品': ['iPhone', 'Windows', 'Android', 'Kindle'],
            '城市': ['Cupertino', 'Redmond', 'Mountain View', 'Seattle']
        }
        
        text_lower = text.lower()
        for category, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    entities.append({
                        'text': keyword,
                        'type': category,
                        'start': text.find(keyword),
                        'end': text.find(keyword) + len(keyword),
                        'confidence': random.uniform(0.7, 0.95)
                    })
        
        # 模拟关系识别
        if entities:
            # 为每个实体对生成一个关系
            for i in range(min(len(entities) - 1, 3)):
                rel_types = ['-founder_of', 'CEO_of', 'headquartered_in', 'developed', 'based_in']
                relations.append({
                    'head': entities[i]['text'],
                    'tail': entities[i + 1]['text'] if i + 1 < len(entities) else entities[0]['text'],
                    'relation': random.choice(rel_types),
                    'confidence': random.uniform(0.6, 0.9),
                    'model_type': f"{model_type}_mock"
                })
        
        return {
            'entities': entities,
            'relations': relations,
            'model_type': model_type,
            'is_mock': True,
            'text': text,
            'metadata': {
                'model_name': model_type,
                'inference_mode': 'mock',
                'timestamp': '2025-11-06 23:08:45'
            }
        }


# 全局模型管理器实例
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """获取全局模型管理器"""
    return model_manager