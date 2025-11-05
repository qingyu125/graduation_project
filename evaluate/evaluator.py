import json
from evaluate.metrics import EvaluationMetrics
from models.extraction.text_to_code import TextToCodeConverter
from models.extraction.code_parser import CodeParser
from models.knowledge.kg_injector import KnowledgeGraphInjector
from models.reasoning.code_chain import CodeChainReasoner

class Evaluator:
    def __init__(self, model_wrapper, kg_path="./data/kg/legal_kg.json"):
        self.model_wrapper = model_wrapper
        self.text_to_code = TextToCodeConverter(model_wrapper)
        self.code_parser = CodeParser()
        self.kg_injector = KnowledgeGraphInjector(kg_path)
        self.reasoner = CodeChainReasoner(model_wrapper)
        self.metrics = EvaluationMetrics()
        
        # 评估结果存储
        self.results = {
            "entity_f1": [],
            "relation_f1": [],
            "reasoning_acc": []
        }
    
    def evaluate_sample(self, sample):
        """评估单条样本"""
        # 1. 文本→伪代码
        pseudo_code = self.text_to_code.convert(sample["text"])
        
        # 2. 解析要素
        pred_elements = self.code_parser.parse(pseudo_code)
        
        # 3. 知识验证
        pred_elements = self.kg_injector.verify_elements(pred_elements)
        
        # 4. 推理
        if sample["relations"]:
            rel = sample["relations"][0]
            reasoning_result = self.reasoner.reason(
                pred_elements,
                rel["head"],
                rel["tail"],
                rel["type"]
            )
        else:
            reasoning_result = {"conclusion": False}
        
        # 5. 计算指标
        entity_f1 = self.metrics.entity_f1(
            pred_elements["entities"],
            sample["elements"]["entities"]
        )
        relation_f1 = self.metrics.relation_f1(
            pred_elements["relations"],
            sample["elements"]["relations"]
        )
        reasoning_acc = self.metrics.reasoning_accuracy(
            reasoning_result["conclusion"],
            sample["inference_label"]
        )
        
        return {
            "entity_f1": entity_f1,
            "relation_f1": relation_f1,
            "reasoning_acc": reasoning_acc,
            "pred_elements": pred_elements,
            "true_elements": sample["elements"]
        }
    
    def evaluate_dataset(self, data_path):
        """评估整个数据集"""
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        for sample in dataset:
            res = self.evaluate_sample(sample)
            self.results["entity_f1"].append(res["entity_f1"])
            self.results["relation_f1"].append(res["relation_f1"])
            self.results["reasoning_acc"].append(res["reasoning_acc"])
        
        # 计算平均指标
        return {
            "avg_entity_f1": sum(self.results["entity_f1"]) / len(self.results["entity_f1"]),
            "avg_relation_f1": sum(self.results["relation_f1"]) / len(self.results["relation_f1"]),
            "avg_reasoning_acc": sum(self.results["reasoning_acc"]) / len(self.results["reasoning_acc"])
        }