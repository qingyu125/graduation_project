from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List

class EvaluationMetrics:
    @staticmethod
    def entity_f1(pred_entities: Dict, true_entities: Dict):
        """实体抽取F1值"""
        pred_ids = set(pred_entities.keys())
        true_ids = set(true_entities.keys())
        # 实体识别F1
        overlap = pred_ids & true_ids
        if not overlap and not pred_ids and not true_ids:
            return 1.0
        precision = len(overlap) / len(pred_ids) if pred_ids else 0
        recall = len(overlap) / len(true_ids) if true_ids else 0
        if precision + recall == 0:
            return 0.0
        # 实体类型F1（在识别正确的实体上）
        type_correct = 0
        for ent in overlap:
            if pred_entities[ent] == true_entities[ent]:
                type_correct += 1
        type_precision = type_correct / len(pred_ids) if pred_ids else 0
        type_recall = type_correct / len(true_ids) if true_ids else 0
        
        # 综合F1（识别F1和类型F1的平均）
        f1_identify = 2 * precision * recall / (precision + recall)
        f1_type = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0
        return (f1_identify + f1_type) / 2
    
    @staticmethod
    def relation_f1(pred_relations: List, true_relations: List):
        """关系抽取F1值"""
        # 转换为(head, tail, type)元组便于比较
        pred_tuples = {(r["head"], r["tail"], r["type"]) for r in pred_relations}
        true_tuples = {(r["head"], r["tail"], r["type"]) for r in true_relations}
        
        overlap = pred_tuples & true_tuples
        precision = len(overlap) / len(pred_tuples) if pred_tuples else 0
        recall = len(overlap) / len(true_tuples) if true_tuples else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def reasoning_accuracy(pred_conclusion: bool, true_label: bool):
        """推理准确率"""
        return 1.0 if pred_conclusion == true_label else 0.0