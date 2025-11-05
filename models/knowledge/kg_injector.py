import json
from typing import Dict, List

class KnowledgeGraphInjector:
    def __init__(self, kg_path="./data/kg/legal_kg.json"):
        self.kg = self._load_kg(kg_path)
        # 关系-事件支撑映射（领域知识规则）
        self.relation_event_support = {
            "合作关系": ["签约", "合作"],
            "雇佣关系": ["任职", "加入"],
            "合同纠纷": ["违约", "解约"]
        }
    
    def _load_kg(self, kg_path):
        """加载知识图谱（三元组列表）"""
        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)
        # 构建头实体→(关系→尾实体列表)映射
        kg_index = {}
        for triple in kg_data:
            head, rel, tail = triple["head"], triple["relation"], triple["tail"]
            if head not in kg_index:
                kg_index[head] = {}
            if rel not in kg_index[head]:
                kg_index[head][rel] = []
            kg_index[head][rel].append(tail)
        return kg_index
    
    def verify_elements(self, elements: Dict):
        """验证要素与知识图谱的一致性"""
        verified_relations = []
        for rel in elements["relations"]:
            head, tail, rel_type = rel["head"], rel["tail"], rel["type"]
            
            # 1. 知识图谱存在性验证
            kg_verify = False
            if head in self.kg and rel_type in self.kg[head]:
                kg_verify = tail in self.kg[head][rel_type]
            
            # 2. 事件支撑验证
            event_support = False
            required_events = self.relation_event_support.get(rel_type, [])
            if required_events:
                event_support = any(
                    e["trigger"] in required_events 
                    for e in elements["events"]
                )
            
            rel["verified"] = kg_verify and event_support
            rel["kg_verify"] = kg_verify
            rel["event_support"] = event_support
            verified_relations.append(rel)
        
        elements["relations"] = verified_relations
        return elements