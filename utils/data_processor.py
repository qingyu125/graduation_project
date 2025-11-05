import json
import random
import yaml
import logging
from sklearn.model_selection import train_test_split

# 初始化日志（便于调试）
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataProcessor:
    def __init__(self, config_path="./config/data_config.yaml"):
        self.config = yaml.safe_load(open(config_path,encoding="utf-8"))
        # 通用实体类型映射（DocRED类型→目标格式类型，保持通用）
        self.entity_type_mapping = {
            "ORG": "ORG",       # 机构
            "PER": "PER",       # 人物
            "LOC": "LOC",       # 地点
            "TIME": "TIME",     # 时间
            "NUM": "NUM",       # 数字
            "MISC": "MISC",     # 其他
            "DATE": "TIME"      # DocRED中的DATE统一为TIME
        }
        # 通用关系类型映射（DocRED的Wikidata关系ID→自然语言关系名，覆盖常见类型）
        self.relation_id_mapping = {
            "P159": "headquartered in",       # 总部位于
            "P17": "located in country",      # 位于国家
            "P131": "located in administrative region",     # 位于行政区
            "P150": "contains administrative territorial entity",     # 包含行政区
            "P178": "founder",         # 创始人
            "P452": "industry",        # 所属行业
            "P31": "belongs to category",        # 属于类别
            "P127": "owned by",        # 被拥有
            "P571": "establishment time",        # 成立时间
            "P576": "dissolution time",        # 解散时间
            "P267": "regulated by",        # 监管机构
            "P169": "chief executive officer",        # 首席执行官
            "P361": "part of",        # 部分组成
            "P190": "sister city",        # 姐妹城市
            "P47": "shares border with",        # 相邻国家
            # 可根据DocRED实际关系ID扩展
        }

    def _docred_to_text(self, docred_sample):
        """将DocRED的sents（token列表）转换为连贯文本（通用领域适配）"""
        text_parts = []
        for sent_tokens in docred_sample["sents"]:
            # 过滤无效token（空字符串、特殊符号）
            valid_tokens = [
                token for token in sent_tokens 
                if token.strip() and not (token.startswith("##") or token in [",", ".", "(", ")", "-", ":"])
            ]
            if not valid_tokens:
                continue
            # 中英文文本拼接逻辑（含中文字符直接拼接，否则用空格）
            is_chinese = any('\u4e00' <= char <= '\u9fff' for char in ''.join(valid_tokens))
            if is_chinese:
                sent_text = ''.join(valid_tokens)
            else:
                sent_text = ' '.join(valid_tokens)
            text_parts.append(sent_text)
        # 句子间用句号分隔，确保文本连贯
        full_text = '。'.join(text_parts) + '。'
        # 截断过长文本（避免模型输入超限）
        if len(full_text) > 1024:
            full_text = full_text[:1024] + "..."
        return full_text.strip()

    def _extract_entities(self, docred_sample):
        """从vertexSet提取实体（处理同指实体，通用领域适配）"""
        entities = {}
        sents_count = len(docred_sample["sents"])  # 句子总数（防止sent_id越界）
        
        for entity_group_idx, entity_group in enumerate(docred_sample["vertexSet"]):
            # 遍历同指实体组，取第一个有效实体（避免重复）
            valid_entity = None
            for entity in entity_group:
                # 检查sent_id有效性（防止越界）
                if 0 <= entity["sent_id"] < sents_count:
                    valid_entity = entity
                    break
            if not valid_entity:
                logging.warning(f"Entity group {entity_group_idx} has no valid entities, skipping")
                continue
            
            # 处理实体名（去重、过滤空值）
            entity_name = valid_entity["name"].strip()
            if not entity_name or len(entity_name) < 2:  # 过滤过短实体名（如单个字符）
                continue
            
            # 转换实体类型（统一格式，未知类型保留原类型）
            docred_entity_type = valid_entity["type"].upper()  # 统一大写（如"org"→"ORG"）
            target_entity_type = self.entity_type_mapping.get(
                docred_entity_type, docred_entity_type
            )
            
            # 去重：同实体名只保留一次（取第一个类型）
            if entity_name not in entities:
                entities[entity_name] = target_entity_type
        
        return entities

    def _extract_relations(self, docred_sample):
        """从labels提取关系（映射Wikidata ID到自然语言，通用领域适配）"""
        relations = []
        vertex_count = len(docred_sample["vertexSet"])  # 实体组总数（防止索引越界）
        entity_groups = docred_sample["vertexSet"]
        
        for rel_idx, rel in enumerate(docred_sample.get("labels", [])):
            # 检查关系核心字段
            required_rel_fields = ["h", "t", "r"]
            if not all(field in rel for field in required_rel_fields):
                logging.warning(f"Relation {rel_idx} missing core fields (h/t/r), skipping")
                continue
            
            # 检查头实体/尾实体索引有效性
            head_group_idx = rel["h"]
            tail_group_idx = rel["t"]
            if (head_group_idx < 0 or head_group_idx >= vertex_count or
                tail_group_idx < 0 or tail_group_idx >= vertex_count):
                logging.warning(
                    f"Relation {rel_idx} has invalid indices: head={head_group_idx}, tail={tail_group_idx} (total entity groups={vertex_count})"
                )
                continue
            
            # 获取头实体/尾实体名称（取同指组第一个有效实体）
            head_entity_group = entity_groups[head_group_idx]
            tail_entity_group = entity_groups[tail_group_idx]
            if not head_entity_group or not tail_entity_group:
                logging.warning(f"Head/tail entity group of relation {rel_idx} is empty, skipping")
                continue
            
            # 提取实体名（去重、过滤）
            head_entity_name = head_entity_group[0]["name"].strip()
            tail_entity_name = tail_entity_group[0]["name"].strip()
            if not head_entity_name or not tail_entity_name:
                logging.warning(f"Head/tail entity name of relation {rel_idx} is empty, skipping")
                continue
            
            # 映射关系ID到自然语言（未知ID保留原ID）
            rel_id = rel["r"]
            rel_name = self.relation_id_mapping.get(rel_id, rel_id)
            
            # 提取证据句信息（用于事件生成）
            evidence_sent_ids = rel.get("evidence", [])
            evidence_sent_text = ""
            if evidence_sent_ids and 0 <= evidence_sent_ids[0] < len(docred_sample["sents"]):
                sent_tokens = docred_sample["sents"][evidence_sent_ids[0]]
                evidence_sent_text = ' '.join([t.strip() for t in sent_tokens if t.strip()])[:50]  # 截断
            
            # 组装关系（含证据信息，便于后续推理）
            relations.append({
                "head": head_entity_name,
                "tail": tail_entity_name,
                "type": rel_name,
                "rel_id": rel_id,  # 保留原始关系ID
                "evidence": evidence_sent_text
            })
        
        return relations

    def _extract_events(self, docred_sample, relations):
        """从关系和文本中提取事件（通用领域适配，基于关系类型生成触发词）"""
        events = []
        if not relations:
            return events  # 无关系时返回空事件
        
        # 通用事件触发词映射（基于关系类型）
        rel_to_trigger = {
            "headquartered in": "establish headquarters",  # 总部位于
            "located in country": "be located in",  # 位于国家
            "located in administrative region": "be located in",  # 位于行政区
            "contains administrative territorial entity": "contain",  # 包含行政区
            "founder": "found",  # 创始人
            "industry": "belong to",  # 所属行业
            "belongs to category": "belong to",  # 属于类别
            "owned by": "be owned by",  # 被拥有
            "establishment time": "be established",  # 成立时间
            "dissolution time": "be dissolved",  # 解散时间
            "regulated by": "be regulated by",  # 监管机构
            "chief executive officer": "serve as CEO",  # 首席执行官
            "part of": "be part of",  # 部分组成
            "sister city": "become sister cities with",  # 姐妹城市
            "shares border with": "be adjacent to"  # 相邻国家
        }
        
        for rel in relations:
            # 生成事件触发词（未知关系用“关联”）
            trigger = rel_to_trigger.get(rel["type"], "be related to")
            # 提取事件时间（从实体中匹配时间类型实体）
            entities = self._extract_entities(docred_sample)
            event_time = ""
            for ent_name, ent_type in entities.items():
                if ent_type == "TIME" and ent_name not in [rel["head"], rel["tail"]]:
                    event_time = ent_name
                    break
            
            # 组装事件（通用格式，含关系证据）
            events.append({
                "trigger": trigger,
                "time": event_time,
                "participants": [rel["head"], rel["tail"]],  # 事件参与方（头实体+尾实体）
                "evidence": rel["evidence"],                # 事件证据句
                "related_relation": rel["type"]             # 关联的关系类型
            })
        
        return events

    def _generate_pseudo_code(self, entities, events, relations):
        """生成通用领域伪代码（确保可被code_parser解析）"""
        # 处理实体格式（确保JSON兼容）
        entities_str = json.dumps(entities, ensure_ascii=False)
        # 处理事件格式
        events_str = json.dumps(events, ensure_ascii=False)
        # 处理关系格式（仅保留核心字段）
        relations_simplified = [
            {"head": r["head"], "tail": r["tail"], "type": r["type"]} 
            for r in relations
        ]
        relations_str = json.dumps(relations_simplified, ensure_ascii=False)
        
        # 生成伪代码（变量名严格为entities/events/relations）
        pseudo_code = f"""entities = {entities_str}
        /events = {events_str}/relations = {relations_str}"""
        return pseudo_code

    def convert_docred_to_target(self, docred_sample):
        """将单个DocRED样本转换为目标格式（通用领域，含完整错误处理）"""
        try:
            # 1. 检查DocRED核心字段
            required_docred_fields = ["sents", "vertexSet", "labels", "title"]
            for field in required_docred_fields:
                if field not in docred_sample:
                    logging.warning(f"Sample {docred_sample.get('title', 'unknown title')} missing field: {field}")
                    return None
            
            # 2. 转换文本（过滤过短文本）
            text = self._docred_to_text(docred_sample)
            if len(text) < 50:  # 过滤过短文本（避免无意义样本）
                logging.warning(f"Sample {docred_sample['title']} text too short ({len(text)} characters): {text}")
                return None
            
            # 3. 提取实体（至少2个实体才有效）
            entities = self._extract_entities(docred_sample)
            if len(entities) < 2:
                logging.warning(f"Sample {docred_sample['title']} has insufficient entities (need ≥2): {len(entities)}")
                return None
            
            # 4. 提取关系（至少1个关系才有效）
            relations = self._extract_relations(docred_sample)
            if len(relations) < 1:
                logging.warning(f"Sample {docred_sample['title']} has no valid relations, skipping")
                return None
            
            # 5. 提取事件（允许空事件，但优先生成）
            events = self._extract_events(docred_sample, relations)
            
            # 6. 生成伪代码（检查格式有效性）
            # 【新增】定义简化关系（用于elements和伪代码）
            relations_simplified = [
                {"head": r["head"], "tail": r["tail"], "type": r["type"]} 
                for r in relations
            ]
            pseudo_code = self._generate_pseudo_code(entities, events, relations_simplified)  # 传入简化关系
            if any(keyword in pseudo_code for keyword in ["entities = {}", "relations = []"]):
                logging.warning(f"Sample {docred_sample['title']} pseudo code abnormal: {pseudo_code[:100]}...")
                return None
            
            # 7. 构建推理目标（取第一个关系作为推理目标，符合通用场景）
            first_rel = relations[0]
            inference_target = f"Determine whether there is a {first_rel['type']} relationship between {first_rel['head']} and {first_rel['tail']}"
            inference_label = True  # DocRED的labels均为真实存在的关系
            
            # 8. 组装目标格式（含样本元信息，便于后续分析）
            return {
                "id": f"docred_{hash(docred_sample['title'])}",  # 基于标题生成唯一ID
                "title": docred_sample["title"],                 # 保留文档标题
                "text": text,
                "pseudo_code": pseudo_code,
                "elements": {
                    "entities": entities,
                    "events": events,
                    "relations": relations_simplified  # 此处已定义，修复NameError
                },
                "inference_target": inference_target,
                "inference_label": inference_label,
                "source": "DocRED_train_annotated"  # 标记数据来源
            }
        
        except Exception as e:
            # 捕获所有异常，避免批量转换中断
            sample_title = docred_sample.get("title", "unknown title")
            logging.error(f"Sample {sample_title} conversion failed: {str(e)}", exc_info=True)
            return None
        
        except Exception as e:
            # 捕获所有异常，避免批量转换中断
            sample_title = docred_sample.get("title", "unknown title")
            logging.error(f"Sample {sample_title} conversion failed: {str(e)}", exc_info=True)
            return None

    def generate_general_pseudo_samples(self, num_samples=None):
        """生成通用领域伪样本（替代原法律伪样本，适配DocRED通用场景）"""
        num = num_samples or self.config["data_process"]["pseudo_sample_num"]
        # 通用领域实体库（覆盖DocRED常见类型）
        orgs = ["Apple Inc.", "Microsoft Corporation", "Tesla Motors", "Amazon.com", "Google LLC", 
                "Toyota Motor Corporation", "Coca-Cola Company", "Samsung Electronics", "IBM", "Facebook"]
        pers = ["Steve Jobs", "Bill Gates", "Elon Musk", "Jeff Bezos", "Larry Page", 
                "Warren Buffett", "Mark Zuckerberg", "Tim Cook", "Sundar Pichai", "Satya Nadella"]
        locs = ["United States", "China", "Japan", "Germany", "United Kingdom", 
                "California", "Tokyo", "Beijing", "New York", "London"]
        times = ["1976年", "1975年", "2003年", "1994年", "1998年", 
                 "2004年", "2012年", "1968年", "1981年", "2015年"]
        # 通用领域关系库（匹配DocRED常见关系）
        relations = [
            {"type": "headquartered in", "trigger": "establish headquarters"},  # 总部位于
            {"type": "founder", "trigger": "found"},  # 创始人
            {"type": "industry", "trigger": "belong to"},  # 所属行业
            {"type": "establishment time", "trigger": "be established"},  # 成立时间
            {"type": "chief executive officer", "trigger": "serve as CEO"},  # 首席执行官
            {"type": "located in country", "trigger": "be located in"},  # 位于国家
            {"type": "owned by", "trigger": "be owned by"},  # 被拥有
            {"type": "belongs to category", "trigger": "belong to"}  # 属于类别
        ]
        
        pseudo_samples = []
        for i in range(num):
            # 随机组合实体与关系
            ent1_type = random.choice(["ORG", "PER", "LOC"])
            ent2_type = random.choice(["LOC", "TIME", "ORG"])
            if ent1_type == "ORG":
                ent1 = random.choice(orgs)
            elif ent1_type == "PER":
                ent1 = random.choice(pers)
            else:
                ent1 = random.choice(locs)
            
            if ent2_type == "LOC":
                ent2 = random.choice(locs)
            elif ent2_type == "TIME":
                ent2 = random.choice(times)
            else:
                ent2 = random.choice(orgs)
            
            rel = random.choice(relations)
            time = random.choice(times) if ent2_type != "TIME" else ent2
            
            # 构建文本（通用领域描述）
            if rel["type"] == "headquartered in":
                text = f"The headquarters of {ent1} is located in {ent2}, founded in {time}, it is a well-known enterprise."
            elif rel["type"] == "founder":
                text = f"{ent1} is the founder of {ent2}, who founded the company in {time}."
            elif rel["type"] == "establishment time":
                text = f"{ent1} was established in {ent2}, with its headquarters located in {random.choice(locs)}."
            else:
                text = f"{ent1} has a {rel['type']} relationship with {ent2}, and related events occurred in {time}."
            
            # 构建实体、事件、关系
            entities = {ent1: ent1_type, ent2: ent2_type, time: "TIME"}
            events = [{
                "trigger": rel["trigger"],
                "time": time,
                "participants": [ent1, ent2],
                "evidence": text[:30],
                "related_relation": rel["type"]
            }]
            relations_list = [{"head": ent1, "tail": ent2, "type": rel["type"], "rel_id": "P159"}]
            
            # 生成伪代码
            pseudo_code = self._generate_pseudo_code(entities, events, relations_list)
            
            # 组装伪样本
            pseudo_samples.append({
                "id": f"pseudo_{i}",
                "title": f"Pseudo Sample {i}: {ent1} - {ent2}",
                "text": text,
                "pseudo_code": pseudo_code,
                "elements": {
                    "entities": entities,
                    "events": events,
                    "relations": [{"head": ent1, "tail": ent2, "type": rel["type"]}]
                },
                "inference_target": f"Determine whether there is a {rel['type']} relationship between {ent1} and {ent2}",
                "inference_label": True,
                "source": "General_Pseudo"
            })
        
        # 保存通用伪样本
        pseudo_save_path = self.config["paths"]["pseudo_data"]
        with open(pseudo_save_path, "w", encoding="utf-8") as f:
            json.dump(pseudo_samples, f, ensure_ascii=False, indent=2)
        logging.info(f"General pseudo samples generated: {len(pseudo_samples)} items, saved to {pseudo_save_path}")
        return pseudo_samples

    def process_docred(self, docred_path):
        """批量转换DocRED数据集（通用领域，支持JSON Lines格式）"""
        try:
            # 1. 加载DocRED数据（支持单行JSON或JSON数组）
            docred_data = []
            with open(docred_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                f.seek(0)  # 重置文件指针
                # 判断是否为JSON Lines格式（每行一个样本）
                if first_line.startswith("{") and first_line.endswith("}"):
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sample = json.loads(line)
                            docred_data.append(sample)
                        except json.JSONDecodeError as e:
                            logging.error(f"DocRED line {line_num} parsing error: {str(e)}")
                            continue
                else:
                    # 标准JSON数组格式
                    docred_data = json.load(f)
            
            if not docred_data:
                logging.error(f"No DocRED samples loaded (path: {docred_path})")
                return []
            
            # 2. 批量转换（限制转换数量，避免内存溢出）
            max_convert_num = self.config["data_process"].get("max_docred_convert", 2000)
            target_data = []
            for sample in docred_data[:max_convert_num]:
                converted_sample = self.convert_docred_to_target(sample)
                if converted_sample:
                    target_data.append(converted_sample)
            
            # 3. 输出转换统计
            total_processed = min(len(docred_data), max_convert_num)
            valid_rate = len(target_data) / total_processed * 100 if total_processed > 0 else 0
            logging.info(
                f"DocRED conversion completed: {len(target_data)}/{total_processed} valid samples (valid rate: {valid_rate:.1f}%)"
            )
            return target_data
        
        except Exception as e:
            logging.error(f"DocRED batch processing failed: {str(e)}", exc_info=True)
            return []

    def process_and_split(self):
        """合并DocRED转换数据与通用伪样本，划分训练/验证/测试集"""
        # 1. 处理DocRED原始数据
        docred_target_data = self.process_docred(self.config["paths"]["raw_docred"])
        # 2. 生成通用伪样本（若未生成）
        try:
            with open(self.config["paths"]["pseudo_data"], "r", encoding="utf-8") as f:
                pseudo_data = json.load(f)
            logging.info(f"Loaded generated general pseudo samples: {len(pseudo_data)} items")
        except FileNotFoundError:
            logging.info("No pseudo samples found, starting generation...")
            pseudo_data = self.generate_general_pseudo_samples()
        
        # 3. 合并数据（DocRED样本+通用伪样本）
        full_data = docred_target_data + pseudo_data
        if not full_data:
            raise ValueError("No valid data to merge, please check DocRED conversion and pseudo sample generation process")
        random.shuffle(full_data)  # 打乱数据
        
        # 4. 划分数据集（按配置比例）
        train_ratio, val_ratio, test_ratio = self.config["data_process"]["train_val_test_split"]
        train_val, test = train_test_split(
            full_data,
            test_size=test_ratio,
            random_state=42,
            stratify=[sample["source"] for sample in full_data]  # 按数据来源分层，确保分布均匀
        )
        train, val = train_test_split(
            train_val,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=42,
            stratify=[sample["source"] for sample in train_val]
        )
        
        # 5. 保存处理后的数据
        processed_paths = [
            (train, self.config["paths"]["processed_train"]),
            (val, self.config["paths"]["processed_val"]),
            (test, self.config["paths"]["processed_test"])
        ]
        for data, path in processed_paths:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"{path.split('/')[-1]} saved: {len(data)} samples")
        
        # 6. 输出数据统计
        logging.info(
            f"Data preprocessing completed:\n"
            f"- Training set: {len(train)} items (DocRED: {sum(1 for s in train if s['source']=='DocRED_train_annotated')} items, Pseudo samples: {sum(1 for s in train if s['source']=='General_Pseudo')} items)\n"
            f"- Validation set: {len(val)} items (DocRED: {sum(1 for s in val if s['source']=='DocRED_train_annotated')} items, Pseudo samples: {sum(1 for s in val if s['source']=='General_Pseudo')} items)\n"
            f"- Test set: {len(test)} items (DocRED: {sum(1 for s in test if s['source']=='DocRED_train_annotated')} items, Pseudo samples: {sum(1 for s in test if s['source']=='General_Pseudo')} items)"
        )
        return train, val, test