import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CodeEnhancedDataset(Dataset):
    def __init__(self, data_path, tokenizer: AutoTokenizer, max_length=512):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = self._build_prompt_template()
    
    def _load_data(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _build_prompt_template(self):
        return """# 任务：根据文本生成包含实体、事件、关系的伪代码
# 文本：{text}
# 伪代码：
{label}"""
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # 构建输入-输出对（文本→伪代码）
        prompt = self.prompt_template.format(
            text=sample["text"],
            label=sample["pseudo_code"]
        )
        
        # 编码
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 标签与输入相同（自回归训练）
        labels = encoding["input_ids"].clone()
        # 掩码提示部分（仅计算伪代码部分损失）
        prompt_len = len(self.tokenizer(sample["text"], return_tensors="pt")["input_ids"][0]) + 30
        labels[0, :prompt_len] = -100  # -100在PyTorch中会被忽略
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }