from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import yaml

class CodeLlamaWrapper:
    def __init__(self, config_path="./config/model_config.yaml", lora_path=None):
        self.config = yaml.safe_load(open(config_path))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["name"],
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config["model"]["quantize"]["load_in_4bit"],
            bnb_4bit_use_double_quant=self.config["model"]["quantize"]["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=self.config["model"]["quantize"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, self.config["model"]["quantize"]["bnb_4bit_compute_dtype"])
        )
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重（若有）
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()  # 合并权重用于推理
    
    def generate(self, prompt, max_new_tokens=None):
        """生成文本/代码"""
        max_tokens = max_new_tokens or self.config["generation"]["max_new_tokens"]
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=self.config["generation"]["temperature"],
            top_p=self.config["generation"]["top_p"],
            do_sample=self.config["generation"]["do_sample"],
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)