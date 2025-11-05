from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from models.base.code_llama_wrapper import CodeLlamaWrapper
from train.dataset import CodeEnhancedDataset
import yaml

class LoRATrainer:
    def __init__(self, config_path="./config/model_config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        # 加载未微调的基础模型
        self.base_model_wrapper = CodeLlamaWrapper(config_path)
        self.model = self.base_model_wrapper.model
        self.tokenizer = self.base_model_wrapper.tokenizer
        # 配置LoRA
        self.model = self._configure_lora()
    
    def _configure_lora(self):
        """配置LoRA参数"""
        lora_config = LoraConfig(
            r=self.config["lora"]["rank"],
            lora_alpha=self.config["lora"]["alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["dropout"],
            bias=self.config["lora"]["bias"],
            task_type=self.config["lora"]["task_type"],
        )
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数比例
        return model
    
    def train(self, train_data_path, val_data_path, output_dir):
        """训练主函数"""
        # 构建数据集
        train_dataset = CodeEnhancedDataset(
            train_data_path, 
            self.tokenizer,
            max_length=512
        )
        val_dataset = CodeEnhancedDataset(
            val_data_path,
            self.tokenizer,
            max_length=512
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            fp16=self.config["training"]["fp16"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            evaluation_strategy=self.config["training"]["evaluation_strategy"],
            eval_steps=self.config["training"]["eval_steps"],
            load_best_model_at_end=self.config["training"]["load_best_model_at_end"],
            report_to="none"
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 自回归模型不使用掩码语言建模
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # 开始训练
        trainer.train()
        # 保存LoRA权重
        self.model.save_pretrained(f"{output_dir}/lora_weights")