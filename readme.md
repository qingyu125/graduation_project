## 简介
  本仓库为本科毕设代码实现，题目：《面向高效推理的要素抽取与应用算法设计与实现》  
## 整体代码结构  
```
text_reasoning/  
├── config/                  # 配置文件目录  
│   ├── model_config.yaml    # 模型参数（量化方式、LoRA配置等）  
│   └── data_config.yaml     # 数据路径、划分比例等  
├── data/                    # 数据目录  
│   ├── raw/                 # 原始数据集（DocRED、伪样本）  
│   ├── processed/           # 预处理后的数据（文本+代码要素）  
│   └── kg/                  # 领域知识图谱（CSV/JSON格式）  
├── models/                  # 模型模块  
│   ├── base/                # 基础模型封装   
│   │   └── code_llama_wrapper.py  # CodeLlama-7B加载与基础调用  
│   ├── extraction/          # 要素抽取模块  
│   │   ├── text_to_code.py  # 文本→伪代码转换（基于Code Prompting）  
│   │   └── code_parser.py   # 伪代码解析为结构化要素  
│   ├── knowledge/           # 知识融合模块  
│   │   └── kg_injector.py   # 知识图谱与要素的验证融合  
│   └── reasoning/           # 推理模块  
│       └── code_chain.py    # 代码链推理与验证（基于CoC）  
├── train/                   # 训练相关模块  
│   ├── lora_trainer.py      # LoRA微调逻辑（基于PEFT）  
│   ├── dataset.py           # 训练数据集构建  
│   └── loss.py              # 自定义损失函数（含逻辑一致性正则）  
├── evaluate/                # 评估模块  
│   ├── metrics.py           # 要素抽取F1、推理准确率等指标  
│   └── evaluator.py         # 模型评估流程  
├── applications/            # 场景应用  
│   └── legal_relation_extraction.py  # 法律文档关系抽取系统  
├── utils/                   # 工具函数  
│   ├── code_safety.py       # 伪代码安全解析（AST替代exec）  
│   └── data_processor.py    # 数据清洗与格式转换  
  
└── main.py                  # 主流程调度（预处理→训练→评估→应用）  
```
