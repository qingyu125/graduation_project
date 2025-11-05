## 简介
  本仓库为本科毕设代码实现，题目：《面向高效推理的要素抽取与应用算法设计与实现》  
## 整体代码结构  
```
code-enhanced-text-reasoning/  
├── config/                  # 配置文件目录  
│   ├── model_config.yaml    # 模型参数配置（量化精度、LoRA参数等）  
│   ├── data_config.yaml     # 数据路径、划分比例等配置  
│   └── train_config.yaml    # 训练参数（学习率、批次大小等）  
├── data/                    # 数据存储目录  
│   ├── raw/                 # 原始数据集  
│   ├── processed/           # 预处理后数据集  
│   └── pseudo_samples/      # 伪标注样本  
├── src/                     # 核心代码目录  
│   ├── data_processing/     # 数据获取与预处理模块  
│   ├── model/               # 模型封装与微调模块  
│   ├── modules/             # 核心功能模块（要素抽取、知识融合等）  
│   ├── evaluation/          # 评估模块  
│   └── application/         # 应用系统开发模块  
├── logs/                    # 训练日志目录  
├── weights/                 # 模型权重存储目录  
├── docs/                    # 项目文档目录（含实验报告、使用说明）  
├── requirements.txt         # 项目依赖库列表  
└── main.py                  # 项目主运行脚本  
```


