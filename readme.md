## 简介
  本仓库为本科毕设代码实现，题目：《面向高效推理的要素抽取与应用算法设计与实现》  
## 整体代码结构  
```
code-enhanced-text-reasoning/  
├── README.md                           # 项目主文档  
├── CODE_FRAMEWORK_SUMMARY.md           # 代码框架总结  
├── QUICK_START.md                      # 快速开始指南  
└── code/                               # 核心代码目录  
    ├── requirements.txt                # 依赖包列表  
    ├── data_processor/                 # DocRED数据处理模块  
    │   ├── data_processor.py           # 数据处理主程序  
    │   ├── test_data_processor.py      # 单元测试  
    │   ├── example_usage.py            # 使用示例  
    │   └── README_DocRED.md            # 模块文档  
    ├── text_to_pseudocode/             # 文本到伪代码转换模块  
    │   ├── text_to_pseudocode.py       # 转换主程序  
    │   ├── test_converter.py           # 单元测试  
    │   └── TEXT_TO_PSEUDOCODE_README.md # 模块文档  
    ├── pseudocode_parser/              # 伪代码解析与要素抽取模块  
    │   ├── pseudocode_parser.py        # 解析主程序  
    │   └── test_parser.py              # 单元测试  
    ├── knowledge_fusion/               # 知识融合与推理验证模块  
    │   ├── knowledge_fusion.py         # 融合主程序  
    │   ├── test_knowledge_fusion.py    # 单元测试  
    │   ├── demo_knowledge_fusion.py    # 演示程序  
    │   └── knowledge_fusion_README.md  # 模块文档  
    ├── model_training/                 # 模型训练与基线对比模块  
    │   ├── model_training.py           # 训练主程序  
    │   ├── train_pipeline.py           # 完整训练流程  
    │   ├── test_framework.py           # 测试框架  
    │   ├── usage_examples.py           # 使用示例  
    │   └── PROJECT_SUMMARY.md          # 项目总结  
    ├── evaluation/                     # 测试评估模块  
    │   ├── evaluation_framework.py     # 评估框架核心  
    │   ├── test_evaluation.py          # 单元测试  
    │   ├── example_usage.py            # 使用示例  
    │   └── README.md                   # 模块文档  
    └── streamlit_app/                  # Streamlit Web应用模块  
        ├── streamlit_app_v2.py         # 简化版Web应用（无模型比较）  
        ├── streamlit_app.py            # 原版Web应用  
        ├── test_streamlit_app.py       # 单元测试  
        ├── run_app.sh                  # Linux/macOS启动脚本  
        ├── run_app.bat                 # Windows启动脚本  
        ├── STREAMLIT_USAGE.md          # 使用指南  
        └──STREAMLIT_PROJECT_SUMMARY.md # 项目总结  
```




