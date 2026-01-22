## 简介
  本仓库为本科毕设代码实现，题目：《面向高效推理的要素抽取与应用算法设计与实现》  
## 整体代码结构  
```
/graduation_project/
├── 数据目录 data/
│   └── aime_2025.jsonl                       # AIME数学题数据集 (30题)
│
├── 核心代码 deepconf_rpc/
│   ├── wrapper.py                            # DeepThinkLLM (已修改支持RPC)
│   ├── outputs.py                            # 输出处理
│   ├── utils.py                              # 工具函数
│   └── rpc/                                  # RPC模块
│       ├── convert.py                        # RPC启用函数
│       ├── rpc_utils.py                      # RPC核心工具
│       ├── llama_simple.py                   # 兼容版Llama实现
│       └── qwen2_simple.py                   # 兼容版Qwen2实现
├── 要素抽取模块 element_extraction/
│   ├── demo_element_extraction.py            # 要素抽取启动函数
│   ├── element_analyzer.py                   # 要素分析
│   └── element_extractor.py                  # 要素抽取核心实现
│
└── 主运行脚本
    ├── debug_traces.py                       # debug
    ├── example_online.py                     # 原始非RPC版本
    ├── example_online_rpc.py                 # RPC版本
    ├── compare_single_question.py            # 单问题详细对比
    ├── example_compare_rpc.py                # 对比分析脚本
    └── run_rpc_comparison.py                 # 一键完整对比
```









