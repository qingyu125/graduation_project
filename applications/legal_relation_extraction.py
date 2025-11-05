import streamlit as st
from models.base.code_llama_wrapper import CodeLlamaWrapper
from models.extraction.text_to_code import TextToCodeConverter
from models.extraction.code_parser import CodeParser
from models.knowledge.kg_injector import KnowledgeGraphInjector
from models.reasoning.code_chain import CodeChainReasoner

def main():
    st.title("法律文档关系抽取与推理系统")
    st.write("基于CodeLlama-7B的代码增强型文本推理系统，支持实体、事件、关系抽取与可解释推理")
    
    # 侧边栏配置
    with st.sidebar:
        lora_path = st.text_input("LoRA权重路径", "./outputs/lora_weights")
        kg_path = st.text_input("知识图谱路径", "./data/kg/legal_kg.json")
        if st.button("加载模型"):
            with st.spinner("加载模型中..."):
                # 初始化模型组件
                st.session_state["model_wrapper"] = CodeLlamaWrapper(lora_path=lora_path)
                st.session_state["text_to_code"] = TextToCodeConverter(st.session_state["model_wrapper"])
                st.session_state["code_parser"] = CodeParser()
                st.session_state["kg_injector"] = KnowledgeGraphInjector(kg_path)
                st.session_state["reasoner"] = CodeChainReasoner(st.session_state["model_wrapper"])
                st.success("模型加载完成！")
    
    # 文本输入
    text_input = st.text_area("输入法律文本", height=150, placeholder="例如：甲公司与乙公司于2024年签约，合作开发AI技术...")
    
    if st.button("执行抽取与推理") and "model_wrapper" in st.session_state:
        with st.spinner("处理中..."):
            # 1. 文本→伪代码
            pseudo_code = st.session_state["text_to_code"].convert(text_input)
            st.subheader("1. 伪代码转换结果")
            st.code(pseudo_code, language="python")
            
            # 2. 解析要素
            elements = st.session_state["code_parser"].parse(pseudo_code)
            st.subheader("2. 结构化要素抽取结果")
            st.write("实体：", elements["entities"])
            st.write("事件：", elements["events"])
            st.write("关系：", elements["relations"])
            
            # 3. 知识验证
            verified_elements = st.session_state["kg_injector"].verify_elements(elements)
            st.subheader("3. 知识验证结果")
            st.write("验证后的关系：", verified_elements["relations"])
            
            # 4. 推理
            if verified_elements["relations"]:
                rel = verified_elements["relations"][0]
                reasoning_result = st.session_state["reasoner"].reason(
                    verified_elements,
                    rel["head"],
                    rel["tail"],
                    rel["type"]
                )
                st.subheader("4. 推理结论")
                st.write(f"结论：{'成立' if reasoning_result['conclusion'] else '不成立'}")
                st.write("推理依据：", reasoning_result["reason"])
                with st.expander("查看代码推理链"):
                    st.code(reasoning_result["code_chain"], language="python")

if __name__ == "__main__":
    main()