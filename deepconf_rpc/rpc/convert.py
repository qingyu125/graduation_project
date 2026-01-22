from importlib.metadata import version
import warnings
import transformers

from .llama_simple import LlamaRPCAttention
from .qwen2_simple import Qwen2RPCAttention


def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    # 支持更新的transformers版本
    version_list = ['4.45', '4.50', '4.57']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be fully tested with RPC. RPC works best with Transformers version {version_list}.")


def enable_rpc():
    check_version()

    # 对于新版本的transformers，直接修改模块中的类
    try:
        import transformers.models.llama.modeling_llama as llama_model
        import transformers.models.qwen2.modeling_qwen2 as qwen2_model
        
        # 直接替换LlamaAttention为我们的RPC版本
        original_llama_attention = llama_model.LlamaAttention
        llama_model.LlamaAttention = LlamaRPCAttention
        
        # 直接替换Qwen2Attention为我们的RPC版本  
        original_qwen2_attention = qwen2_model.Qwen2Attention
        qwen2_model.Qwen2Attention = Qwen2RPCAttention
        
        print("✓ Successfully enabled RPC for Llama and Qwen2 models")
        
    except Exception as e:
        print(f"⚠ Warning: Could not enable RPC automatically: {e}")
        print("You may need to manually patch the attention classes in your model.")