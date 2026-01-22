import warnings

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache

# perform qk calculation and get indices
# this version will not update in inference mode

def set_rpc_config(
    llm,
    P=1024,
    R=32,
    c=4,
    selectors='recent',
    aggregation='all',
    kernel_size=7, 
    pooling='avgpool',
    ):
    """
    Set RPC configuration for vLLM LLM object
    
    Args:
        llm: vLLM LLM object
        P, R, c, selectors, aggregation, kernel_size, pooling: RPC parameters
    """
    # Access the underlying model from vLLM LLM object
    # Try multiple possible paths to access the model
    model = None
    layers = None
    
    try:
        # Path 1: llm.llm_engine.model_executor.driver_worker.model_runner.model
        if hasattr(llm, 'llm_engine'):
            engine = llm.llm_engine
            if hasattr(engine, 'model_executor'):
                executor = engine.model_executor
                if hasattr(executor, 'driver_worker'):
                    worker = executor.driver_worker
                    if hasattr(worker, 'model_runner'):
                        runner = worker.model_runner
                        if hasattr(runner, 'model'):
                            model = runner.model
        
        # Path 2: llm.llm_engine.workers[0].model_runner.model
        if model is None and hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'workers'):
            if llm.llm_engine.workers:
                worker = llm.llm_engine.workers[0]
                if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                    model = worker.model_runner.model
        
        # Path 3: llm.llm_engine.model_executor.workers[0].model
        if model is None and hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
            executor = llm.llm_engine.model_executor
            if hasattr(executor, 'workers') and executor.workers:
                worker = executor.workers[0]
                if hasattr(worker, 'model'):
                    model = worker.model
        
        # Now access the model layers
        if model is not None:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
            elif hasattr(model, 'layers'):
                layers = model.layers
            elif hasattr(model, 'module') and hasattr(model.module, 'model') and hasattr(model.module.model, 'layers'):
                layers = model.module.model.layers
        
        if layers is not None:
            num_layers = len(layers)
            configured_count = 0
            
            # Configure each layer's RPC cluster
            for i in range(num_layers):
                if hasattr(layers[i], 'self_attn') and hasattr(layers[i].self_attn, 'kv_cluster'):
                    layers[i].self_attn.kv_cluster.P = P
                    layers[i].self_attn.kv_cluster.T = int(P/c)
                    layers[i].self_attn.kv_cluster.R = R
                    layers[i].self_attn.kv_cluster.selectors = selectors
                    layers[i].self_attn.kv_cluster.aggregation = aggregation
                    layers[i].self_attn.kv_cluster.kernel_size = kernel_size
                    layers[i].self_attn.kv_cluster.pooling = pooling
                    configured_count += 1
                else:
                    print(f"Warning: Layer {i} does not have RPC cluster initialized. Make sure enable_rpc() was called before model initialization.")
            
            if configured_count > 0:
                print(f"[RPC Config][P={P}, R={R}, c={c}][selectors={selectors}, aggregation={aggregation}]", flush=True)
                print(f"Successfully configured {configured_count}/{num_layers} layers")
                return
        
    except Exception as e:
        print(f"Error in primary access method: {e}")
    
    # If primary methods failed, try fallback methods
    if layers is None:
        print("Primary access methods failed. Attempting alternative access methods...")
        
        # Try multiple alternative paths
        model = None
        layers = None
        
        # Method 1: Try via workers
        try:
            if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'workers'):
                worker = llm.llm_engine.workers[0] if llm.llm_engine.workers else None
                if worker and hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                    model = worker.model_runner.model
        except Exception:
            pass
        
        # Method 2: Try direct access to llm_engine
        if model is None:
            try:
                if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
                    executor = llm.llm_engine.model_executor
                    if hasattr(executor, 'workers') and executor.workers:
                        worker = executor.workers[0]
                        if hasattr(worker, 'model') and hasattr(worker.model, 'model'):
                            model = worker.model
            except Exception:
                pass
        
        # Method 3: Try accessing via cache_engine
        if model is None:
            try:
                if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'cache_engine'):
                    # Some vLLM versions store model here
                    pass
            except Exception:
                pass
        
        # If we found a model, try to get layers
        if model is not None:
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    layers = model.model.layers
                elif hasattr(model, 'layers'):
                    layers = model.layers
                elif hasattr(model, 'module') and hasattr(model.module, 'model') and hasattr(model.module.model, 'layers'):
                    layers = model.module.model.layers
            except Exception:
                pass
        
        # Configure layers if found
        if layers is not None:
            num_layers = len(layers)
            configured_count = 0
            for i in range(num_layers):
                if hasattr(layers[i], 'self_attn') and hasattr(layers[i].self_attn, 'kv_cluster'):
                    layers[i].self_attn.kv_cluster.P = P
                    layers[i].self_attn.kv_cluster.T = int(P/c)
                    layers[i].self_attn.kv_cluster.R = R
                    layers[i].self_attn.kv_cluster.selectors = selectors
                    layers[i].self_attn.kv_cluster.aggregation = aggregation
                    layers[i].self_attn.kv_cluster.kernel_size = kernel_size
                    layers[i].self_attn.kv_cluster.pooling = pooling
                    configured_count += 1
            
            if configured_count > 0:
                print(f"[RPC Config][P={P}, R={R}, c={c}][selectors={selectors}, aggregation={aggregation}]", flush=True)
                print(f"Successfully configured {configured_count}/{num_layers} layers")
                return
            else:
                print("Warning: Found layers but none have RPC cluster initialized")
        
        # If all methods failed, print debug info
        print("Failed to access model. Debug info:")
        print(f"  llm type: {type(llm)}")
        if hasattr(llm, 'llm_engine'):
            print(f"  llm_engine type: {type(llm.llm_engine)}")
            print(f"  llm_engine attributes: {[a for a in dir(llm.llm_engine) if not a.startswith('_')][:10]}")
        print("RPC configuration failed. Model may use default RPC values from initialization.")
        # Don't raise - allow execution to continue with default values

# Copied from transformers.models.llama.modeling_llama.repeat_kv for gqa_support
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RPCCluster():
    def __init__(self,
                 layer_idx = None, 
                 P=1024,
                 R=32,
                 c=4,
                 selectors='recent', # options prompt, new, recent
                 aggregation='all', # all, group, none
                 kernel_size=7, 
                 pooling='avgpool',
                 num_key_value_groups=1,
                 ):

        self.layer_idx = layer_idx

        # compression arguments
        self.P = P
        self.R = R
        self.c = c
        self.T = int(P/c)
        self.prompt_len = 0
        self.num_comp = 0

        self.kernel_size = kernel_size
        self.pooling = pooling

        self.selectors = selectors

        self.cached_prompt = None
        self.cached_recent = None

        # support gqa
        self.aggregation = aggregation
        self.num_key_value_groups = num_key_value_groups
        self.agg_func = 'mean'

        
    def cache_recent(self, current_query_states):
        if self.cached_recent is None:
            self.cached_recent = current_query_states
        else:
            self.cached_recent = torch.cat([self.cached_recent, current_query_states], dim=-2)

    def compress_kv(self, origin_key_states, origin_value_states, query_states):

        if self.selectors == 'recent' and self.cached_recent is not None:
            selectors = torch.cat([self.cached_recent, query_states], dim=-2)
            self.cached_recent = None # for next compress
        else:
            selectors = query_states

        # # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        
        bsz, num_heads, q_len, head_dim = selectors.shape

  
        attn_weights = torch.matmul(selectors, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        # no need to deal with attention mask

        attn_weights = nn.functional.softmax(attn_weights[:, :, :, self.prompt_len:-self.R], dim=-1, dtype=torch.float32).to(selectors.dtype)
        attn_weights_sum = attn_weights.sum(dim = -2)

        if self.aggregation == 'all':

            attn_weights_sum = attn_weights_sum.view(attn_weights_sum.shape[0], -1, 1, attn_weights_sum.shape[-1])
            if self.agg_func == 'max':
                attn_weights_sum = attn_weights_sum.max(dim=-3).values
            elif self.agg_func == 'mean':
                attn_weights_sum = attn_weights_sum.mean(dim=-3)
            else:
                raise ValueError('agg_func not supported')
        
        elif self.aggregation == 'group':

            attn_weights_sum = attn_weights_sum.view(attn_weights_sum.shape[0], -1, self.num_key_value_groups, attn_weights_sum.shape[-1])
            if self.agg_func == 'max':
                attn_weights_sum = attn_weights_sum.max(dim=-2).values
            elif self.agg_func == 'mean':
                attn_weights_sum = attn_weights_sum.mean(dim=-2)
            else:
                raise ValueError('agg_func not supported')

        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        indices = attn_cache.topk((self.num_comp + 1) * self.T, dim=-1, largest=True).indices.sort(dim=-1).values        
        indices = indices.unsqueeze(-1).expand(-1, origin_key_states.size(1), -1, head_dim)


        # support gqa
        if self.aggregation == 'all' or 'group':
            k_prompt = origin_key_states[:, :, :self.prompt_len, :]
            v_prompt = origin_value_states[:, :, :self.prompt_len, :]

            k_past_compress = origin_key_states[:, :, self.prompt_len:-self.R, :].gather(dim = 2, index = indices)
            v_past_compress = origin_value_states[:, :, self.prompt_len:-self.R, :].gather(dim = 2, index = indices)
            
            k_cur = origin_key_states[:, :, -self.R:, :]
            v_cur = origin_value_states[:, :, -self.R:, :]

        else:
            k_prompt = key_states[:, :, :self.prompt_len, :]
            v_prompt = value_states[:, :, :self.prompt_len, :]

            k_past_compress = key_states[:, :, self.prompt_len:-self.R, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, self.prompt_len:-self.R, :].gather(dim = 2, index = indices)

            k_cur = key_states[:, :, -self.R:, :]
            v_cur = value_states[:, :, -self.R:, :]

        key_states = torch.cat([k_prompt, k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_prompt, v_past_compress, v_cur], dim = 2)


        return key_states, value_states
   

def init_rpc(self):

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = RPCCluster(
            layer_idx = self.layer_idx,
            P = 1024,
            R = 32,
            c = 4,
            selectors='recent', # options: new, recent
            aggregation='all', # options: all, group, none
            kernel_size = 7,
            pooling = 'avgpool',
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            )