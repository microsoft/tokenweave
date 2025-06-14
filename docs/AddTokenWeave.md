## Summary of Changes to models/llama.py for TokenWeave Integration

### Import and Configuration
```py
# Original Implementation
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
# TokenWeave Implementation
# --- TokenWeave Specific Imports ---
import torch.distributed._symmetric_memory as symm_mem
import os
from vllm.distributed.triton_comm.triton_comm import (
    multimem_all_reduce, multimem_reduce_scatter, multimem_all_gather, multimem_all_gather_async)
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank, get_device_group,
                              get_tensor_model_parallel_world_size,
                              pytorch_all_reduce)
import json
from functools import lru_cache
# --- TokenWeave Configuration Loader ---
@lru_cache(maxsize=None)
def load_config(config_path="tokenweave_configs/llama_config_8.json"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.normpath(os.path.join(base_dir, "..", "..", config_path))
    with open(full_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
```

### class LlamaMLP(nn.Module):
```py
# Original Implementation
def forward(self, x):
    x, _ = self.gate_up_proj(x)
    x = self.act_fn(x)
    x, _ = self.down_proj(x)
    return x

# TokenWeave Implementation
def forward(self, hidden_states):
    x, _ = self.gate_up_proj(hidden_states)
    x = self.act_fn(x)
    # in-place, no all-reduce
    self.down_proj(x, hidden_states, is_tokenweave=True)
    return hidden_states
```
TokenWeave performs an in-place update to `hidden_states` instead of returning a new tensor, and does not call `all-reduce` here.

### class LlamaAttention(nn.Module):
```py
# Original Implementation
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output

# TokenWeave Implementation
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    split_id: Optional[int] = None,
    chunk_size: Optional[int] = None,
    num_actual_tokens: Optional[int] = None,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    self.rotary_emb(positions, q[:num_actual_tokens], k[:num_actual_tokens])
    attn_output = self.attn(q, k, v, split_id, chunk_size)
    # in-place, no-allreduce
    self.o_proj(attn_output, hidden_states,
                            is_tokenweave=True)
    return hidden_states
```
The attention forward pass is adapted to handle chunks of tokens, performs the final projection in-place on the `hidden_states` buffer, and does not call `all-reduce` here.

### class LlamaDecoderLayer(nn.Module):
```py
# Original Implementation
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual)
    hidden_states = self.self_attn(positions=positions,
                                    hidden_states=hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual

# TokenWeave Implementation
def forward_default(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    symm_mem_hdl: Any,
    layer_id: int,
    rank: Optional[int] = 0,
    world_size: Optional[int] = 1,
    next_layer_norm: Optional[RMSNorm] = None,
    actual_tokens: Optional[int] = None,
    nearest_multiple_of_world_size: Optional[int] = None,
    MAX_CTAS_ATTN: Optional[int] = 16,
    MAX_CTAS_MLP: Optional[int] = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    An optimized, non-pipelined forward pass for low num_tokens.
    Uses fused All-Reduce plus add-rmsnorm kernels.
    """
    num_tokens_per_rank = nearest_multiple_of_world_size // world_size
    # Self Attention
    if residual is None:
        residual = torch.empty_like(hidden_states)
    if layer_id == 0: # First layer
        self.input_layernorm(hidden_states, out=residual)

    self.self_attn.forward_default(positions=positions,
                                    hidden_states=hidden_states[:actual_tokens])
    # Fused_AllReduce_plus_add_RMSNorm
    self.post_attention_layernorm(
        hidden_states[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank], 
        residual[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank],
        MAX_CTAS=min(MAX_CTAS_ATTN, num_tokens_per_rank),
        fused_ar=True,
        symm_mem_hdl=symm_mem_hdl,
        rank=rank,
        world_size=world_size,
        offset=rank * num_tokens_per_rank * hidden_states.shape[1] * hidden_states.element_size(),
    )
    self.mlp.forward_default(hidden_states[:actual_tokens])
    next_layer_norm(
        hidden_states[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank], 
        residual[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank],
        MAX_CTAS=min(MAX_CTAS_MLP, num_tokens_per_rank),
        fused_ar=True,
        symm_mem_hdl=symm_mem_hdl,
        rank=rank,
        world_size=world_size,
        offset=rank * num_tokens_per_rank * hidden_states.shape[1] * hidden_states.element_size(),
    )
    return hidden_states, residual

def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    symm_mem_hdl: Any,
    layer_id: int,
    end_layer: Optional[int] = None,
    rank: Optional[int] = 0,
    world_size: Optional[int] = 1,
    current_stream: Optional[torch.cuda.Stream] = None,
    copy_stream: Optional[torch.cuda.Stream] = None,
    next_layer_norm: Optional[RMSNorm] = None,
    chunk_size: Optional[int] = None,
    actual_tokens: Optional[int] = None,
    nearest_multiple_of_256: Optional[int] = None,
    MAX_CTAS_ATTN: Optional[int] = 16,
    MAX_CTAS_MLP: Optional[int] = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The main TokenWeave forward pass, which pipelines computation and communication.
    """
    offset_second = chunk_size * hidden_states.shape[1] * hidden_states.element_size()
    if residual is None:
        residual = torch.empty_like(hidden_states)
    
    # Split tensors for pipelining
    hidden_states_1, hidden_states_2 = hidden_states[:chunk_size], hidden_states[chunk_size:]
    residual_1, residual_2 = residual[:chunk_size], residual[chunk_size:]
    blpr_1, blpr_2 = chunk_size // world_size, hidden_states_2.shape[0] // world_size # bl_per_rank

    # --- Attention Block Pipelining ---
    if layer_id == 0: # First layer requires special handling
        hidden_states_1 = self.input_layernorm(hidden_states_1, out=residual_1) 
        multimem_reduce_scatter(
            hidden_states_2,
            symm_mem_hdl,
            offset_second,
            MAX_CTAS=8
        )
        self.input_layernorm(hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], out=residual_2[rank * blpr_2: (rank + 1) * blpr_2])
        symm_mem_hdl.barrier(channel=7)
        multimem_all_gather_async(
            hidden_states_2,
            symm_mem_hdl,
            offset_second,
            blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(), # nbytes_per_rank
            current_stream,
        )
        symm_mem_hdl.barrier(channel=9)
    else: # Subsequent layers overlap LayerNorm/All-Reduce with attention
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(current_stream)
            self.input_layernorm(
                hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], 
                residual_2[rank * blpr_2: (rank + 1) * blpr_2],
                MAX_CTAS=MAX_CTAS_ATTN,
                fused_ar=True,
                symm_mem_hdl=symm_mem_hdl,
                rank=rank,
                world_size=world_size,
                offset=offset_second +  rank * blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(),
            )
    with torch.cuda.stream(current_stream):
        hidden_states_1 = self.self_attn(
            positions=positions[:chunk_size],
            hidden_states=hidden_states_1,
            split_id=0,
            chunk_size=chunk_size,
            num_actual_tokens=chunk_size,
        )
        current_stream.wait_stream(copy_stream)

    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(current_stream)
        self.post_attention_layernorm(
            hidden_states_1[rank * blpr_1: (rank + 1) * blpr_1], 
            residual_1[rank * blpr_1: (rank + 1) * blpr_1],
            MAX_CTAS=MAX_CTAS_ATTN,
            fused_ar=True,
            symm_mem_hdl=symm_mem_hdl,
            rank=rank,
            world_size=world_size,
            offset=rank * blpr_1 * hidden_states_1.shape[1] * hidden_states_1.element_size(),
        )
    with torch.cuda.stream(current_stream):
        hidden_states_2 = self.self_attn(
            positions=positions[chunk_size:],
            hidden_states=hidden_states_2,
            split_id=1,
            chunk_size=chunk_size,
            num_actual_tokens=actual_tokens - chunk_size,
        )
        current_stream.wait_stream(copy_stream)
    
    # --- MLP Block Pipelining ---
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(current_stream)
        self.post_attention_layernorm(
            hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], 
            residual_2[rank * blpr_2: (rank + 1) * blpr_2],
            MAX_CTAS=MAX_CTAS_MLP,
            fused_ar=True,
            symm_mem_hdl=symm_mem_hdl,
            rank=rank,
            world_size=world_size,
            offset=offset_second +  rank * blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(),
        )
    
    with torch.cuda.stream(current_stream):
        hidden_states_1 = self.mlp(hidden_states_1)
        current_stream.wait_stream(copy_stream)

    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(current_stream)
        next_layer_norm(
            hidden_states_1[rank * blpr_1: (rank + 1) * blpr_1], 
            residual_1[rank * blpr_1: (rank + 1) * blpr_1],
            MAX_CTAS=MAX_CTAS_MLP,
            fused_ar=True,
            symm_mem_hdl=symm_mem_hdl,
            rank=rank,
            world_size=world_size,
            offset=rank * blpr_1 * hidden_states_1.shape[1] * hidden_states_1.element_size(),
        )
    with torch.cuda.stream(current_stream):
        hidden_states_2 = self.mlp(hidden_states_2)
        current_stream.wait_stream(copy_stream)        
    if layer_id == end_layer - 1:
        next_layer_norm(
            hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], 
            residual_2[rank * blpr_2: (rank + 1) * blpr_2],
            MAX_CTAS=16 if actual_tokens < 16384 else 32,
            fused_ar=True,
            symm_mem_hdl=symm_mem_hdl,
            rank=rank,
            world_size=world_size,
            offset=offset_second +  rank * blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(),
        )
    return hidden_states, residual
```

TokenWeave uses two different forward pass implementations based on `num_tokens`. If `num_tokens` is low (i.e., less than 1K), TokenWeave doesn't perform overlap but does use a fused all-reduce plus add-RMSNorm kernel. If `num_tokens` is high, it uses the fully overlapped version.
### class LlamaModel(nn.Module):

*def __init__*
```py
# Original Implementation

# TokenWeave Implementation
CHUNK_SIZE = vllm_config.scheduler_config.max_num_batched_tokens + 512
self.staging_buffer = symm_mem.empty((CHUNK_SIZE, config.hidden_size),
                                    dtype=vllm_config.model_config.dtype,
                                    device="cuda")
self.symm_mem_hdl = symm_mem.rendezvous(self.staging_buffer, get_device_group())
self.current_stream = torch.cuda.current_stream()
self.copy_stream = torch.cuda.Stream(priority=-1)
self.buff = None

world_size = get_tensor_model_parallel_world_size()
self.config_data = load_config(f"tokenweave_configs/llama_config_{world_size}.json")
self.MAX_CTAS_ATTN = 16
self.MAX_CTAS_MLP = 16
self.CHUNK_OFFSET = 0
```
TokenWeave uses symmetric memory and requires the storage of some metadata.

*def get_input_embeddings*

```py
# Original Implementation
def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.embed_tokens(input_ids)

# TokenWeave Implementation
def get_input_embeddings(self, input_ids: torch.Tensor, output_buffer: torch.Tensor, 
                is_tokenweave: Optional[bool] = False, chunk_size: Optional[int] = None) -> torch.Tensor:
    return self.embed_tokens(input_ids, output_parallel=output_buffer, use_pytorch_all_reduce=False, is_overlap=is_tokenweave, symm_mem_hdl=self.symm_mem_hdl, chunk_size=chunk_size)
```
Modified the code to support split embedding computation.

*def forward*
```py
# Original Implementation
def forward(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors],
    inputs_embeds: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, IntermediateTensors]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    for layer in self.layers[self.start_layer:self.end_layer]:
        hidden_states, residual = layer(positions, hidden_states, residual)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states

# TokenWeave Implementation
def forward(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors],
    inputs_embeds: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, IntermediateTensors]:
    rank, world_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    num_tokens = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    is_tokenweave = num_tokens >= 1024 # spliting requires at least 1024 tokens
    tokenweave_chunk_size = None
    if is_tokenweave:
        # Load the tokenweave config based on the number of tokens
        closest_len = min(self.config_data.keys(), key=lambda k: abs(k - num_tokens))
        tokenweave_config = self.config_data[closest_len]
        self.MAX_CTAS_ATTN = tokenweave_config["attention_ctas"]
        self.MAX_CTAS_MLP = tokenweave_config["mlp_ctas"]
        self.CHUNK_OFFSET = tokenweave_config["chunk_offset"]
        tokenweave_chunk_size = (((num_tokens + 255) & ~255) // 2 + self.CHUNK_OFFSET) if is_tokenweave else None
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            self.buff = self.staging_buffer[:inputs_embeds.shape[0]]
            hidden_states = inputs_embeds
        else:
            self.buff = self.staging_buffer[:input_ids.shape[0]]
            hidden_states = self.get_input_embeddings(input_ids, self.buff, is_tokenweave, tokenweave_chunk_size)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    if not is_tokenweave: # default
        nearest_multiple_of_world_size = (num_tokens + world_size - 1) // world_size * world_size
        hidden_states = self.staging_buffer[:nearest_multiple_of_world_size]
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_id]
            next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
            hidden_states, residual = layer.forward_default(
                positions, 
                hidden_states, 
                residual, 
                self.symm_mem_hdl, 
                layer_id,
                # end_layer is not used in default flow
                rank,
                world_size,
                # current_stream is not used in default flow
                # copy_stream is not used in default flow
                next_layer_norm,
                # tokenweave_chunk_size is not used in default flow
                num_tokens,
                nearest_multiple_of_world_size,
                self.MAX_CTAS_ATTN,
                self.MAX_CTAS_MLP,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        return hidden_states[:num_tokens]
    # TokenWeave
    nearest_multiple_of_256 = (num_tokens + 255) & ~255
    hidden_states = self.staging_buffer[:nearest_multiple_of_256]

    for layer_id in range(self.start_layer, self.end_layer):
        layer = self.layers[layer_id]
        next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
        hidden_states, residual = layer(positions, 
                                        hidden_states, 
                                        residual, 
                                        self.symm_mem_hdl, 
                                        layer_id, 
                                        self.end_layer, 
                                        rank, 
                                        world_size,
                                        self.current_stream,
                                        self.copy_stream,
                                        next_layer_norm,
                                        tokenweave_chunk_size,
                                        num_tokens,
                                        nearest_multiple_of_256,
                                        self.MAX_CTAS_ATTN,
                                        self.MAX_CTAS_MLP,
                                        )
    return hidden_states[:num_tokens]
```

Main forward pass that decides whether to use the default or TokenWeave path based on the number of tokens.
It additionally calls the appropriate decoder layer forward pass with the correct arguments