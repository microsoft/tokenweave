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
from .tokenweave_utils import (load_config, tokenweave_with_fuse_only, tokenweave_overlap)
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
    split_size: Optional[int] = None,
    num_actual_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Forward pass for the attention layer with optional TokenWeave mode.
    """
    # ----------------------------------------
    # Arguments:
    # - positions: Tensor containing positional indices for rotary embeddings.
    #              Shape: [num_tokens]
    #
    # - hidden_states: Input tensor containing embeddings to be processed by the attention mechanism.
    #                  Shape: [num_tokens, hidden_dim]
    #
    # - split_id: Optional identifier (int): 0 or 1 — 0 for the first split batch, 1 for the second.
    #             Relevant only in TokenWeave mode.
    #
    # - split_size: Optional identifier (int): Number of tokens in the first split batch.
    #             Relevant only in TokenWeave mode.
    #
    # - num_actual_tokens: The number of tokens used to exclude padding or non-real tokens in TokenWeave mode.
    #
    # Returns:
    # - Updated hidden_states tensor after attention and output projection.
    # ----------------------------------------
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    if split_id is not None and split_size is not None:
        # TokenWeave Mode
        assert num_actual_tokens is not None
        self.rotary_emb(positions, q[:num_actual_tokens], k[:num_actual_tokens])
        attn_output = self.attn(q, k, v, split_id, split_size)
    else:
        # Default Mode
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
    # inplace + no all reduce
    self.o_proj(attn_output, hidden_states,
                            is_tokenweave=True)
    return hidden_states
```
The attention forward pass is adapted to handle splits of tokens, performs the final projection in-place on the `hidden_states` buffer, and does not call `all-reduce` here.

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
def forward_with_fuse_only(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    symm_mem_hdl: Any,
    layer_id: int,
    rank: int = 0,
    world_size: int = 1,
    next_layer_norm: RMSNorm = None,
    actual_tokens: int = None,
    num_tokens_padded: int = None,
    MAX_CTAS_ATTN: int = 16,
    MAX_CTAS_MLP: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with fused all-reduce + RMSNorm + residual add only (no TokenWeave overlap).

    Args:
        positions (torch.Tensor): Positional encoding indices.
        hidden_states (torch.Tensor): Input hidden states.
        residual (Optional[torch.Tensor]): Optional Residual.
        symm_mem_hdl (Any): Symmetric memory handle for all-reduce.
        layer_id (int): Current layer index.
        rank (int): Local process rank.
        world_size (int): Total number of distributed processes.
        next_layer_norm (RMSNorm): LayerNorm for the next block.
        actual_tokens (int): Number of valid tokens.
        num_tokens_padded (int): Padding length (multiple of world_size).
        MAX_CTAS_ATTN (int): Max CTAs for attention norm kernel.
        MAX_CTAS_MLP (int): Max CTAs for MLP norm kernel.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated hidden_states and residual.
    """
    assert actual_tokens is not None, "actual_tokens must be provided"
    assert num_tokens_padded is not None, "num_tokens_padded must be set"
    assert next_layer_norm is not None, "next_layer_norm must be provided"

    return tokenweave_with_fuse_only(
            self,
            *(
                positions,
                hidden_states,
                residual,
                symm_mem_hdl,
                layer_id,
                rank,
                world_size,
                next_layer_norm,
                actual_tokens,
                num_tokens_padded,
                MAX_CTAS_ATTN,
                MAX_CTAS_MLP,
                self.mlp.forward,
            )
        )

def forward_tokenweave(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    symm_mem_hdl: Any,
    layer_id: int,
    end_layer: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    current_stream: torch.cuda.Stream = None,
    copy_stream: torch.cuda.Stream = None,
    next_layer_norm: RMSNorm = None,
    split_size: int = None,
    actual_tokens: int = None,
    num_tokens_padded: int = None,
    MAX_CTAS_ATTN: int = 16,
    MAX_CTAS_MLP: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs forward pass of a transformer block using TokenWeave overlap strategy.
    Processes two token splits (interleaved) across GPUs with communication-compute overlap.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated hidden_states and residual tensors.
    """
    assert split_size is not None and actual_tokens is not None, "split_size and actual_tokens are required"
    assert current_stream is not None and copy_stream is not None, "CUDA streams must be provided"
    assert next_layer_norm is not None, "next_layer_norm must be provided"
    
    return tokenweave_overlap(
        self,
        * (
            positions,
            hidden_states,
            residual,
            symm_mem_hdl,
            layer_id,
            end_layer,
            rank,
            world_size,
            current_stream,
            copy_stream,
            next_layer_norm,
            split_size,
            actual_tokens,
            num_tokens_padded,
            MAX_CTAS_ATTN,
            MAX_CTAS_MLP,
            self.mlp.forward,
        )
    )
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
self.SPLIT_OFFSET = 0
```
TokenWeave uses symmetric memory and requires the storage of some metadata.

*def get_input_embeddings*

```py
# Original Implementation
def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.embed_tokens(input_ids)

# TokenWeave Implementation
def get_input_embeddings(
    self, 
    input_ids: torch.Tensor, 
    output_buffer: torch.Tensor, 
    is_tokenweave: Optional[bool] = False, 
    split_size: Optional[int] = None) -> torch.Tensor:
    return self.embed_tokens(
        input_ids, 
        output_parallel=output_buffer, 
        use_pytorch_all_reduce=False, 
        is_overlap=is_tokenweave, 
        symm_mem_hdl=self.symm_mem_hdl, 
        split_size=split_size)
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
    # TokenWeave is enabled when num_tokens >= 1024
    # This can be adjusted based on the model, world size and other factors.
    is_tokenweave = num_tokens >= 1024 
    tokenweave_split_size = None
    if is_tokenweave:
        # Load the tokenweave config based on the number of tokens
        closest_len = min(self.config_data.keys(), key=lambda k: abs(k - num_tokens))
        tokenweave_config = self.config_data[closest_len]
        self.MAX_CTAS_ATTN = tokenweave_config["attention_ctas"]
        self.MAX_CTAS_MLP = tokenweave_config["mlp_ctas"]
        self.SPLIT_OFFSET = tokenweave_config["split_offset"]
        tokenweave_split_size = (((num_tokens + 255) & ~255) // 2 + self.SPLIT_OFFSET) if is_tokenweave else None
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            self.buff = self.staging_buffer[:inputs_embeds.shape[0]]
            hidden_states = inputs_embeds
        else:
            self.buff = self.staging_buffer[:input_ids.shape[0]]
            hidden_states = self.get_input_embeddings(input_ids, self.buff, is_tokenweave, tokenweave_split_size)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    if not is_tokenweave: # with fuse only
        num_tokens_padded = (num_tokens + world_size - 1) // world_size * world_size
        hidden_states = self.staging_buffer[:num_tokens_padded]
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_id]
            next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
            hidden_states, residual = layer.forward_with_fuse_only(
                positions, 
                hidden_states, 
                residual, 
                self.symm_mem_hdl, 
                layer_id,
                # end_layer is not used in with fuse only flow
                rank,
                world_size,
                # current_stream is not used in with fuse only flow
                # copy_stream is not used in with fuse only flow
                next_layer_norm,
                # tokenweave_split_size is not used in with fuse only flow
                num_tokens,
                num_tokens_padded,
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
    num_tokens_padded = (num_tokens + 255) & ~255
    hidden_states = self.staging_buffer[:num_tokens_padded]

    for layer_id in range(self.start_layer, self.end_layer):
        layer = self.layers[layer_id]
        next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
        hidden_states, residual = layer.forward_tokenweave(positions, 
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
                                        tokenweave_split_size,
                                        num_tokens,
                                        num_tokens_padded,
                                        self.MAX_CTAS_ATTN,
                                        self.MAX_CTAS_MLP,
                                        )
    return hidden_states[:num_tokens]
```

Main forward pass that decides whether to use the default or TokenWeave path based on the number of tokens.
It additionally calls the appropriate decoder layer forward pass with the correct arguments